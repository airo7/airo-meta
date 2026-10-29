import asyncio
import math

import cv2
import numpy as np
import pytest
from test_loading import Feeling

from airo import meta_api
from airo.fleet import Fleet, crew, footprint
from airo.life import Hazard
from airo.mjsim import fleet
from airo.site import FACTORY
from airo.workers import Step


def run(n, commands, level="calm", until=200.0, hook=None):
    """A fleet on the factory with life at level, the commands routed; hook(fleet, world) runs every tick. Returns
    the fleet, the world, and per tick the gaps between car footprints and from them to people."""
    rigs = fleet(FACTORY, n, log_dir=None)
    seg = meta_api.segment
    meta_api.segment = rigs[0].sam
    f = Fleet(FACTORY, crew(FACTORY, rigs, Feeling))
    world = rigs[0].world
    f.route(level)
    gaps, people = [], []

    async def go():
        jobs = [asyncio.create_task(r.mission.run()) for r in f.robots]
        for text in commands:
            f.route(text)
        await asyncio.sleep(0.1)
        while world.t < until and any(r.mission.busy for r in f.robots):
            if hook:
                hook(f, world)
            cars = [footprint(r.pose, pad=0.0) for r in rigs]
            gaps.extend(_gap(a, b) for i, a in enumerate(cars) for b in cars[i + 1:])
            people.extend(_gap(a, np.array([(w.x, w.y)])) for a in cars for w in world.life.workers)
            await asyncio.sleep(0)
        for j in jobs:
            j.cancel()

    try:
        asyncio.run(go())
    finally:
        meta_api.segment = seg
    return f, world, gaps, people


def _gap(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between two convex outlines on the floor (a point is an outline too), 0 if they overlap."""
    inside = lambda poly, pts: any(cv2.pointPolygonTest(poly.astype(np.float32), tuple(map(float, q)), False) >= 0
                                   for q in pts)
    if len(a) > 2 and inside(a, b) or len(b) > 2 and inside(b, a):
        return 0.0
    edges = lambda poly: zip(poly, np.roll(poly, -1, axis=0)) if len(poly) > 1 else [(poly[0], poly[0])]
    return min(_seg(p, s0, s1) for poly, other in ((a, b), (b, a)) for s0, s1 in edges(poly) for p in other)


def _seg(p, a, b) -> float:
    d = b - a
    t = np.clip((p - a) @ d / max(d @ d, 1e-12), 0, 1)
    return float(np.linalg.norm(a + t * d - p))


def delivered(r, station: str) -> bool:
    title = station.upper() if len(station) <= 2 else station.capitalize()
    return any(n["text"].startswith("Delivered a ") and n["text"].endswith(f"{title}.") for n in r.mission.notes)


def test_the_factory_builds_with_and_without_the_menagerie_models(models):
    rigs = fleet(FACTORY, 3, log_dir=None)
    world = rigs[0].world
    f = Fleet(FACTORY, crew(FACTORY, rigs, Feeling))
    movers = f.state()["site"]["movers"]
    kinds = [m["kind"] for m in movers]
    assert kinds == (["human"] * 3 if models == "menagerie" else []) + ["forklift"]
    assert all(len(m["pose"]) == 3 and m["id"] for m in movers)
    world._physics(2.0)
    assert world.frame().shape == (720, 1280, 3)
    if models == "procedural":
        assert f.route("send a worker across").startswith("no workers")
    assert f.route("busy") == "the floor is busy" and world.life.level == "busy"


def test_a_worker_crossing_its_lane_stops_the_car_until_clear_then_it_delivers(models):
    if models == "procedural":
        pytest.skip("no workers without the Unitree G1 model")
    seen = {"blocked": [], "set": False, "gap": True}

    def cross(f, world):
        r, w = f.robots[0], world.life.walker
        p = r.rig.true_pose
        if not seen["set"] and r.mission.status == "delivering" and r.rig.pose and \
                math.dist(p[:2], (FACTORY.pick.x, FACTORY.pick.y)) > 0.35:
            c, s = math.cos(p[2]), math.sin(p[2])
            ahead = p[:2] + 0.6 * np.array([c, s])
            w.x, w.y = ahead + 0.35 * np.array([-s, c])
            w.heading = math.atan2(-c, s)
            w.plan = [Step("walk", [tuple(ahead - 0.35 * np.array([-s, c]))])]
            seen["set"] = True
        blocked = r.guard.blocked_by
        if isinstance(blocked, Hazard) and blocked.person:
            if not seen["blocked"] or seen["gap"]:
                seen["blocked"].append([])
            seen["blocked"][-1].append(p.copy())
            seen["now"] = f._now(r)
        seen["gap"] = not (isinstance(blocked, Hazard) and blocked.person)

    f, world, _, people = run(1, ["airo 1 deliver to assembly"], hook=cross)
    r = f.robots[0]
    longest = max(seen["blocked"], key=len)
    assert seen["set"] and len(longest) > 10
    # the commands reach the motors 0.17 s late, so it coasts a few centimetres first, then stands
    stops = np.array(longest)[len(longest) // 2:]
    assert np.ptp(stops[:, :2], axis=0).max() < 0.02
    assert seen["now"] == "Waiting: Worker 1 is crossing its path."
    assert delivered(r, "assembly") and min(people) > 0.12


def test_a_dropped_pallet_is_seen_on_the_next_scan_and_planned_around():
    state = {}

    def drop(f, world):
        r = f.robots[0]
        if "spot" not in state and r.mission.status == "delivering" and len(r.mission.path) > 1 and \
                math.dist(r.rig.true_pose[:2], (FACTORY.pick.x, FACTORY.pick.y)) > 0.3:
            state["path"] = [tuple(q) for q in r.mission.path]
            state["said"] = f.route("drop a pallet")
            d = next(d for d in world.life.drops if d.placed)
            state["spot"], state["drop"] = d.thing.xy, d
        if "spot" in state:
            state.setdefault("near", []).append(math.dist(r.rig.true_pose[:2], state["spot"]))
            if r.mission.path and [tuple(q) for q in r.mission.path] != state["path"] and "replanned" not in state:
                state["replanned"] = [tuple(q) for q in r.mission.path]
                state["polygons"] = [lb for lb, _ in r.mission.scene.polygons]

    f, world, _, _ = run(1, ["airo 1 deliver to qa"], hook=drop)
    r = f.robots[0]
    assert state["said"].startswith("a pallet dropped in the aisle")
    path = np.array(state["path"])
    assert min(np.linalg.norm(path - state["spot"], axis=1).min(), _to(path, state["spot"])) < 0.15
    new = np.array(state["replanned"])
    assert "obstacle" in state["polygons"] and _to(new, state["spot"]) > 0.2
    assert min(state["near"]) > 0.2 and delivered(r, "qa")


def _to(path: np.ndarray, p) -> float:
    a, b = path[:-1], path[1:]
    d = b - a
    t = np.clip(((p - a) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-9), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * d - p, axis=1).min())


def test_three_robots_deliver_on_a_busy_floor_without_collisions_or_deadlock(models):
    f, world, gaps, people = run(3, ["airo 1 deliver to assembly", "airo 2 bring a heavy tote to packing",
                                     "airo 3 deliver to dock", "send a worker across", "drop a carton"],
                                 level="busy", until=260)
    for r, station in zip(f.robots, ("assembly", "packing", "dock")):
        assert delivered(r, station), r.name
        assert not r.mission.busy and math.dist(r.rig.true_pose[:2], r.mission.dock[:2]) < 0.06
    # no car touches another, and none touches a person (a G1 is about 5 cm across at this scale)
    assert min(gaps) > 0.0 and (not people or min(people) > 0.03)
    assert all(not line.robots for line in world.cell.lines.values())

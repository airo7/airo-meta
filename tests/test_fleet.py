import asyncio
import math
from dataclasses import replace

import numpy as np
import pytest

from airo.body import Body, ensemble
from airo.fleet import Fleet, Guard, crew, parse_robot
from airo.mjsim import fleet
from airo.orbit import Orbit
from airo.rig import DT
from airo.scientist import Scientist
from airo.site import FACTORY
from airo.voice import Intent, parse_intent

# The twin's body as the learning gate finds it; the deliveries below test the fleet, not the fit.
TWIN = Body(g_l=0.00312, g_r=0.00305, dz=27.4, eta=0.303, tau=0.213, lag=0.128, kappa=-0.008)


@pytest.fixture(scope="module")
def rigs():
    return fleet(FACTORY, 3)


def steps(rigs, cmds, n):
    async def run():
        for _ in range(n):
            await asyncio.gather(*(r.step(*c) for r, c in zip(rigs, cmds)))
    asyncio.run(run())


def mass(rig) -> float:
    return float(rig.m.body_subtreemass[rig.root])


def test_factory_compiles_and_renders(rigs):
    world = rigs[0].world
    assert [r.prefix for r in rigs] == ["airo1/", "airo2/", "airo3/"] and world.m.nu == 12
    assert world.frame().shape == (720, 1280, 3) and rigs[1].view(320, 180).shape == (180, 320, 3)
    frame = world.frame()
    for concept in [*FACTORY.stations.values(), *FACTORY.totes, "box"]:
        assert len(asyncio.run(world.sam(frame, concept))) == 1, concept
    assert FACTORY.station_xy(world.objects).keys() == {"assembly", "qa", "packing", "dock"}


def test_three_cars_step_together():
    rigs = fleet(FACTORY, 3)
    starts = [r.true_pose for r in rigs]
    steps(rigs, [(120, 120), (0, 0), (-100, 100)], 30)
    world = rigs[0].world
    assert world.t == pytest.approx(30 * DT) and all(r.t == pytest.approx(world.t) for r in rigs)
    moved = [np.linalg.norm(r.true_pose[:2] - s[:2]) for r, s in zip(rigs, starts)]
    turned = [abs(math.remainder(r.true_pose[2] - s[2], math.tau)) for r, s in zip(rigs, starts)]
    assert moved[0] > 0.15 and moved[1] < 0.002 and turned[2] > 0.5 and moved[2] < 0.03
    assert all(len(r.history[0]) == 30 for r in rigs)


def test_body_changes_touch_only_their_car(rigs):
    before = [mass(r) for r in rigs]
    poses = [r.true_pose for r in rigs]
    rigs[1].add_weight(0.4)
    assert mass(rigs[1]) == pytest.approx(before[1] + 0.4)
    assert [mass(rigs[0]), mass(rigs[2])] == pytest.approx([before[0], before[2]])
    assert all(np.allclose(r.true_pose, p) for r, p in zip(rigs, poses))
    rigs[2].attach_tow()
    rigs[2].flip_motor("l")
    assert rigs[1].changes == ["weight"] and rigs[2].changes == ["tow", "flip"] and rigs[0].changes == []
    assert rigs[1].remove_weight() and rigs[2].detach_tow()
    rigs[2].flip_motor("l")
    assert [mass(r) for r in rigs] == pytest.approx(before) and all(r.changes == [] for r in rigs)
    steps(rigs, [(0, 0)] * 3, 2)


def test_spill_changes_friction_only_inside_the_zone():
    x0, y0, x1, y1 = FACTORY.spill
    site = replace(FACTORY, starts=(((x0 + x1) / 2, (y0 + y1) / 2, 0.0), (0.6, 1.0, 0.0)))

    def turns(spill: bool):
        rigs = fleet(site, 2, noise=(0, 0))
        if spill:
            rigs[0].world.spill()
        steps(rigs, [(-100, 100)] * 2, 30)
        return [abs(np.ptp(np.unwrap(r.segment().pose[:, 2]))) for r in rigs], rigs

    (dry_in, dry_out), _ = turns(False)
    (wet_in, wet_out), rigs = turns(True)
    assert wet_in > 1.3 * dry_in and wet_out == pytest.approx(dry_out, rel=0.02)
    world = rigs[0].world
    assert world.zone == FACTORY.spill and "slippery" in rigs[1].changes
    world.clean()
    assert world.zone is None and "spill" not in [world.m.geom(i).name for i in range(world.m.ngeom)]


def test_orbit_follows_a_chosen_car_and_names_whose_part_it_is(rigs):
    orbit = Orbit(rigs[0], (640, 360))
    orbit.follow(rigs[1])
    orbit.set("wheel")
    orbit._t = None
    cam = orbit.update()
    rigs[1].view(640, 360, cam)
    assert np.linalg.norm(np.array(cam.lookat[:2]) - rigs[1].true_pose[:2]) < 0.15
    assert orbit.handle({"pick": [0.5, 0.5]})["part"]["name"] == "AIRO 2 · Front-left wheel"
    orbit.set("floor")
    orbit._t = None
    rigs[1].view(640, 360, orbit.update())
    hits = {orbit.handle({"pick": [u, v]})["part"]["name"] for u in np.linspace(0.6, 0.9, 7)
            for v in np.linspace(0.3, 0.6, 7)}
    assert "Packing station" in hits


@pytest.mark.parametrize("text, out", [
    ("airo 2 deliver to packing", (1, False, "deliver to packing")),
    ("AIRO-3: weight", (2, False, "weight")),
    ("Airo two, stop", (1, False, "stop")),
    ("select airo 1", (0, True, "")),
    ("airo 3", (2, True, "")),
    ("deliver to dock", (None, False, "deliver to dock")),
])
def test_parse_robot(text, out):
    assert parse_robot(text) == out


def test_station_names_come_from_the_site():
    assert parse_intent("bring it over to Q.A. please", FACTORY.stations) == Intent("deliver", "qa")
    assert parse_intent("airo deliver to the packing station", FACTORY.stations) == Intent("deliver", "packing")


def scientist(rig, robot=None):
    sci = Scientist(rig)
    sci.best, sci.members, sci.name, sci.n = TWIN, ensemble([], TWIN, 4), "twin", 1
    return sci


def test_fleet_routes_commands_and_publishes_v3_state():
    rigs = fleet(FACTORY, 3)
    f = Fleet(FACTORY, crew(FACTORY, rigs, scientist))
    assert f.route("select airo 2") == "selected AIRO 2" and f.robot.id == "airo-2"
    m0 = mass(rigs[1])
    f.route("weight")
    assert mass(rigs[1]) > m0 and rigs[0].changes == []
    f.route("airo 3 deliver to packing")
    f.route("deliver to dock")
    assert f.robots[2].mission.inbox.get_nowait() == "deliver to packing"
    assert f.robots[1].mission.inbox.get_nowait() == "deliver to dock"
    f.route("airo 1 tow")
    f.route("spill")
    assert rigs[0].changes == ["tow", "slippery"] and rigs[1].changes == ["weight", "slippery"]
    assert f.route("airo 7 stop") == "no robot 7"
    s = f.state()
    assert s["selected"] == "airo-2" and s["mode"] == "sim" and s["site"]["name"] == "Factory"
    assert s["site"]["stations"].keys() == {"assembly", "qa", "packing", "dock"}
    assert s["site"]["zones"][0]["label"] == "Oil spill" and len(s["site"]["zones"][0]["polygon"]) == 4
    assert [r["id"] for r in s["robots"]] == ["airo-1", "airo-2", "airo-3"]
    assert all({"name", "color", "real", "now", "status", "pose", "body", "notebook"} <= r.keys() for r in s["robots"])
    assert "models" not in s and s["robots"][0]["now"].startswith("Parked at its charging dock as Body #1")
    assert f.route("restock") == "the conveyor brings totes by itself"


def test_guard_holds_still_rather_than_drive_into_another_car():
    site = replace(FACTORY, starts=((1.0, 1.0, 0.0), (1.3, 1.02, math.pi)))
    a, b = fleet(site, 2)
    guard = Guard(a, lambda: [b])

    async def run():
        await guard.step(120, 120)
        blocked = guard.blocked_by
        await guard.step(-100, 100)
        return blocked, guard.blocked_by

    assert asyncio.run(run()) == (b, None)
    assert a.history[1].tolist() == [[0, 0], [-100, 100]]


def test_in_a_standoff_the_higher_numbered_car_gives_way_and_any_car_gives_way_to_a_parked_one():
    site = replace(FACTORY, starts=((1.0, 1.0, 0.0), (1.3, 1.02, math.pi)))
    a, b = fleet(site, 2)
    guards = {}
    ga, gb = Guard(a, lambda: [b], guards=guards, rank=0), Guard(b, lambda: [a], guards=guards, rank=1)

    async def run():
        await ga.step(120, 120)
        alone = ga.should_yield(), gb.should_yield()
        await gb.step(120, 120)
        return alone, (ga.should_yield(), gb.should_yield())

    # b stands still: a gives way to it; once b also waits for a, b gives way
    assert asyncio.run(run()) == ((True, False), (False, True))


def test_one_command_per_robot_and_all_robots():
    from types import SimpleNamespace
    robots = [SimpleNamespace(name=f"AIRO {k}", rig=None) for k in (1, 2, 3)]
    fleet = Fleet(FACTORY, robots, dry=True)
    said = fleet.route("airo 1 bring a heavy tote to packing, airo 2 bring a light tote to qa and airo 3 stop")
    assert said.split("; ") == ["dry run, not driving: airo 1 bring a heavy tote to packing",
                                "dry run, not driving: airo 2 bring a light tote to qa",
                                "dry run, not driving: airo 3 stop"]
    assert fleet.route("all robots bring a heavy tote to dock").count("AIRO") == 3
    assert fleet.route("airo 2 bring a heavy and light tote to qa").count("dry run") == 1

import asyncio
import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from airo import meta_api
from airo.body import Body, ensemble
from airo.control import face, follow
from airo.fleet import Fleet, crew
from airo.loader import Arm
from airo.mjsim import fleet
from airo.scientist import Scientist
from airo.site import FACTORY, KINDS
from airo.warehouse import CLEAR, TOTE, Line, cells

TWIN = Body(g_l=0.00312, g_r=0.00305, dz=27.4, eta=0.303, tau=0.213, lag=0.128, kappa=-0.008)
# a heavy tote costs the twin about 19% of its speed and 13% of its turn
HEAVY = replace(TWIN, g_l=TWIN.g_l * 0.81, g_r=TWIN.g_r * 0.81, eta=TWIN.eta * 1.07)
AT_PICK = replace(FACTORY, starts=(FACTORY.pick.pose,))


def scientist(rig, robot=None):
    sci = Scientist(rig)
    sci.best, sci.members, sci.name, sci.n = TWIN, ensemble([], TWIN, 4), "twin", 1
    return sci


class Feeling(Scientist):
    """Answers check_body like the memory scientist: the heavy-tote body when loaded, the bare one unloaded. Its
    cloud is tight, as after a real fit."""

    def __init__(self, rig, robot=None):
        super().__init__(rig)
        self.best, self.members, self.name, self.n = TWIN, [TWIN], "twin", 1
        self.causes = []

    async def check_body(self, cause: str) -> dict:
        self.causes.append(cause)
        for _ in range(10):
            await self.rig.step(0, 0)
        if cause == "loaded a light tote":
            return {"result": "same", "n": self.n, "name": self.name, "seconds": 0.5}
        if cause.startswith("loaded"):
            self.best, self.members, self.n, self.name = HEAVY, [HEAVY], 2, "heavy tote"
            return {"result": "recalled", "n": 2, "name": "heavy tote", "seconds": 1.1}
        self.best, self.members, self.n, self.name = TWIN, [TWIN], 1, "twin"
        return {"result": "recalled", "n": 1, "name": "twin", "seconds": 0.9}


def deck_offset(car) -> float:
    cell = car.world.cell
    t = cell.on(car)
    return float(np.linalg.norm(car.world.d.xpos[t.body] - cell.deck_pose(car)[0]))


async def loaded(car, kind: str = "heavy"):
    cell = car.world.cell
    tote = cell.order(kind)
    while not cell.ready(tote):
        await car.step(0, 0)
    await cell.pick.load(car, tote)
    return tote


def test_arm_reaches_with_the_fingers_across_the_tote():
    car = fleet(AT_PICK, 1, log_dir=None)[0]
    m, d, arm = car.world.m, car.world.d, car.world.cell.pick.arm
    for target, yaw in [((0.3, 1.0, 0.16), 0.4), ((-0.2, 1.08, 0.18), math.pi / 2), ((0.1, 0.8, 0.3), -1.0)]:
        q = arm.ik(target, yaw)
        for j, v in zip(arm.joints, q):
            d.qpos[m.jnt_qposadr[m.joint(j).id]] = v
        mujoco.mj_forward(m, d)
        assert np.allclose(d.xpos[m.body(arm.tcp).id], target, atol=1e-3)
        tool = d.xmat[m.body(arm.tcp).id].reshape(3, 3)
        assert tool[2, 0] == pytest.approx(-1, abs=1e-3)
        across = tool[:2, 1]
        assert abs(across @ (math.cos(yaw), math.sin(yaw))) < 1e-3
    with pytest.raises(ValueError, match="out of reach"):
        arm.ik((1.5, 1.0, 0.1), 0.0)
    assert isinstance(arm, Arm) and arm.reach > 0.4


def test_every_bay_has_clear_floor_ahead():
    """check_body's straight probe drives up to 25 cm forward out of a bay."""
    blocks = [*cells(FACTORY).values()]
    for name, bay in [("pick", FACTORY.pick), *FACTORY.drops.items()]:
        c, s = math.cos(bay.theta), math.sin(bay.theta)
        for d in np.linspace(0, 0.25, 6):
            x, y = bay.x + d * c, bay.y + d * s
            assert 0.15 <= x <= FACTORY.floor[0] - 0.15 and 0.15 <= y <= FACTORY.floor[1] - 0.15, name
            for cl in blocks:
                assert math.dist((x, y), cl.base) > 0.25, (name, cl.name)


def test_loader_loads_the_tote_onto_the_deck_and_off_at_a_station():
    car = fleet(AT_PICK, 1, log_dir=None)[0]
    world, cell = car.world, car.world.cell
    m0 = world.m.body_subtreemass[car.root]

    async def run():
        tote = await loaded(car)
        on = cell.on(car), deck_offset(car)
        bay = FACTORY.drops["packing"]
        await follow(car, TWIN, [(car.pose.x, car.pose.y), (bay.x, bay.y - 0.3), (bay.x, bay.y)])
        await face(car, TWIN, bay.theta)
        slot = cell.slot("packing")
        await cell.loaders["packing"].unload(car, slot)
        for _ in range(10):
            await car.step(0, 0)
        return tote, on, slot

    tote, (on, offset), slot = asyncio.run(run())
    assert on is tote and offset < 0.005 and tote.kind == "heavy"
    assert cell.on(car) is None and tote.slot == slot
    assert np.allclose(world.d.xpos[tote.body], cell.slot_pose(slot)[0], atol=1e-3)
    assert world.m.body_subtreemass[car.root] == pytest.approx(m0)
    assert world.m.body_mass[tote.body] == KINDS["heavy"]


@pytest.mark.parametrize("cmd", [(150, 150), (-140, 140), (200, -200)], ids=["straight", "pivot", "fast pivot"])
def test_tote_stays_on_through_hard_driving(cmd):
    car = fleet(AT_PICK, 1, log_dir=None)[0]

    async def run():
        await loaded(car)
        worst = 0.0
        for k in range(40):
            await car.step(*cmd)
            worst = max(worst, deck_offset(car))
        return worst

    assert asyncio.run(run()) < 0.02


def test_a_heavy_tote_makes_the_car_slower():
    def drive(kind):
        car = fleet(AT_PICK, 1, noise=(0, 0), log_dir=None)[0]

        async def run():
            if kind:
                await loaded(car, kind)
            for _ in range(10):
                await car.step(0, 0)
            p0 = car.true_pose
            for _ in range(30):
                await car.step(120, 120)
            p1 = car.true_pose
            for _ in range(30):
                await car.step(-110, 110)
            return np.linalg.norm(p1[:2] - p0[:2]), abs(math.remainder(car.true_pose[2] - p1[2], math.tau))
        return asyncio.run(run())

    (d0, turn0), (d_light, _), (d_heavy, turn_heavy) = drive(None), drive("light"), drive("heavy")
    assert d_heavy < 0.92 * d0 and turn_heavy < 0.9 * turn0
    assert d_heavy < d_light <= d0 * 1.01


def run_fleet(n, commands, make=scientist, until=150.0):
    rigs = fleet(FACTORY, n, log_dir=None)
    seg = meta_api.segment
    meta_api.segment = rigs[0].sam
    f = Fleet(FACTORY, crew(FACTORY, rigs, make))
    world = rigs[0].world
    gaps, seen = [], {r.id: [] for r in f.robots}

    async def run():
        jobs = [asyncio.create_task(r.mission.run()) for r in f.robots]
        for text in commands:
            f.route(text)
        await asyncio.sleep(0.1)
        while world.t < until and any(r.mission.busy for r in f.robots):
            for r in f.robots:
                if not seen[r.id] or seen[r.id][-1][0] != r.mission.status:
                    seen[r.id].append((r.mission.status, f._now(r)))
            ps = [r.true_pose[:2] for r in rigs]
            gaps.extend(np.linalg.norm(p - q) for i, p in enumerate(ps) for q in ps[i + 1:])
            await asyncio.sleep(0)
        for j in jobs:
            j.cancel()

    try:
        asyncio.run(run())
    finally:
        meta_api.segment = seg
    return f, world, gaps, seen


def test_one_robot_fetches_feels_its_body_and_delivers():
    f, world, _, seen = run_fleet(1, ["airo 1 bring a heavy tote to packing"], Feeling)
    r, cell = f.robots[0], world.cell
    statuses = [s for s, _ in seen[r.id]]
    assert statuses[:7] == ["fetching", "loading", "feeling", "delivering", "unloading", "feeling", "returning"]
    assert r.sci.causes == ["loaded a heavy tote", "unloaded"]
    now = dict(seen[r.id])
    assert now["loading"] == "Waiting for the arm to load a heavy tote."
    assert now["delivering"] == "Recalled Body #2 · heavy tote · 1.1 s. Driving to Packing with a heavy tote."
    assert now["unloading"] == "Unloading at Packing."
    texts = [n["text"] for n in r.mission.notes]
    assert "Loaded a heavy tote: recalled Body #2 · heavy tote · 1.1 s." in texts
    assert "Delivered a heavy tote to Packing." in texts
    placed = [t for t in cell.totes if t.slot is not None]
    assert len(placed) == 1 and placed[0].slot.station == "packing" and placed[0].kind == "heavy"
    assert math.dist(r.rig.true_pose[:2], FACTORY.starts[0][:2]) < 0.06 and not r.mission.busy


def test_three_robots_fetch_and_deliver_without_colliding_or_deadlock():
    f, world, gaps, seen = run_fleet(3, ["airo 1 deliver to assembly", "airo 2 deliver to packing",
                                         "airo 3 deliver to dock"])
    for r, station in zip(f.robots, ("assembly", "packing", "dock")):
        assert any(n["text"].startswith(f"Delivered a ") and n["text"].endswith(f"{station.capitalize()}.")
                   for n in r.mission.notes), r.name
        assert math.dist(r.rig.true_pose[:2], r.mission.dock[:2]) < 0.06
    assert world.t < 110 and min(gaps) > 0.25
    assert any(s == "queued" for s, _ in seen["airo-3"])
    assert {t.slot.station for t in world.cell.totes if t.slot} == {"assembly", "packing", "dock"}
    assert all(not line.robots for line in world.cell.lines.values())


def test_conveyor_keeps_a_tote_for_every_order():
    w = fleet(FACTORY, 1, log_dir=None)[0].world
    cell = w.cell
    orders = [cell.order(k) for k in ("heavy", "light", "heavy")]
    assert len({id(t) for t in orders}) == 3 and [t.kind for t in orders] == ["heavy", "light", "heavy"]
    assert len(cell.belt) == 4 and cell.ready(orders[0])
    w._physics(w.t + 3)
    ends = [w.d.xpos[t.body][:2] for t in cell.belt]
    assert all(np.linalg.norm(np.diff(ends, axis=0), axis=1) > 2 * TOTE[0])
    assert not any(cell.ready(t) for t in orders[1:])


def test_bay_line_lets_robots_move_up_in_order():
    line = FACTORY.pick
    q = Line("pick", line)
    for r in "abc":
        q.join(r)
    assert q.spot("a") == (0, line.pose) and q.spot("b") == (1, line.wait[0]) and q.spot("c")[0] == 2
    assert q.clear("a") and not q.clear("b")
    q.arrive("a", 0)
    assert q.clear("b") and not q.clear("c")
    q.leave("a")
    assert q.spot("b")[0] == 0 and q.spot("c") == (1, line.wait[0]) and CLEAR > 0.25


def test_tote_kind_comes_from_the_command():
    rigs = fleet(FACTORY, 1, log_dir=None)
    m = crew(FACTORY, rigs, scientist)[0].mission

    async def run(text):
        await m.handle(text)
        kind = m.kind
        await asyncio.sleep(0)
        await m._cancel()
        return kind

    assert [asyncio.run(run(t)) for t in ("bring a heavy tote to qa", "deliver a light one to dock",
                                           "deliver to packing")] == ["heavy", "light", None]


def test_picking_names_loaders_and_totes():
    from airo.orbit import Orbit

    car = fleet(AT_PICK, 1, log_dir=None)[0]
    cell = car.world.cell

    async def run():
        tote = cell.order("heavy")
        task = asyncio.create_task(cell.pick.load(car, tote))
        for _ in range(20):
            await car.step(0, 0)
        doing = cell.label("arm_pick/l3")
        await task
        return tote, doing

    tote, doing = asyncio.run(run())
    assert doing[0] == "Pick cell arm · loading AIRO 1"
    assert cell.label(tote.name) == ("Tote · heavy", "0.8 kg of steel parts, latched on AIRO 1's tray")
    assert cell.label("arm_qa") == ("QA arm · idle", doing[1]) and cell.label("rack_beam") is None
    m = car.world.m
    orbit = Orbit(car)
    assert orbit._label(0, m.geom("fence_fence_mesh").id, np.zeros(3))["name"] == "Safety fence"


@pytest.mark.parametrize("change, kind", [("flip", "heavy"), ("tow", "light"), (None, "heavy")])
def test_unannounced_change_is_still_caught_while_carrying(change, kind):
    rigs = fleet(FACTORY, 1, log_dir=None)
    seg = meta_api.segment
    meta_api.segment = rigs[0].sam
    r = crew(FACTORY, rigs, Feeling)[0]
    caught = []

    async def relearn(window, budget=3):
        caught.append((r.mission.status, r.mission.loaded))
        raise asyncio.CancelledError

    r.sci.relearn = relearn

    async def run():
        job = asyncio.create_task(r.mission.run())
        r.mission.inbox.put_nowait(f"bring a {kind} tote to qa")
        while r.mission.status != "delivering" and rigs[0].t < 60:
            await asyncio.sleep(0)
        for _ in range(10):
            await asyncio.sleep(0)
        if change:
            rigs[0].flip_motor("l") if change == "flip" else rigs[0].attach_tow()
        while not caught and r.mission.status == "delivering":
            await asyncio.sleep(0)
        job.cancel()

    try:
        asyncio.run(run())
    finally:
        meta_api.segment = seg
    assert caught == ([("relearning", True)] if change else [])

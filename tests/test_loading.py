import asyncio
import math
from dataclasses import replace

import mujoco
import numpy as np
import pytest

from airo import meta_api
from airo.body import Body, ensemble, rollout
from airo.control import face, follow
from airo.camera import Pose
from airo.fleet import Fleet, crew, footprint
from airo.loader import Arm
from airo.mjsim import fleet
from airo.rig import DT
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

    async def relearn(self, window, budget=3):
        """The fleet tests check the flow, not the fit: a relearn just takes a second."""
        for _ in range(20):
            await self.rig.step(0, 0)
        return 1.0


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
    """check_body's straight probe drives up to 25 cm forward out of a bay, clear of every cell's guarding."""
    blocks = [*cells(FACTORY).values()]
    for name, bay in [("pick", FACTORY.pick), *FACTORY.drops.items()]:
        c, s = math.cos(bay.theta), math.sin(bay.theta)
        for d in np.linspace(0, 0.25, 6):
            x, y = bay.x + d * c, bay.y + d * s
            assert 0.15 <= x <= FACTORY.floor[0] - 0.15 and 0.15 <= y <= FACTORY.floor[1] - 0.15, name
            car = footprint(Pose(x, y, bay.theta, 0.0), pad=0.02)
            for cl in blocks:
                out, along = (car - cl.base) @ np.array(cl.out), (car - cl.base) @ np.array(cl.along)
                assert out.max() < cl.front or along.max() < -cl.side or along.min() > cl.side, (name, cl.name)


@pytest.mark.parametrize("station", ["packing", "dock"])
def test_loader_loads_the_tote_onto_the_deck_and_off_at_a_station(models, station):
    car = fleet(AT_PICK, 1, log_dir=None)[0]
    world, cell = car.world, car.world.cell
    m0 = world.m.body_subtreemass[car.root]

    async def run():
        tote = await loaded(car)
        on = cell.on(car), deck_offset(car)
        bay = FACTORY.drops[station]
        c, s = math.cos(bay.theta), math.sin(bay.theta)
        await follow(car, TWIN, [(car.pose.x, car.pose.y), (bay.x - 0.3 * c, bay.y - 0.3 * s), (bay.x, bay.y)])
        await face(car, TWIN, bay.theta)
        slot = cell.slot(station)
        await cell.loaders[station].unload(car, slot)
        for _ in range(10):
            await car.step(0, 0)
        return tote, on, slot

    tote, (on, offset), slot = asyncio.run(run())
    assert on is tote and offset < 0.005 and tote.kind == "heavy"
    assert (type(cell.loaders[station].arm).__name__ == "UR5e") == (models == "menagerie")
    assert cell.on(car) is None and tote.slot == slot
    assert np.allclose(world.d.xpos[tote.body], cell.slot_pose(slot)[0], atol=1e-3)
    assert world.m.body_subtreemass[car.root] == pytest.approx(m0)
    assert world.m.body_mass[tote.body] == KINDS["heavy"]


def _unexplained(car, body: Body, t0: float, window: float = 0.5) -> float:
    """The largest gap, over half-second windows since t0, between where the car went and where its commands take
    the body: a push from anything but its own wheels shows here."""
    seg = car.segment(t0)
    worst = 0.0
    for k in range(0, len(seg.t) - round(window / DT), round(window / 4 / DT)):
        j = k + round(window / DT)
        t = seg.t[k:j + 1]
        pred = rollout(body, t, seg.cmd_t, seg.cmd, seg.pose[k])
        worst = max(worst, float(np.linalg.norm(pred[-1, :2] - seg.pose[j, :2])))
    return worst


@pytest.mark.parametrize("station", ["assembly", "qa", "packing", "dock"])
def test_a_loaded_car_turns_into_a_drop_bay_on_its_path_and_stays_put_while_unloaded(models, station):
    """Regression: cars were flung sideways off the floor turning into the packing and dock bays, by a new order
    for the conveyor resetting the latch of the tote they carried. Turning into the bay and pivoting in it, the car
    goes where its commands take it, and it stands still while unloaded, as other orders come in."""
    car = fleet(AT_PICK, 1, noise=(0, 0), log_dir=None)[0]
    cell = car.world.cell
    bay = FACTORY.drops[station]
    via = (bay.x - 0.3 * math.cos(bay.theta), bay.y - 0.3 * math.sin(bay.theta))

    async def run():
        await loaded(car)
        await follow(car, TWIN, [(car.pose.x, car.pose.y), via])
        t0 = car.pose.t
        cell.order("light")
        await follow(car, TWIN, [via, (bay.x, bay.y)])
        for heading in (bay.theta + 0.47, bay.theta - 0.3, bay.theta):
            await face(car, TWIN, heading)
        cell.order("heavy")
        parked = car.true_pose.copy()
        await cell.loaders[station].unload(car, cell.slot(station))
        return t0, parked

    t0, parked = asyncio.run(run())
    assert _unexplained(car, HEAVY, t0) < 0.02
    assert np.linalg.norm(car.true_pose[:2] - parked[:2]) < 0.003
    assert abs(math.remainder(car.true_pose[2] - parked[2], math.tau)) < 0.01


def test_a_pivot_at_the_pick_bay_right_after_loading_measures_the_loaded_body(models):
    """Regression: something moved a car parked at the pick bay while it felt its new body. The cell keeps still,
    even as the next order comes in, and check_body's pivot lands where the heavy-tote body predicts."""
    from airo.probes import LIBRARY, halt, measure, predict, run as probe, zscore
    from airo.scientist import CHECK, CONFIRM_Z

    car = fleet(AT_PICK, 1, noise=(0, 0), log_dir=None)[0]
    w, cell = car.world, car.world.cell
    arm = [w.m.jnt_qposadr[w.m.joint(j).id] for j in cell.pick.arm.joints]

    async def run():
        await loaded(car)
        await halt(car)
        cell.order("light")
        q0, p = w.d.qpos[arm].copy(), car.pose
        want = predict([HEAVY], LIBRARY[CHECK], (p.x, p.y, p.theta))
        got = measure(await probe(car, LIBRARY[CHECK]))
        return want, got, np.abs(w.d.qpos[arm] - q0).max()

    want, got, moved = asyncio.run(run())
    assert max(map(abs, zscore(want, got).values())) < CONFIRM_Z and moved < 1e-9


@pytest.mark.parametrize("cmd", [(150, 150), (-140, 140), (200, -200)], ids=["straight", "pivot", "fast pivot"])
def test_tote_stays_on_through_hard_driving(cmd):
    car = fleet(AT_PICK, 1, log_dir=None)[0]

    async def run():
        await loaded(car)
        worst = 0.0
        for _ in range(40):
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
    assert now["delivering"] == "Recognised Body #2 (heavy tote) in 1.1 s. Driving to Packing with a heavy tote."
    assert now["unloading"] == "Unloading at Packing."
    texts = [n["text"] for n in r.mission.notes]
    assert "Loaded a heavy tote: recognised Body #2 (heavy tote) in 1.1 s." in texts
    assert "Delivered a heavy tote to Packing." in texts
    placed = [t for t in cell.totes if t.slot is not None]
    assert len(placed) == 1 and placed[0].slot.station == "packing" and placed[0].kind == "heavy"
    assert math.dist(r.rig.true_pose[:2], FACTORY.starts[0][:2]) < 0.06 and not r.mission.busy


def test_three_robots_fetch_and_deliver_without_colliding_or_deadlock():
    f, world, gaps, seen = run_fleet(3, ["airo 1 deliver to assembly", "airo 2 deliver to packing",
                                         "airo 3 deliver to dock"], Feeling)
    for r, station in zip(f.robots, ("assembly", "packing", "dock")):
        assert any(n["text"].startswith("Delivered a ") and n["text"].endswith(f"{station.capitalize()}.")
                   for n in r.mission.notes), r.name
        assert math.dist(r.rig.true_pose[:2], r.mission.dock[:2]) < 0.06
    assert world.t < 130 and min(gaps) > 0.25
    assert any(s == "queued" for s, _ in seen["airo-3"])
    assert {t.slot.station for t in world.cell.totes if t.slot} == {"assembly", "packing", "dock"}
    assert all(not line.robots for line in world.cell.lines.values())


def test_a_fleet_run_is_the_same_every_time():
    """Lockstep simulation must not depend on the host's timing: missions wake on events, not on the wall clock."""
    runs = [run_fleet(3, ["airo 1 deliver to assembly", "airo 2 deliver to packing", "airo 3 deliver to dock"],
                      Feeling) for _ in range(2)]
    ends = [(round(w.t, 3), [tuple(np.round(r.rig.true_pose, 4)) for r in f.robots]) for f, w, _, _ in runs]
    assert ends[0] == ends[1]


def test_the_real_check_body_runs_at_both_cells():
    f, world, _, seen = run_fleet(1, ["airo 1 bring a heavy tote to qa"], scientist, until=250)
    texts = [n["text"] for n in f.robots[0].mission.notes]
    assert [t.split(":")[0] for t in texts if t.startswith(("Loaded", "Unloaded"))] == ["Loaded a heavy tote",
                                                                                     "Unloaded"]
    assert "Delivered a heavy tote to QA." in texts and not f.robots[0].mission.busy


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


def test_picking_names_loaders_and_totes(models):
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
    if models == "menagerie":
        name, detail = cell.label("arm_qa/hand/left_pad")
        assert name == "Station arm · UR5e · QA" and detail.startswith("Idle; Robotiq 2F-85 gripper")
    else:
        assert cell.label("arm_qa") == ("QA arm · idle", doing[1])
    assert cell.label("rack_beam") is None
    m = car.world.m
    orbit = Orbit(car)
    assert orbit._label(0, m.geom("fence_fence_panel").id, np.zeros(3))["name"] == "Safety guarding"


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

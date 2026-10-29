import asyncio
import time
import json
import math
from dataclasses import replace

import pytest

from airo import planner
from airo.body import KEY
from airo.control import V_MAX
from airo.memory import Memory
from airo.mission import Mission, _gauge
from airo.rig import RigAbort
from airo.sim import PARCEL, SimRig, world

BLUE = (1.7, 0.3)
HEAVY = replace(KEY, g_l=KEY.g_l * 0.6, g_r=KEY.g_r * 0.6, eta=KEY.eta * 0.7)


class FakeScientist:
    """Relearning reads the truth off the sim, so these tests exercise the mission, not the fit."""

    def __init__(self, rig):
        self.rig, self.best, self.n, self.name, self.notebook = rig, KEY, 1, "bare chassis", []

    @property
    def members(self):
        return [self.best]

    async def learn(self):
        self.best = self.rig.truth

    async def relearn(self, seg, budget=3):
        for _ in range(20):
            await self.rig.step(0, 0)
        self.best, self.n = self.rig.truth, self.n + 1
        return 1.0


@pytest.fixture
def mission(monkeypatch):
    rig = SimRig(KEY, floor=(2.0, 1.4), objects=world(), pose0=(1.0, 0.7, math.pi), log_dir=None)
    monkeypatch.setattr(planner.meta_api, "segment", rig.sam)
    return Mission(rig, FakeScientist(rig))


def parcel(rig):
    return next(o.xy for o in rig.objects if o.concept == PARCEL)


async def until(rig, t):
    while rig.t < t:
        await asyncio.sleep(0)


def test_delivers_without_false_surprise(mission):
    assert asyncio.run(mission.deliver("blue"))
    assert mission.sci.n == 1 and max(mission.values) < 3
    assert math.dist(parcel(mission.rig), BLUE) < 0.1


def test_body_change_triggers_relearn_then_delivers(mission):
    async def run():
        async def add_weight():
            await until(mission.rig, 0.3)
            mission.rig.truth = HEAVY

        asyncio.create_task(add_weight())
        return await mission.deliver("blue")

    assert asyncio.run(run())
    assert mission.sci.n == 2 and mission.relearned_s == pytest.approx(1.0) == mission.probing_s
    assert [n["kind"] for n in mission.notes] == ["conclusion", "result"]
    state = mission.state()
    assert [p["kind"] for p in state["episode"]["phases"]][:3] == ["drive", "surprise", "relearn"]
    assert state["relearn"]["phase"] == "done" and state["body"]["params"][0]["key"] is None
    assert math.dist(parcel(mission.rig), BLUE) < 0.1


def test_stop_wins_and_state_serialises(mission):
    async def run():
        await mission.handle("deliver the parcel to blue")
        await until(mission.rig, 1.0)
        state = mission.state()
        await mission.handle("stop")
        return state

    state = asyncio.run(run())
    assert mission.status == "stopped" and not mission.busy
    assert tuple(mission.rig.history[1][-1]) == (0, 0)
    assert state["status"] == "delivering" and len(state["cloud"]) == 1 and len(state["path"]) >= 2
    json.dumps(state, default=lambda o: o.tolist())


def test_unknown_station_is_rejected(mission):
    asyncio.run(mission.handle("deliver to green"))
    assert not mission.busy and "Stations" in mission.notes[-1]["text"]


@pytest.mark.parametrize("watch_push, relearns", [(False, 0), (True, 1)])
def test_push_leg_is_watched_only_on_request(mission, watch_push, relearns):
    mission.watch_push = watch_push

    async def run():
        async def add_weight():
            while mission.status != "pushing":
                await asyncio.sleep(0)
            mission.rig.truth = HEAVY

        asyncio.create_task(add_weight())
        return await mission.deliver("blue")

    assert asyncio.run(run()) and mission.sci.n == 1 + relearns


@pytest.mark.parametrize("reach, moved", [(0.6, False), (1.0, True)])
def test_relearn_moves_clear_of_a_nearby_parcel(mission, monkeypatch, reach, moved):
    calls = []

    async def recentre(rig):
        calls.append(rig)

    monkeypatch.setattr("airo.mission.PROBE_REACH", reach)
    mission.sci.recentre = recentre

    async def run():
        mission.scene = await mission.scan()
        await mission.rig.step(0, 0)
        await mission.relearn("test")

    asyncio.run(run())
    assert bool(calls) == moved and mission.sci.n == 2


def test_unconfirmed_body_drives_slower(monkeypatch):
    def deliver(verified):
        rig = SimRig(KEY, floor=(2.0, 1.4), objects=world(), pose0=(1.0, 0.7, math.pi), log_dir=None)
        monkeypatch.setattr(planner.meta_api, "segment", rig.sam)
        m = Mission(rig, FakeScientist(rig))
        m.sci.verified = verified
        assert asyncio.run(m.deliver("blue"))
        return m.v_max, rig.t

    (v_ok, t_ok), (v_slow, t_slow) = deliver(True), deliver(False)
    assert v_ok == V_MAX and v_slow < V_MAX and t_slow > 1.2 * t_ok


def test_inconsistent_relearn_stops_with_a_fault(mission):
    async def broken(seg, budget=3):
        mission.sci.notebook.append({"t": time.time(), "kind": "result", "z": {"turn_deg": 9.0}})
        mission.sci.verified = False
        return 1.0

    mission.sci.relearn = broken
    with pytest.raises(RigAbort, match="hardware fault"):
        asyncio.run(mission.relearn("test"))
    assert mission.notes[-1]["text"].startswith("Hardware fault")


def test_notebook_outcomes_become_gauge_numbers():
    e = _gauge({"kind": "result", "predicted": {"forward_cm": (1.0, 0.2), "lateral_cm": (0.1, 0.1),
                                                "turn_deg": (51.3, 11.0)},
                "measured": {"forward_cm": 0.5, "lateral_cm": 0.0, "turn_deg": 68.8}})
    assert (e["predicted"], e["sd"], e["measured"], e["unit"]) == (51.3, 11.0, 68.8, "°")
    assert _gauge({"kind": "prediction", "predicted": {"turn_deg": (5.0, 1.0)}})["predicted"] == {"turn_deg": (5.0, 1.0)}


@pytest.mark.parametrize("x, y, theta, quiet", [(0.3, 0.5, 0.0, True), (0.3, 0.5, math.radians(80), False),
                                                (0.5, 0.65, 0.0, False), (0.1, 0.5, 0.0, False)],
                         ids=["run-in", "pivot near parcel", "beside the axis", "behind the run-in"])
def test_only_the_straight_run_in_is_unwatched(mission, x, y, theta, quiet):
    from airo.camera import Pose

    mission._near = (0.5, 0.5, 0.0)
    assert mission._running_in(Pose(x, y, theta, 0.0)) == quiet


@pytest.mark.parametrize("x, quiet", [(0.75, True), (0.9, False)])
def test_no_surprise_within_reach_of_the_parcel(mission, x, quiet):
    """After a slip the car turns around beside the parcel, and a pivot there can sweep it."""
    from airo.camera import Pose

    mission.scene = planner.Scene(None, 0.02, {}, (0.5, 0.7), [])
    assert mission._touching(Pose(x, 0.7, 0.0, 0.0)) == quiet


def test_timeline_shows_each_probe_and_refit(mission):
    t = mission.t0
    mission.status = "learning"
    mission.sci.notebook += [{"t": t + 1, "kind": "hypothesis", "probe": "ramp_120", "text": ""},
                             {"t": t + 2, "kind": "result", "probe": "ramp_120", "text": ""},
                             {"t": t + 3, "kind": "hypothesis", "probe": "arc_r", "text": ""},
                             {"t": t + 4, "kind": "conclusion", "name": "bare chassis", "text": ""}]
    phases = mission.state()["episode"]["phases"]
    assert [(p["kind"], p["label"]) for p in phases] == [("probe", "learn the body"), ("probe", "ramp_120"),
                                                         ("probe", "arc_r"), ("refit", "bare chassis")]


@pytest.mark.parametrize("online", [True, False])
def test_self_portrait_after_a_confirmed_body(mission, monkeypatch, tmp_path, online):
    prompts = []

    async def image(prompt):
        prompts.append(prompt)
        if not online:
            raise KeyError("MODEL_API_KEY")
        return b"png"

    monkeypatch.setattr("airo.portrait.meta_api.image", image)
    mission.portraits = tmp_path

    async def run():
        await mission.learn()
        await mission._drawing

    asyncio.run(run())
    assert "bare chassis" in prompts[0]
    assert (mission.state()["body"]["portrait"], (tmp_path / "body_1.png").exists()) == ((1, True) if online
                                                                                          else (None, False))


def test_recall_shows_the_remembered_portrait_and_memory(mission, tmp_path):
    (tmp_path / "body_2.png").write_bytes(b"heavy png")
    mem = Memory()
    mem.add("bare chassis", "learned from scratch", KEY, [KEY], True)
    mem.add("heavier", "a weight on top", HEAVY, [HEAVY], True).portrait = str(tmp_path / "body_2.png")
    mission.sci.memory = mem

    async def recalled(seg, budget=3):
        mission.sci.n, mission.sci.name = 2, "heavier"
        mission.sci.notebook.append({"t": time.time(), "kind": "recall", "n": 2, "name": "heavier",
                                  "text": "I know this one."})
        return 1.0

    assert mission.state()["now"] == "Waiting for a command as Body #1 (bare chassis)."
    mission.sci.relearn = recalled
    asyncio.run(mission.relearn("test"))
    state = mission.state()
    assert "recalled in" in mission.notes[-1]["text"] and mission.portrait == b"heavy png"
    assert [(m["n"], m["active"], m["portrait"]) for m in state["memory"]] == [(1, False, None),
                                                                              (2, True, "/portraits/body_2.png")]
    assert state["memory"][1]["params"][0]["value"] == pytest.approx(HEAVY.g_l * 1000)
    assert state["episode"]["phases"][-1]["label"] == "heavier"

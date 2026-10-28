import asyncio
import json
import math
from dataclasses import replace

import pytest

from airo import planner
from airo.body import KEY
from airo.mission import Mission
from airo.sim import PARCEL, SimRig, world

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

    async def relearn(self, seg):
        self.best, self.n = self.rig.truth, self.n + 1
        return 1.0


@pytest.fixture
def mission(monkeypatch):
    rig = SimRig(KEY, floor=(2.0, 1.4), objects=world(), pose0=(0.3, 0.6, 0.0), log_dir=None)
    monkeypatch.setattr(planner.meta_api, "segment", rig.sam)
    return Mission(rig, FakeScientist(rig))


def parcel(rig):
    return next(o.xy for o in rig.objects if o.concept == PARCEL)


async def until(rig, t):
    while rig.t < t:
        await asyncio.sleep(0)


def test_delivers_without_false_surprise(mission):
    assert asyncio.run(mission.deliver("red"))
    assert mission.sci.n == 1 and max(mission.values) < 3
    assert math.dist(parcel(mission.rig), (1.7, 1.1)) < 0.1


def test_body_change_triggers_relearn_then_delivers(mission):
    async def run():
        async def add_weight():
            await until(mission.rig, 0.3)
            mission.rig.truth = HEAVY

        asyncio.create_task(add_weight())
        return await mission.deliver("red")

    assert asyncio.run(run())
    assert mission.sci.n == 2 and mission.relearned_s == 1.0
    assert [n["kind"] for n in mission.notes] == ["alert", "conclusion", "result"]
    assert math.dist(parcel(mission.rig), (1.7, 1.1)) < 0.1


def test_stop_wins_and_state_serialises(mission):
    async def run():
        await mission.handle("deliver the parcel to red")
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

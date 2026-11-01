import asyncio
import json
import time

import numpy as np
import pytest

from airo.body import KEY, rollout
from airo import rig as rigmod
from airo.camera import Pose
from airo.rig import BaseRig, Rig, RigAbort
from airo.sim import SimRig


async def drive(rig, l, r, n):
    for _ in range(n):
        await rig.step(l, r)


def test_sim_matches_rollout_and_lags_camera():
    rig = SimRig(KEY, noise=(0, 0), latency=0.1, log_dir=None)
    asyncio.run(drive(rig, 100, 80, 30))
    seg = rig.segment()
    truth = rollout(KEY, seg.t - 0.1, *rig.history, (1.0, 0.6, 0.0))
    assert np.abs(seg.pose[2:] - truth[2:]).max() < 0.003


def test_sim_aborts_at_inset_and_logs(tmp_path):
    rig = SimRig(KEY, log_dir=tmp_path)
    with pytest.raises(RigAbort, match="inset"):
        asyncio.run(drive(rig, 200, 200, 200))
    assert tuple(rig.history[1][-1]) == (0, 0)
    assert rig.pose.x > rig.floor[0] - rig.inset
    asyncio.run(rig.close())
    rows = [json.loads(line) for line in next(tmp_path.glob("*.jsonl")).read_text().splitlines()]
    assert rows[-1].keys() == {"t", "pwm_l", "pwm_r", "sent", "x", "y", "theta", "pose_t"}


class BlindRig(Rig):
    def __init__(self):
        BaseRig.__init__(self, (0.8, 1.2), log_dir=None)
        self.car, self.safety, self._latest, self._next, self._t0, self.stops = None, True, None, 0.0, time.monotonic(), 0
        self.pose = Pose(0.4, 0.6, 0.0, self.clock())

    async def stop(self):
        self.stops += 1


async def blind_tick(rig, back_after):
    async def marker_back():
        await asyncio.sleep(back_after)
        rig._latest = Pose(0.41, 0.6, 0.0, rig.clock())
    if back_after is not None:
        asyncio.get_running_loop().create_task(marker_back())
    await asyncio.sleep(rigmod.LOST_S)
    return await rig._tick(rig.clock(), 60, 60, True, None)


def test_brief_blindness_holds_still_and_resumes():
    rig = BlindRig()
    pose = asyncio.run(blind_tick(rig, 0.4))
    assert pose.x == 0.41 and rig.stops == 1 and rig.held_t > 0


def test_lasting_blindness_is_a_safety_stop(monkeypatch):
    monkeypatch.setattr(rigmod, "HOLD_S", 0.2)
    with pytest.raises(RigAbort, match="pose lost"):
        asyncio.run(blind_tick(BlindRig(), None))

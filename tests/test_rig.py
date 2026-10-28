import asyncio
import json

import numpy as np
import pytest

from airo.body import KEY, rollout
from airo.rig import RigAbort
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

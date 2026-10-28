import asyncio
import math

import numpy as np
import pytest

from airo.body import KEY, Body
from airo.control import follow, goto
from airo.sim import SimRig

DRIFTY = Body(g_l=0.0029, g_r=0.0034, dz=28, eta=0.27, tau=0.25, lag=0.12, kappa=-0.15)


@pytest.mark.parametrize("truth, model, latency", [(KEY, KEY, 0), (DRIFTY, DRIFTY, 0.08), (DRIFTY, KEY, 0)])
def test_goto_reaches_ten_goals(truth, model, latency):
    async def run():
        rig = SimRig(truth, latency=latency, log_dir=None)
        goals = np.random.default_rng(0).uniform((0.3, 0.3), (1.7, 0.9), (10, 2))
        return [await goto(rig, model, g) for g in goals]

    assert max(asyncio.run(run())) < 0.05


def test_follow_path():
    path = [(0.4, 0.4), (1.6, 0.4), (1.6, 0.8), (0.6, 0.8)]

    async def run():
        rig = SimRig(DRIFTY, pose0=(0.4, 0.4, 0.0), log_dir=None)
        err = await follow(rig, KEY, np.vstack([np.linspace(a, b, 20) for a, b in zip(path, path[1:])]))
        return err, rig.segment().pose

    err, track = asyncio.run(run())
    assert err < 0.05
    assert track[:, 1].max() < 0.8 + 0.08 and track[:, 0].max() < 1.6 + 0.08
    assert math.dist(track[-1, :2], path[-1]) < 0.05

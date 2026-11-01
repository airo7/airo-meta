import asyncio
import math

import numpy as np
import pytest

from airo.body import KEY, Body
from dataclasses import replace

from airo.control import RECENTRE_PWM, face, follow, goto, recentre
from airo.sim import SimRig

DRIFTY = Body(g_l=0.0029, g_r=0.0034, dz=28, eta=0.27, tau=0.25, lag=0.12, kappa=-0.15)


@pytest.mark.parametrize("truth, model, latency", [(KEY, KEY, 0), (DRIFTY, DRIFTY, 0.08), (DRIFTY, KEY, 0)])
def test_goto_reaches_ten_goals(truth, model, latency):
    async def run():
        rig = SimRig(truth, latency=latency, log_dir=None)
        goals = np.random.default_rng(0).uniform((0.3, 0.3), (1.7, 0.9), (10, 2))
        return [await goto(rig, model, g) for g in goals]

    assert max(asyncio.run(run())) < 0.05


def test_follow_sparse_path():
    path = [(0.4, 0.4), (1.6, 0.4), (1.6, 0.8), (0.6, 0.8)]

    async def run():
        rig = SimRig(DRIFTY, pose0=(0.4, 0.4, 0.0), log_dir=None)
        err = await follow(rig, KEY, path)
        return err, rig.segment().pose

    err, track = asyncio.run(run())
    assert err < 0.05
    assert track[:, 1].max() < 0.8 + 0.08 and track[:, 0].max() < 1.6 + 0.08
    assert math.dist(track[-1, :2], path[-1]) < 0.05


@pytest.mark.parametrize("theta", [math.pi / 2, -2.5, 0.1])
def test_face_turns_in_place(theta):
    async def run():
        rig = SimRig(DRIFTY, log_dir=None)
        await face(rig, KEY, theta)
        return rig

    rig = asyncio.run(run())
    assert abs((rig.pose.theta - theta + math.pi) % (2 * math.pi) - math.pi) < math.radians(10)
    assert math.dist((rig.pose.x, rig.pose.y), (1.0, 0.6)) < 0.05


def test_recentre_caps_power_when_the_body_is_wrong():
    async def run():
        rig = SimRig(KEY, pose0=(0.4, 0.4, 0.0), log_dir=None)
        await recentre(rig, replace(KEY, eta=KEY.eta / 4))
        return rig
    rig = asyncio.run(run())
    assert np.abs(rig.history[1]).max() <= RECENTRE_PWM
    assert math.dist((rig.pose.x, rig.pose.y), (rig.floor[0] / 2, rig.floor[1] / 2)) < 0.08


def test_pivots_get_the_floor_and_straights_do_not():
    from airo.body import inverse
    assert min(map(abs, inverse(KEY, 0, 0.3, pivot_min=70))) >= 70
    assert inverse(KEY, 0.1, 0, pivot_min=70) == inverse(KEY, 0.1, 0)

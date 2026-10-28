import asyncio
from dataclasses import replace

import numpy as np
import pytest

from airo.body import Body
from airo.probes import LIBRARY, SCHEDULE, commands, measure, predict, run, safe, zscore
from airo.sim import SimRig

TRUTH = Body(g_l=0.0030, g_r=0.0033, dz=25, eta=0.3, tau=0.2, lag=0.1, kappa=-0.1)


def test_library_and_schedule():
    assert {"step_40", "step_150", "step_back_110", "ramp_120", "pivot_l_60", "pivot_r_150", "arc_l", "arc_r",
            "pulse_100"} <= LIBRARY.keys()
    assert len(LIBRARY) == 17 and 8 <= len(SCHEDULE) <= 10 and set(SCHEDULE) <= LIBRARY.keys()


def test_commands():
    step = commands(LIBRARY["step_80"])
    assert step.shape == (32, 2) and (step[:20] == 80).all() and (step[20:] == 0).all()
    ramp = commands(LIBRARY["ramp_120"])[:, 0]
    assert ramp[0] == 0 and ramp[39] == 120 and (np.diff(ramp[:40]) >= 0).all() and (ramp[40:] == 0).all()
    assert tuple(commands(LIBRARY["pivot_l_100"])[0]) == (-100, 100)
    assert len(commands(LIBRARY["pulse_100"])) == 6 + 12


@pytest.mark.parametrize("name", ["step_110", "ramp_120", "pivot_l_100", "arc_r", "pulse_100", "step_back_60"])
def test_prediction_matches_measurement_on_truth(name):
    rig = SimRig(TRUTH, log_dir=None)
    p = rig.pose
    pred = predict([TRUTH], LIBRARY[name], (p.x, p.y, p.theta))
    meas = measure(asyncio.run(run(rig, LIBRARY[name])))
    assert max(map(abs, zscore(pred, meas).values())) < 1.5


def test_prediction_spread_and_sign():
    members = [replace(TRUTH, g_l=TRUTH.g_l * f, g_r=TRUTH.g_r * f) for f in (0.8, 1.0, 1.2)]
    pred = predict(members, LIBRARY["step_110"], (0, 0, 0))
    assert pred["forward_cm"][0] > 20 and pred["forward_cm"][1] > 2
    pivot = predict([TRUTH], LIBRARY["pivot_r_100"], (0, 0, 0))
    assert pivot["turn_deg"][0] < -45 and abs(pivot["forward_cm"][0]) < 3


def test_safe_near_the_edge():
    rig = SimRig(TRUTH, pose0=(1.6, 0.6, 0.0), log_dir=None)
    assert not safe(rig, LIBRARY["step_150"], TRUTH)
    assert safe(rig, LIBRARY["pivot_l_150"], TRUTH)
    assert safe(rig, LIBRARY["step_back_110"], TRUTH)
    assert safe(SimRig(TRUTH, log_dir=None), LIBRARY["step_150"], TRUTH)


def test_safe_accounts_for_coasting():
    rig = SimRig(TRUTH, pose0=(1.55, 0.6, 0.0), log_dir=None)

    async def drive():
        for _ in range(8):
            await rig.step(200, 200)

    asyncio.run(drive())
    at_rest = SimRig(TRUTH, pose0=(rig.pose.x, rig.pose.y, rig.pose.theta), log_dir=None)
    assert safe(at_rest, LIBRARY["pivot_l_100"], TRUTH) and not safe(rig, LIBRARY["pivot_l_100"], TRUTH)

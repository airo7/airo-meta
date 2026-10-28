import asyncio
import json
import math

import numpy as np
import pytest

from airo.body import KEY, PRIOR, TRACK, fit
from airo.camera import Pose
from airo.mjsim import MjRig
from airo.rig import DT, RigAbort
from airo.sim import SimRig, world


async def drive(rig, cmds):
    for l, r in cmds:
        await rig.step(l, r)


def probe(l, r, dur, rest=0.8, ramp=False, **kw):
    """One labelled probe from rest, the way a human runs it: 0.2 s idle, the command, then a stop."""
    rig = MjRig(**{"objects": (), "log_dir": None, "noise": (0, 0), **kw})
    n = round(dur / DT)
    cmds = [(0, 0)] * 4 + [(round(l * (i + 1) / n), round(r * (i + 1) / n)) if ramp else (l, r) for i in range(n)]
    asyncio.run(drive(rig, cmds + [(0, 0)] * round(rest / DT)))
    return rig.segment()


def rates(seg, t0, t1):
    """Steady speed and yaw rate from a straight-line fit over [t0, t1]."""
    k = (seg.t >= t0) & (seg.t <= t1)
    d = np.linalg.norm(seg.pose[k, :2] - seg.pose[k][0, :2], axis=1)
    return np.polyfit(seg.t[k], d, 1)[0], np.polyfit(seg.t[k], np.unwrap(seg.pose[k, 2]), 1)[0]


def calibrate(**kw) -> dict:
    """Step probes for gain and drift, a dead-zone ladder, a pulse for the delay, a pivot for eta."""
    ps = np.array([60, 110, 160])
    steps = [rates(probe(p, p, 1.5, **kw), 1.0, 1.7) for p in ps]
    v = np.array([s[0] for s in steps])
    gain = float(v @ ps / (ps @ ps))
    moved = [p for p in (20, 22, 24, 26, 28, 30, 32)
             if np.linalg.norm(np.ptp(probe(p, p, 1.0, **kw).pose[:, :2], axis=0)) > 0.02]
    pulse = probe(100, 100, 0.3, **kw)
    d = np.linalg.norm(pulse.pose[:, :2] - pulse.pose[0, :2], axis=1)
    _, yaw = rates(probe(-100, 100, 1.5, **kw), 1.0, 1.7)
    return dict(gain=gain, dead_zone=moved[0], delay=float(pulse.t[np.argmax(d > 0.001)] - 4 * DT),
                eta=float(yaw / (2 * gain * 100 / TRACK)), kappa=float(steps[1][1] / v[1]), speed_110=float(v[1]))


@pytest.fixture(scope="module")
def cal():
    bare, heavy = calibrate(), calibrate(weight=0.4)
    print("\nbare ", {k: round(v, 5) for k, v in bare.items()}, "\nheavy", {k: round(v, 5) for k, v in heavy.items()})
    return bare, heavy


def test_model_loads():
    assert MjRig(objects=(), log_dir=None).m.body_subtreemass[1] == pytest.approx(0.94)
    rig = MjRig(bumper=True, log_dir=None)
    assert rig.m.nu == 4 and {o.concept for o in rig.objects} >= {"red paper", "blue paper", "cardboard box"}
    assert rig.frame().shape == (1080, 1920, 3)
    assert rig.view(320, 180).shape == (180, 320, 3)


def test_interface_matches_simrig(tmp_path):
    rigs = [SimRig(KEY, log_dir=tmp_path / "a"), MjRig(objects=(), log_dir=tmp_path / "b")]
    for rig in rigs:
        assert isinstance(asyncio.run(rig.step(80, 80)), Pose)
        seg = rig.segment()
        assert seg.cmd.shape == (1, 2) and seg.pose.shape == (1, 3) and rig.H.shape == (3, 3)
        assert rig.inside(1.0, 0.6) and not rig.inside(0.05, 0.6)
        with pytest.raises(RigAbort, match="inset"):
            asyncio.run(drive(rig, [(200, 200)] * 200))
        assert tuple(rig.history[1][-1]) == (0, 0)
        asyncio.run(rig.close())
    last = [json.loads(next((tmp_path / d).glob("*.jsonl")).read_text().splitlines()[-1]) for d in "ab"]
    assert last[0].keys() == last[1].keys()


@pytest.mark.parametrize("pose0", [(1.0, 0.6, 0.0), (0.4, 0.3, 2.0), (1.7, 0.95, -2.5)])
def test_aruco_pose_within_a_centimetre(pose0):
    rig = MjRig(pose0=pose0, source="aruco", log_dir=None)
    asyncio.run(drive(rig, [(90, 140)] * 6))
    x, y, th = rig.true_pose
    assert rig.pose.t == rig.t and rig.tracker.residual_m < 0.002
    assert math.hypot(rig.pose.x - x, rig.pose.y - y) < 0.01
    assert abs(math.remainder(rig.pose.theta - th, math.tau)) < math.radians(1)


def test_calibration_probes(cal):
    bare, _ = cal
    assert bare["gain"] == pytest.approx(0.00316, rel=0.08)
    assert 22 <= bare["dead_zone"] <= 30
    assert 0.15 <= bare["delay"] <= 0.27
    assert 0.22 <= bare["eta"] <= 0.38
    assert -0.1 < bare["kappa"] < 0


def test_add_weight_lowers_gain_and_eta(cal):
    bare, heavy = cal
    assert heavy["gain"] < 0.95 * bare["gain"]
    assert heavy["eta"] < bare["eta"]


def test_body_fit_recovers_the_physics():
    probes = [(p, p, 1) for p in (40, 60, 80, 110, 150, -60, -110)] + [(p, -p, 1) for p in (60, 100, 150)] + \
             [(120, 60, 1), (60, 120, 1), (100, 100, 0.3)]
    segs = [probe(*p, noise=(0.002, math.radians(0.3))) for p in probes] + [probe(120, 120, 2, ramp=True)]
    got = fit(segs, PRIOR)
    print("\nfit", got)
    assert (got.g_l + got.g_r) / 2 == pytest.approx(0.00316, rel=0.1)
    assert got.g_l > got.g_r and 18 <= got.dz <= 33 and 0.22 <= got.eta <= 0.38 and 0.15 <= got.tau <= 0.25


def test_hidden_fault_tow_and_patch():
    flipped = probe(100, 100, 1.0, flip="l")
    assert abs(flipped.pose[-1, 2]) > 0.5 and np.ptp(flipped.pose[:, 0]) < 0.05

    rig = MjRig(objects=(), pose0=(0.6, 0.6, 0.0), tow=True, log_dir=None)
    x0 = rig.d.body("tow").xpos[0]
    asyncio.run(drive(rig, [(120, 120)] * 30))
    gap = np.linalg.norm(rig.d.body("tow").xpos[:2] - rig.true_pose[:2])
    assert rig.d.body("tow").xpos[0] - x0 > 0.25 and gap < 0.118 + 0.25 + 0.07
    before = rig.true_pose
    rig.add_weight(0.4)
    assert np.allclose(rig.true_pose, before) and rig.m.body_subtreemass[1] == pytest.approx(0.94 + 0.4)
    asyncio.run(drive(rig, [(120, 120)] * 2))

    pivot = [rates(probe(-100, 100, 1.5, **kw), 1.0, 1.7)[1] for kw in ({}, {"patch": (0.5, 0.2, 1.5, 1.0)})]
    assert pivot[1] > 1.3 * pivot[0]


def test_sam_stand_in_and_ghosts():
    rig = MjRig(objects=world(), log_dir=None)
    frame = rig.frame()
    boxes = asyncio.run(rig.sam(frame, "cardboard box"))
    assert len(boxes) == 2 and all(s.mask.shape == frame.shape[:2] and s.mask.sum() > 1000 for s in boxes)
    plain = rig.view(320, 180)
    rig.render_ghosts(np.array([[[1.0, 0.6, 0.0], [1.3, 0.6 + dy, 0.0]] for dy in (-0.05, 0, 0.05)]), 0.2)
    assert np.abs(rig.view(320, 180).astype(int) - plain).sum() > 0

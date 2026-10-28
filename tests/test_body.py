import math
from dataclasses import replace

import numpy as np
import pytest

from airo.body import KEY, PRIOR, Body, Segment, ensemble, fit, inverse, rollout, surprise

TRUTH = Body(g_l=0.0030, g_r=0.0033, dz=25, eta=0.3, tau=0.2, lag=0.1, kappa=-0.1)
PROBES = [(p, p, 1) for p in (40, 60, 80, 110, 150, -60, -110)] + \
         [(p, -p, 1) for p in (60, 100, 150)] + [(-100, 100, 1), (120, 60, 1), (60, 120, 1), (100, 100, 0.3)]


def probe(body, l, r, dur, rng, rest=0.8):
    t = np.arange(0, dur + rest, 0.05)
    cmd = np.where((t < dur)[:, None], [l, r], 0)
    pose = rollout(body, t + 0.05, t, cmd, (1.0, 1.0, 0.3), wheels0=(0, 0))
    pose += rng.normal(0, [0.002, 0.002, math.radians(0.3)], pose.shape)
    return Segment(t, cmd, t + 0.05, pose)


def ramp(body, rng):
    t = np.arange(0, 2.5, 0.05)
    cmd = np.repeat(np.clip(t / 2 * 120, 0, 120)[:, None], 2, axis=1) * (t < 2)[:, None]
    pose = rollout(body, t, t, cmd, (1.0, 1.0, 0.0), wheels0=(0, 0))
    return Segment(t, cmd, t, pose + rng.normal(0, 0.002, pose.shape))


def data(body, seed=0):
    rng = np.random.default_rng(seed)
    return [probe(body, *p, rng) for p in PROBES] + [ramp(body, rng)]


@pytest.mark.parametrize("seed", [0, 3])
def test_fit_recovers_truth_from_prior(seed):
    got = fit(data(TRUTH, seed), PRIOR)
    for name in ("g_l", "g_r", "dz", "eta"):
        assert getattr(got, name) == pytest.approx(getattr(TRUTH, name), rel=0.1), name
    # Delay and lag trade off at 20 Hz, and drift is nearly collinear with the gain imbalance.
    assert got.tau + got.lag == pytest.approx(TRUTH.tau + TRUTH.lag, rel=0.1)
    assert 0.15 <= got.tau <= 0.25
    assert got.kappa == pytest.approx(TRUTH.kappa, abs=0.03)


def test_fit_masks_missing_poses():
    segs = data(TRUTH)
    for s in segs:
        s.pose[5:9] = np.nan
    assert fit(segs, KEY).g_l == pytest.approx(TRUTH.g_l, rel=0.05)


def test_delay_holds_still_until_tau():
    t = np.arange(0, 1, 0.01)
    pose = rollout(TRUTH, t, [0], [(150, 150)], (0, 0, 0), wheels0=(0, 0))
    assert np.abs(pose[t < TRUTH.tau]).max() < 1e-9
    assert pose[-1, 0] > 0.1


@pytest.mark.parametrize("v, om", [(0.2, 0), (0.2, 0.4), (0.25, -0.5), (0, 1.5), (-0.15, 0)])
def test_inverse_matches_rollout(v, om):
    t = np.arange(0, 3, 0.01)
    pose = rollout(TRUTH, t, [0], [inverse(TRUTH, v, om)], (0, 0, 0), wheels0=(0, 0))[-101:]
    step = np.diff(pose, axis=0)
    fwd = step[:, 0] * np.cos(pose[1:, 2]) + step[:, 1] * np.sin(pose[1:, 2])
    assert fwd.sum() == pytest.approx(v, abs=0.01)
    assert step[:, 2].sum() == pytest.approx(om, abs=0.03)


def test_inverse_bumps_past_dead_zone_and_clips():
    l, r = inverse(KEY, 0.005, 0)
    assert l == r and KEY.dz < l < KEY.dz + 5
    assert inverse(KEY, 0, 0) == (0, 0)
    assert inverse(KEY, 5, 0) == (230, 230)


def test_surprise_low_on_own_body_high_on_changed_body():
    segs = data(TRUTH)
    members = ensemble(segs[:6], TRUTH, k=5)
    assert np.nanmedian(np.concatenate([surprise(members, s) for s in segs[:5]])) < 3
    heavy = data(replace(TRUTH, g_l=TRUTH.g_l * 0.6, g_r=TRUTH.g_r * 0.6), seed=1)
    assert np.nanmedian(np.concatenate([surprise(members, s) for s in heavy[:5]])) > 3

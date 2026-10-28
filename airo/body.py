"""Learned body of the skid-steer car: per-side gain with a soft dead zone, wheel lag, command delay, an
effective track (eta) and drift (kappa)."""

import json
import math
from dataclasses import asdict, astuple, dataclass, replace
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.special import expit

from airo.protocol import PWM_MAX

BODY_PATH = Path("data/body.json")
SIM_BODY_PATH = Path("data/body_sim.json")
TRACK = 0.145
DT = 0.01
WARMUP = 2.5
HEADING_W = 0.1
ANCHOR_W = 0.03


@dataclass(frozen=True)
class Body:
    g_l: float
    g_r: float
    dz: float
    eta: float
    tau: float
    lag: float
    kappa: float

    def wheel_target(self, cmd: np.ndarray) -> np.ndarray:
        p = np.asarray(cmd, float)
        return np.array([self.g_l, self.g_r]) * p * expit((np.abs(p) - self.dz) / 2)


PRIOR = Body(g_l=0.0025, g_r=0.0025, dz=10, eta=1.0, tau=0.1, lag=0.1, kappa=0.0)
KEY = Body(g_l=0.00316, g_r=0.00316, dz=25, eta=0.3, tau=0.2, lag=0.1, kappa=-0.05)
LOWER = Body(-0.02, -0.02, 0, 0.05, 0, 0.01, -2)
UPPER = Body(0.02, 0.02, 120, 1.5, 0.6, 0.6, 2)
SCALE = np.array([1e-3, 1e-3, 10, 0.3, 0.1, 0.1, 0.1])
PRIOR_SD = np.array([1.5e-3, 1.5e-3, 15, 0.4, 0.15, 0.1, 0.2])
# Calibrated in sim for ~95% predictive coverage over 30 seeds (tests/test_body.py checks it). The Laplace fit
# misses each segment's noisy start pose and the curved tau-lag ridge, so the residual variance needs 12-20x.
INFLATE = 14.0
# tau and lag move the command interpolation by whole grid kinks, so their steps span one DT.
JAC_STEP = np.array([1e-5, 1e-5, 0.1, 3e-3, DT, DT, 1e-3])


@dataclass
class Segment:
    cmd_t: np.ndarray
    cmd: np.ndarray
    t: np.ndarray
    pose: np.ndarray


def wrap(a):
    return (np.asarray(a) + math.pi) % (2 * math.pi) - math.pi


def command_at(cmd_t, cmd, q) -> np.ndarray:
    cmd = np.asarray(cmd, float).reshape(-1, 2)
    if not len(cmd):
        return np.zeros((np.size(q), 2))
    return np.column_stack([np.interp(q, cmd_t, cmd[:, i], left=0) for i in range(2)])


def _integral(y, g, t):
    c = cumulative_trapezoid(y, g, initial=0)
    return np.interp(t, g, c) - np.interp(t[0], g, c)


def rollout(body: Body, t, cmd_t, cmd, pose0, wheels0=None) -> np.ndarray:
    """Poses (N, 3) on the grid t from pose0 at t[0], heading unwrapped.

    With wheels0=None the wheel state is warmed up on the command history before t[0].
    """
    t = np.asarray(t, float)
    t0 = t[0] - (WARMUP if wheels0 is None else 0)
    g = np.linspace(t0, t[-1], max(2, math.ceil((t[-1] - t0) / DT) + 1))
    a = math.exp(-(g[1] - g[0]) / body.lag)
    w0 = np.zeros(2) if wheels0 is None else np.asarray(wheels0, float)
    target = body.wheel_target(command_at(cmd_t, cmd, g - body.tau))
    w, _ = lfilter([(1 - a) / 2] * 2, [1, -a], target, axis=0, zi=(a * w0 + (1 - a) / 2 * target[0])[None])

    v = w.mean(axis=1)
    om = body.eta * (w[:, 1] - w[:, 0]) / TRACK + body.kappa * v
    c = cumulative_trapezoid(om, g, initial=0)
    th = pose0[2] + c - np.interp(t[0], g, c)
    return np.column_stack([pose0[0] + _integral(v * np.cos(th), g, t),
                            pose0[1] + _integral(v * np.sin(th), g, t),
                            np.interp(t, g, th)])


def _residual(body: Body, s: Segment) -> np.ndarray:
    ok = ~np.isnan(s.pose).any(axis=1)
    t, pose = s.t[ok], s.pose[ok]
    d = rollout(body, t, s.cmd_t, s.cmd, pose[0]) - pose
    return np.concatenate([d[:, :2].ravel(), HEADING_W * wrap(d[:, 2])])


def fit(segments: list[Segment], x0: Body = PRIOR, taus: tuple[float, ...] = (0.05, 0.15, 0.3),
        anchor: Body | None = None, weight: float = ANCHOR_W, signs: bool = False) -> Body:
    """Least squares from x0, plus restarts at each tau in taus: tau and lag trade off into local minima.

    anchor adds a weak pull toward a previous body, per SCALE unit, so parameters the segments say little about
    (the dead zone after a few pivots) stay put instead of wandering. signs also restarts from the best fit with
    each motor reversed: gradient descent cannot pass a gain through zero, where that side stops responding.
    """
    lo, hi = np.array(astuple(LOWER)), np.array(astuple(UPPER))

    def residuals(x):
        r = [_residual(Body(*x), s) for s in segments]
        if anchor is not None:
            r.append(weight * (x - astuple(anchor)) / SCALE)
        return np.concatenate(r)

    def run(start: Body):
        return least_squares(residuals, np.clip(astuple(start), lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                             x_scale=SCALE)

    best = min((run(s) for s in [x0, *(replace(x0, tau=t) for t in taus)]), key=lambda r: r.cost)
    if signs:
        b = Body(*best.x)
        flips = [replace(b, g_l=-b.g_l), replace(b, g_r=-b.g_r), replace(b, g_l=-b.g_l, g_r=-b.g_r)]
        best = min([best, *map(run, flips)], key=lambda r: r.cost)
    return Body(*map(float, best.x))


def _corr_time(r: np.ndarray) -> float:
    """Integrated autocorrelation time in samples, summed up to the first non-positive lag."""
    r = r - r.mean()
    if len(r) < 4 or not r.any():
        return 1.0
    ac = np.correlate(r, r, "full")[len(r) - 1:] / (r @ r)
    stop = np.nonzero(ac[1:] <= 0)[0]
    return 1 + 2 * ac[1:(stop[0] + 1 if len(stop) else len(r))].sum()


def _excited(segments: list[Segment], best: Body) -> np.ndarray:
    """Which parameters the segments' commands can inform. Gains, tau and lag need any motion, dz a command near
    the dead zone, eta and kappa a turn: straight runs only see eta * (g_r - g_l) and kappa together."""
    cmd = np.abs(np.concatenate([s.cmd[(s.cmd_t >= s.t[0]) & (s.cmd_t <= s.t[-1])] for s in segments]))
    p = cmd.max(axis=1)
    moving, low = (p > 0).any(), ((p > 0) & (p < best.dz + 20)).any()
    turn = (np.abs(cmd[:, 0] - cmd[:, 1]) > 20).any()
    return np.array([moving, moving, low, turn, moving, moving, turn])


def posterior(segments: list[Segment], best: Body, anchor: Body | None = None) -> np.ndarray:
    """Laplace covariance at best: the data and anchor Jacobian, plus a broad prior. Parameters no command has
    excited keep their prior spread. The noise variance is inflated by INFLATE times the residuals' autocorrelation
    time."""
    x = np.array(astuple(best))
    precision = np.diag(1 / PRIOR_SD ** 2)
    if not segments:
        return np.linalg.inv(precision)

    def residuals(x):
        per = [_residual(Body(*x), s) for s in segments]
        return np.concatenate(per), (ANCHOR_W * (x - astuple(anchor)) / SCALE if anchor else np.zeros(0)), per

    data, _, per = residuals(x)
    corr = np.mean([_corr_time(r[i:2 * (len(r) // 3):2]) for r in per for i in (0, 1)])
    h = JAC_STEP
    J = np.column_stack([(np.concatenate(residuals(x + e)[:2]) - np.concatenate(residuals(x - e)[:2])) / (2 * hi)
                         for e, hi in zip(np.diag(h), h)])
    J[:len(data), ~_excited(segments, best)] = 0
    s2 = INFLATE * corr * (data ** 2).sum() / max(1, len(data) - len(x))
    return np.linalg.inv(J.T @ J / s2 + precision)


def ensemble(segments: list[Segment], best: Body, k: int = 20, seed: int = 0, anchor: Body | None = None) -> list[Body]:
    """k draws from the Laplace posterior around best, clipped to the bounds."""
    draws = np.random.default_rng(seed).multivariate_normal(astuple(best), posterior(segments, best, anchor), k)
    return [Body(*map(float, d)) for d in np.clip(draws, astuple(LOWER), astuple(UPPER))]


def cloud(members: list[Body], t, cmd_t, cmd, pose0, wheels0=None) -> np.ndarray:
    return np.stack([rollout(b, t, cmd_t, cmd, pose0, wheels0) for b in members])


def surprise(members: list[Body], seg: Segment, window: float = 0.5) -> np.ndarray:
    """One score per window from seg.t[0]; NaN where a window has a pose dropout."""
    out = []
    for a in np.arange(seg.t[0], seg.t[-1] - window + 1e-9, window):
        i, j = np.searchsorted(seg.t, a), np.searchsorted(seg.t, a + window, "right") - 1
        p = seg.pose[i:j + 1]
        if j <= i or np.isnan(p).any():
            out.append(math.nan)
            continue
        d = cloud(members, seg.t[[i, j]], seg.cmd_t, seg.cmd, p[0])
        d = d[:, 1] - d[:, 0]
        m = d.mean(axis=0)
        var = ((d[:, :2] - m[:2]) ** 2).sum(axis=1).mean()
        s_xy = np.linalg.norm(p[-1, :2] - p[0, :2] - m[:2]) / math.sqrt(var + 0.01 ** 2)
        s_th = abs(wrap(p[-1, 2] - p[0, 2] - m[2])) / math.sqrt(d[:, 2].var() + 0.03 ** 2)
        out.append(max(s_xy, s_th))
    return np.array(out)


def cloud_radius_cm(segments: list[Segment], best: Body, anchor: Body | None = None) -> float:
    """Spread of where the posterior puts the car after 1.5 s of a gentle left arc from rest; 200 draws keep the
    Monte Carlo noise well under the changes between probes."""
    t = np.linspace(0, 1.5, 16)
    end = cloud(ensemble(segments, best, 200, anchor=anchor), t, [0], [(100, 140)], (0, 0, 0), wheels0=(0, 0))
    end = end[:, -1, :2]
    return 100 * math.sqrt(((end - end.mean(axis=0)) ** 2).sum(axis=1).mean())


_P = np.linspace(0, PWM_MAX, 4 * PWM_MAX + 1)


def inverse(body: Body, v: float, omega: float, bump: float = 3) -> tuple[int, int]:
    """PWM for steady-state (v, omega); non-zero sides are bumped just past the dead zone."""
    d = (omega - body.kappa * v) * TRACK / body.eta
    w = np.array([v - d / 2, v + d / 2])
    speed = np.abs(body.wheel_target(np.column_stack([_P, _P])))
    p = np.array([np.interp(abs(w[i]), speed[:, i], _P) for i in range(2)])
    p = np.where(np.abs(w) > 1e-3, np.maximum(p, body.dz + bump), 0)
    l, r = np.clip(np.sign(w) * np.sign([body.g_l, body.g_r]) * p, -PWM_MAX, PWM_MAX).round().astype(int)
    return int(l), int(r)


def save(body: Body, members: list[Body], name: str, path: Path = BODY_PATH):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"name": name, "body": asdict(body), "members": [asdict(m) for m in members]}, indent=1))


def load(path: Path = BODY_PATH) -> tuple[Body, list[Body], str] | None:
    if not Path(path).exists():
        return None
    d = json.loads(Path(path).read_text())
    return Body(**d["body"]), [Body(**m) for m in d["members"]] or [Body(**d["body"])], d["name"]

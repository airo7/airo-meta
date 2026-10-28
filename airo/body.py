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
LOWER = Body(1e-4, 1e-4, 0, 0.05, 0, 0.01, -2)
UPPER = Body(0.02, 0.02, 120, 1.5, 0.6, 0.6, 2)
SCALE = np.array([1e-3, 1e-3, 10, 0.3, 0.1, 0.1, 0.1])


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
        anchor: Body | None = None, weight: float = ANCHOR_W) -> Body:
    """Least squares from x0, plus restarts at each tau in taus: tau and lag trade off into local minima.

    anchor adds a weak pull toward a previous body, per SCALE unit, so parameters the segments say little about
    (the dead zone after a few pivots) stay put instead of wandering.
    """
    lo, hi = np.array(astuple(LOWER)), np.array(astuple(UPPER))

    def residuals(x):
        r = [_residual(Body(*x), s) for s in segments]
        if anchor is not None:
            r.append(weight * (x - astuple(anchor)) / SCALE)
        return np.concatenate(r)

    best = None
    for start in [x0, *(replace(x0, tau=t) for t in taus)]:
        r = least_squares(residuals, np.clip(astuple(start), lo + 1e-9, hi - 1e-9), bounds=(lo, hi), x_scale=SCALE)
        if best is None or r.cost < best.cost:
            best = r
    return Body(*map(float, best.x))


def ensemble(segments: list[Segment], best: Body, k: int = 20, seed: int = 0, anchor: Body | None = None) -> list[Body]:
    rng = np.random.default_rng(seed)
    n = len(segments)
    return [fit([segments[i] for i in rng.integers(n, size=n)], best, taus=(), anchor=anchor) for _ in range(k)]


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


_P = np.linspace(0, PWM_MAX, 4 * PWM_MAX + 1)


def inverse(body: Body, v: float, omega: float, bump: float = 3) -> tuple[int, int]:
    """PWM for steady-state (v, omega); non-zero sides are bumped just past the dead zone."""
    d = (omega - body.kappa * v) * TRACK / body.eta
    w = np.array([v - d / 2, v + d / 2])
    speed = body.wheel_target(np.column_stack([_P, _P]))
    p = np.array([np.interp(abs(w[i]), speed[:, i], _P) for i in range(2)])
    p = np.where(np.abs(w) > 1e-3, np.maximum(p, body.dz + bump), 0)
    l, r = np.clip(np.sign(w) * p, -PWM_MAX, PWM_MAX).round().astype(int)
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

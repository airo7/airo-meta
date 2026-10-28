"""Open-loop probe experiments: the library, a free-space check, a runner, and predicted vs measured outcomes."""

import math
from dataclasses import dataclass, replace
from typing import Awaitable, Callable

import numpy as np

from airo.body import Body, Segment, cloud, rollout
from airo.rig import DT, BaseRig

QTY = ("forward_cm", "lateral_cm", "turn_deg")
FLOOR = np.array([1.0, 1.0, 2.0])

Recentre = Callable[[BaseRig], Awaitable]


@dataclass(frozen=True)
class Probe:
    name: str
    kind: str
    pwm_l: int
    pwm_r: int
    seconds: float
    settle: float = 0.6


LIBRARY = {p.name: p for p in [
    *(Probe(f"step_{p}", "step", p, p, 1) for p in (40, 60, 80, 110, 150)),
    *(Probe(f"step_back_{p}", "step", -p, -p, 1) for p in (60, 110)),
    Probe("ramp_120", "ramp", 120, 120, 2),
    *(Probe(f"pivot_{s}_{p}", "pivot", -p * d, p * d, 1) for s, d in (("l", 1), ("r", -1)) for p in (60, 100, 150)),
    Probe("arc_l", "arc", 60, 120, 1),
    Probe("arc_r", "arc", 120, 60, 1),
    Probe("pulse_100", "pulse", 100, 100, 0.3),
]}

SCHEDULE = ["ramp_120", "step_60", "step_110", "step_150", "step_back_110", "pivot_l_100", "pivot_r_150",
            "pivot_l_60", "arc_r", "pulse_100"]


def commands(probe: Probe, dt: float = DT) -> np.ndarray:
    n = round(probe.seconds / dt)
    f = np.linspace(0, 1, n) if probe.kind == "ramp" else np.ones(n)
    on = np.outer(f, (probe.pwm_l, probe.pwm_r)).round()
    return np.vstack([on, np.zeros((round(probe.settle / dt), 2))]).astype(int)


def safe(rig: BaseRig, probe: Probe, body: Body, speed: float = 1.3, margin: float = 0.05) -> bool:
    """Rolls out the command history too, so a car still coasting from earlier commands is accounted for."""
    fast = replace(body, g_l=body.g_l * speed, g_r=body.g_r * speed)
    p = rig.pose
    cmd_t, cmd = rig.history
    t0 = max(p.t, cmd_t[-1] + DT / 2) if len(cmd_t) else p.t
    new = commands(probe)
    t = t0 + np.arange(len(new) + 1) * DT
    path = rollout(fast, np.r_[p.t, t], np.r_[cmd_t, t[:-1]], np.vstack([cmd, new]), (p.x, p.y, p.theta))
    return all(rig.inside(x, y, margin) for x, y, _ in path)


async def halt(rig: BaseRig, seconds: float = 0.5):
    for _ in range(round(seconds / DT)):
        await rig.step(0, 0)


async def run(rig: BaseRig, probe: Probe) -> Segment:
    t0 = rig.pose.t
    for l, r in commands(probe):
        await rig.step(int(l), int(r))
    # Start at the first pose after the first command, as predict() assumes; the pose at t0 may predate a recentre.
    return rig.segment(t0 + DT / 2)


def _local(start, end) -> np.ndarray:
    """(forward_cm, lateral_cm, turn_deg) of end poses (..., 3) in the frame of start; headings unwrapped."""
    d = end[..., :2] - start[:2]
    c, s = math.cos(start[2]), math.sin(start[2])
    return np.stack([100 * (d[..., 0] * c + d[..., 1] * s), 100 * (d[..., 1] * c - d[..., 0] * s),
                     np.degrees(end[..., 2] - start[2])], axis=-1)


def outcome(members: list[Body], t, cmd_t, cmd, pose0, wheels0=None) -> dict:
    """Ensemble mean and sd of the motion from t[0] to t[-1]."""
    q = _local(np.asarray(pose0, float), cloud(members, [t[0], t[-1]], cmd_t, cmd, pose0, wheels0)[:, -1])
    return {k: (round(float(m), 2), round(float(s), 2)) for k, m, s in zip(QTY, q.mean(axis=0), q.std(axis=0))}


def predict(members: list[Body], probe: Probe, pose0) -> dict:
    # body.command_at interpolates between samples, and the rig history holds a zero tick before each probe.
    cmd = np.vstack([(0, 0), commands(probe)])
    t = np.arange(-1, len(cmd) - 1) * DT
    return outcome(members, [DT, t[-1] + DT], t, cmd, pose0, wheels0=(0, 0))


def valid(seg: Segment) -> tuple[np.ndarray, np.ndarray]:
    ok = ~np.isnan(seg.pose).any(axis=1)
    t, pose = seg.t[ok], seg.pose[ok].copy()
    pose[:, 2] = np.unwrap(pose[:, 2])
    return t, pose


def measure(seg: Segment) -> dict:
    _, pose = valid(seg)
    return {k: round(float(v), 2) for k, v in zip(QTY, _local(pose[0], pose[-1]))}


def zscore(predicted: dict, measured: dict) -> dict:
    return {k: round(float((measured[k] - predicted[k][0]) / math.hypot(predicted[k][1], f)), 1)
            for k, f in zip(QTY, FLOOR)}

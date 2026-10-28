"""Delay-compensated pure pursuit: steer from the pose predicted one delay ahead, pivot in place on large heading
errors, slow down near the goal."""

import math

import numpy as np

from airo.body import Body, inverse, rollout, wrap
from airo.rig import HZ, BaseRig

V_MAX = 0.25
V_MIN = 0.08
OM_MAX = 2.0
PIVOT = math.radians(60)
RESUME = math.radians(20)
SLOW_R = 0.3
LOOKAHEAD = 0.15


def predict(rig: BaseRig, body: Body, horizon: float | None = None, stop: bool = False) -> np.ndarray:
    """Pose after horizon (default tau) given the commands already sent, and zero from now if stop."""
    p = rig.pose
    cmd_t, cmd = rig.history
    if stop:
        cmd_t, cmd = np.r_[cmd_t, p.t], np.vstack([cmd, (0, 0)])
    h = body.tau if horizon is None else horizon
    return rollout(body, [p.t, p.t + h], cmd_t, cmd, (p.x, p.y, p.theta))[-1]


def _steer(pose, target, v: float, pivoting: bool) -> tuple[float, float, bool]:
    d = np.asarray(target) - pose[:2]
    dist = math.hypot(*d)
    alpha = float(wrap(math.atan2(d[1], d[0]) - pose[2]))
    if abs(alpha) > (RESUME if pivoting else PIVOT):
        return 0.0, float(np.clip(3 * alpha, -OM_MAX, OM_MAX)), True
    return v, float(np.clip(2 * v * math.sin(alpha) / max(dist, 1e-3), -OM_MAX, OM_MAX)), False


async def _settle(rig: BaseRig, seconds: float = 0.5):
    for _ in range(round(seconds * HZ)):
        await rig.step(0, 0)


async def goto(rig: BaseRig, body: Body, xy, tol: float = 0.04, timeout: float = 30) -> float:
    """Drive to xy and stop; returns the final distance once the car has settled."""
    goal = np.asarray(xy, float)
    pivoting = False
    for _ in range(round(timeout * HZ)):
        if math.dist(predict(rig, body, body.tau + 3 * body.lag, stop=True)[:2], goal) < tol / 2:
            break
        pose = predict(rig, body)
        v = float(np.clip(V_MAX * math.dist(pose[:2], goal) / SLOW_R, V_MIN, V_MAX))
        v, om, pivoting = _steer(pose, goal, v, pivoting)
        await rig.step(*inverse(body, v, om))
    await _settle(rig)
    return math.dist((rig.pose.x, rig.pose.y), goal)


async def follow(rig: BaseRig, body: Body, path, tol: float = 0.04, lookahead: float = LOOKAHEAD,
                 timeout: float = 60) -> float:
    """Pure pursuit along path with a fixed lookahead, then goto its last point."""
    path = np.asarray(path, float)
    i, pivoting = 0, False
    for _ in range(round(timeout * HZ)):
        pose = predict(rig, body)
        dist = np.hypot(*(path[i:] - pose[:2]).T)
        i += int(np.argmin(dist))
        ahead = np.nonzero(dist[dist.argmin():] >= lookahead)[0]
        if math.dist(pose[:2], path[-1]) < lookahead or not len(ahead):
            break
        v, om, pivoting = _steer(pose, path[i + ahead[0]], V_MAX, pivoting)
        await rig.step(*inverse(body, v, om))
    return await goto(rig, body, path[-1], tol, timeout)

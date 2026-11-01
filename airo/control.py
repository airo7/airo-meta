"""Delay-compensated pure pursuit: steer from the pose predicted one delay ahead, pivot in place on large heading
errors, slow down near the goal."""

import math

import numpy as np

from airo.body import Body, inverse, rollout, wrap
from airo.rig import HZ, BaseRig

V_MAX = 0.25
V_MIN = 0.08
OM_MAX = 2.0
RECENTRE_PWM = 160
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


def _steer(pose, target, v: float, pivoting: bool, om_max: float = OM_MAX) -> tuple[float, float, bool]:
    d = np.asarray(target) - pose[:2]
    dist = math.hypot(*d)
    alpha = float(wrap(math.atan2(d[1], d[0]) - pose[2]))
    if abs(alpha) > (RESUME if pivoting else PIVOT):
        return 0.0, float(np.clip(3 * alpha, -om_max, om_max)), True
    return v, float(np.clip(2 * v * math.sin(alpha) / max(dist, 1e-3), -om_max, om_max)), False


async def _settle(rig: BaseRig, seconds: float = 0.5):
    for _ in range(round(seconds * HZ)):
        await rig.step(0, 0)


async def goto(rig: BaseRig, body: Body, xy, tol: float = 0.04, timeout: float = 30, v_max: float = V_MAX,
               om_max: float = OM_MAX) -> float:
    """Drive to xy and stop; returns the final distance once the car has settled."""
    goal = np.asarray(xy, float)
    pivoting = False
    for _ in range(round(timeout * HZ)):
        if math.dist(predict(rig, body, body.tau + 3 * body.lag, stop=True)[:2], goal) < tol / 2:
            break
        pose = predict(rig, body)
        v = float(np.clip(v_max * math.dist(pose[:2], goal) / SLOW_R, V_MIN, v_max))
        v, om, pivoting = _steer(pose, goal, v, pivoting, om_max)
        await rig.step(*inverse(body, v, om, pivot_min=rig.pivot_min, drive_min=rig.drive_min))
    await _settle(rig)
    return math.dist((rig.pose.x, rig.pose.y), goal)


async def face(rig: BaseRig, body: Body, theta: float, tol: float = math.radians(8), timeout: float = 10,
               om_max: float = OM_MAX):
    """Pivot in place until the car will come to rest within tol of heading theta."""
    for _ in range(round(timeout * HZ)):
        err = float(wrap(theta - predict(rig, body, body.tau + 3 * body.lag, stop=True)[2]))
        if abs(err) < tol:
            break
        om = float(np.clip(3 * err, -om_max, om_max))
        await rig.step(*inverse(body, 0, om, pivot_min=rig.pivot_min, drive_min=rig.drive_min))
    await _settle(rig)


class _Capped:
    """Scales commands down to cap: a body refitted on a few seconds can ask for full power, and a heavy car then
    overshoots every turn and swings back and forth."""

    def __init__(self, rig: BaseRig, cap: int):
        self._rig, self._cap = rig, cap

    def __getattr__(self, name):
        return getattr(self._rig, name)

    async def step(self, pwm_l: int, pwm_r: int):
        k = min(1.0, self._cap / max(abs(pwm_l), abs(pwm_r), 1))
        return await self._rig.step(round(pwm_l * k), round(pwm_r * k))


async def recentre(rig: BaseRig, body: Body, cap: int = RECENTRE_PWM):
    """Floor centre, then face along the long axis in whichever direction is nearer: the long probes fit there."""
    rig = _Capped(rig, cap)
    await goto(rig, body, (rig.floor[0] / 2, rig.floor[1] / 2))
    axis = 0.0 if rig.floor[0] >= rig.floor[1] else math.pi / 2
    await face(rig, body, axis if abs(wrap(rig.pose.theta - axis)) <= math.pi / 2 else axis + math.pi)


def _densify(path: np.ndarray, step: float = 0.02) -> np.ndarray:
    s = np.r_[0, np.cumsum(np.hypot(*np.diff(path, axis=0).T))]
    q = np.linspace(0, s[-1], max(2, math.ceil(s[-1] / step) + 1))
    return np.column_stack([np.interp(q, s, path[:, 0]), np.interp(q, s, path[:, 1])])


async def follow(rig: BaseRig, body: Body, path, tol: float = 0.04, lookahead: float = LOOKAHEAD,
                 timeout: float = 60, v_max: float = V_MAX, om_max: float = OM_MAX) -> float:
    """Pure pursuit along path with a fixed lookahead, then goto its last point."""
    path = _densify(np.asarray(path, float))
    i, pivoting = 0, False
    for _ in range(round(timeout * HZ)):
        pose = predict(rig, body)
        dist = np.hypot(*(path[i:] - pose[:2]).T)
        i += int(np.argmin(dist))
        ahead = np.nonzero(dist[dist.argmin():] >= lookahead)[0]
        if math.dist(pose[:2], path[-1]) < lookahead or not len(ahead):
            break
        v, om, pivoting = _steer(pose, path[i + ahead[0]], v_max, pivoting, om_max)
        await rig.step(*inverse(body, v, om, pivot_min=rig.pivot_min, drive_min=rig.drive_min))
    return await goto(rig, body, path[-1], tol, timeout, v_max, om_max)

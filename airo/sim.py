"""Simulated car and overhead camera on a virtual clock, driven by a truth Body."""

import asyncio
import math
from collections import deque

import cv2
import numpy as np

from airo.body import Body, rollout, wrap
from airo.camera import Pose, to_floor
from airo.rig import DT, BaseRig

PX = 300
MARGIN = 40
CAR = np.array([[0.13, 0.08], [0.13, -0.08], [-0.13, -0.08], [-0.13, 0.08]])


class SimRig(BaseRig):
    """Poses carry noise and optional camera latency, stamped with grab time like the camera thread.

    realtime=False advances the clock instantly, for tests.
    """

    def __init__(self, truth: Body, floor=(2.0, 1.2), pose0=None, noise=(0.002, math.radians(0.3)),
                 latency: float = 0.0, realtime: bool = False, seed: int = 0, **kw):
        super().__init__(floor, **kw)
        self.truth = truth
        self.noise = noise
        self.latency = latency
        self.realtime = realtime
        self.t = 0.0
        self._rng = np.random.default_rng(seed)
        start = pose0 or (floor[0] / 2, floor[1] / 2, 0.0)
        self._truth = deque([(0.0, *start)], maxlen=int(latency / DT) + 2)
        self.pose = self._measure()
        rows = int(floor[1] * PX) + 2 * MARGIN
        self.H = np.array([[1 / PX, 0, -MARGIN / PX], [0, -1 / PX, (rows - MARGIN) / PX], [0, 0, 1]])
        self._shape = (rows, int(floor[0] * PX) + 2 * MARGIN, 3)

    @property
    def true_pose(self) -> np.ndarray:
        return np.array(self._truth[-1][1:])

    def _measure(self) -> Pose:
        tr = np.array(self._truth)
        q = max(self.t - self.latency, tr[0, 0])
        x, y, th = (np.interp(q, tr[:, 0], tr[:, i]) for i in (1, 2, 3))
        n = self._rng.normal(0, 1, 3) * (self.noise[0], self.noise[0], self.noise[1])
        return Pose(float(x + n[0]), float(y + n[1]), float(wrap(th + n[2])), self.t)

    async def step(self, pwm_l: int, pwm_r: int) -> Pose:
        now = self.t
        cmd_t, cmd = self.history
        self.t = now + DT
        p = rollout(self.truth, [now, self.t], np.r_[cmd_t, now], np.vstack([cmd, (pwm_l, pwm_r)]),
                    self.true_pose)[-1]
        self._truth.append((self.t, *p))
        await asyncio.sleep(DT if self.realtime else 0)
        return await self._tick(now, pwm_l, pwm_r, True, self._measure())

    async def stop(self):
        self._cmd_t.append(self.t)
        self._cmd.append((0, 0))

    def frame(self) -> np.ndarray:
        img = np.full(self._shape, 235, np.uint8)
        inv = np.linalg.inv(self.H)
        w, h, m = *self.floor, self.inset
        for (x0, y0, x1, y1), colour in [((0, 0, w, h), (90, 90, 90)), ((m, m, w - m, h - m), (80, 80, 220))]:
            box = to_floor(inv, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]))
            cv2.polylines(img, [box.astype(np.int32)], True, colour, 2)
        x, y, th = self.true_pose
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        body = to_floor(inv, CAR @ rot.T + (x, y)).astype(np.int32)
        cv2.fillPoly(img, [body], (60, 140, 60))
        cv2.line(img, tuple(body[0]), tuple(body[1]), (20, 20, 20), 3)
        return img

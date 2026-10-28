"""Simulated car and overhead camera on a virtual clock, driven by a truth Body."""

import asyncio
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

from airo import meta_api
from airo.body import Body, rollout, wrap
from airo.camera import Pose, to_floor
from airo.rig import DT, BaseRig

PX = 300
MARGIN = 40
CAR = np.array([[0.13, 0.08], [0.13, -0.08], [-0.13, -0.08], [-0.13, 0.08]])
PARCEL = "small parcel"


@dataclass
class Thing:
    concept: str
    xy: tuple[float, float]
    size: tuple[float, float]
    colour: tuple[int, int, int]

    def corners(self) -> np.ndarray:
        return np.array(self.xy) + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.array(self.size) / 2


def world() -> list[Thing]:
    """A 2.0 x 1.4 m floor with two stations, two boxes and the parcel, labelled with planner.scan's concepts.

    Everything sits outside the band y 0.45-1.05 m that learning from the default start pose sweeps.
    """
    return [Thing("red paper", (1.6, 1.2), (0.297, 0.21), (60, 60, 210)),
            Thing("blue paper", (1.7, 0.3), (0.297, 0.21), (190, 110, 40)),
            Thing("cardboard box", (1.3, 1.2), (0.14, 0.14), (80, 125, 170)),
            Thing("cardboard box", (0.9, 1.2), (0.16, 0.16), (80, 125, 170)),
            Thing(PARCEL, (0.5, 0.32), (0.1, 0.1), (110, 160, 205))]


class SimRig(BaseRig):
    """Poses carry noise and optional camera latency, stamped with grab time like the camera thread.

    realtime=False advances the clock instantly, for tests.
    """

    def __init__(self, truth: Body, floor=(2.0, 1.2), pose0=None, noise=(0.002, math.radians(0.3)),
                 latency: float = 0.0, realtime: bool = False, seed: int = 0, objects=(), **kw):
        super().__init__(floor, **kw)
        self.truth = truth
        self.objects = list(objects)
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
        self._push(p)
        await asyncio.sleep(DT if self.realtime else 0)
        return await self._tick(now, pwm_l, pwm_r, True, self._measure())

    def _push(self, pose):
        """The bumper face carries the parcel along and its V shape draws it toward the centre line."""
        x, y, th = pose
        c, s = math.cos(th), math.sin(th)
        for o in self.objects:
            if o.concept != PARCEL:
                continue
            fx, fy = c * (o.xy[0] - x) + s * (o.xy[1] - y), -s * (o.xy[0] - x) + c * (o.xy[1] - y)
            front = CAR[0, 0] + o.size[0] / 2
            if 0 < fx < front and abs(fy) < CAR[0, 1] + o.size[1] / 2:
                fy *= 0.9
                o.xy = (x + c * front - s * fy, y + s * front + c * fy)

    async def sam(self, image: np.ndarray, concept: str) -> list[meta_api.Segment]:
        """Stand-in for meta_api.segment on frames from this rig."""
        inv = np.linalg.inv(self.H)
        out = []
        for o in self.objects:
            if o.concept == concept:
                mask = np.zeros(image.shape[:2], np.uint8)
                cv2.fillPoly(mask, [to_floor(inv, o.corners()).astype(np.int32)], 1)
                x, y, w, h = cv2.boundingRect(mask)
                out.append(meta_api.Segment((x, y, x + w, y + h), mask > 0, 0.9))
        return out

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
        for o in self.objects:
            cv2.fillPoly(img, [to_floor(inv, o.corners()).astype(np.int32)], o.colour)
        x, y, th = self.true_pose
        rot = np.array([[math.cos(th), -math.sin(th)], [math.sin(th), math.cos(th)]])
        body = to_floor(inv, CAR @ rot.T + (x, y)).astype(np.int32)
        cv2.fillPoly(img, [body], (60, 140, 60))
        cv2.line(img, tuple(body[0]), tuple(body[1]), (20, 20, 20), 3)
        return img

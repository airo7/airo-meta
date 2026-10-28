"""20 Hz drive loop over the car and the overhead camera, with a safety envelope and a per-tick jsonl log."""

import asyncio
import json
import math
import threading
import time
from pathlib import Path

import numpy as np

from airo.body import Segment
from airo.camera import Pose, PoseTracker, floor_layout, open_camera
from airo.car import Car

HZ = 20
DT = 1 / HZ
INSET = 0.15
LOST_S = 0.3


class RigAbort(RuntimeError):
    pass


class BaseRig:
    """step() sends a command, waits one tick and returns the pose that arrived during it, if any."""

    H: np.ndarray | None = None

    def __init__(self, floor: tuple[float, float], log_dir: str | None = "data/runs", inset: float = INSET):
        self.floor = floor
        self.inset = inset
        self.pose: Pose | None = None
        self._cmd_t: list[float] = []
        self._cmd: list[tuple[int, int]] = []
        self._ticks: list[tuple[float, Pose | None]] = []
        self._log = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self._log = open(Path(log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl", "w")

    @property
    def history(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self._cmd_t), np.array(self._cmd, float).reshape(-1, 2)

    def inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        m = self.inset + margin
        return m <= x <= self.floor[0] - m and m <= y <= self.floor[1] - m

    def segment(self, t0: float = -math.inf) -> Segment:
        """Commands so far and one pose row per tick since t0, NaN for ticks without a fresh pose."""
        rows = [(p.t, p.x, p.y, p.theta) if p else (t, math.nan, math.nan, math.nan)
                for t, p in self._ticks if t >= t0]
        rows = np.array(rows).reshape(-1, 4)
        return Segment(*self.history, rows[:, 0], rows[:, 1:])

    async def _tick(self, t: float, pwm_l: int, pwm_r: int, sent: bool, fresh: Pose | None) -> Pose | None:
        if sent:
            self._cmd_t.append(t)
            self._cmd.append((pwm_l, pwm_r))
        self._ticks.append((t, fresh))
        if fresh:
            self.pose = fresh
        if self._log:
            p = dict(x=fresh.x, y=fresh.y, theta=fresh.theta, pose_t=fresh.t) if fresh else \
                dict.fromkeys(("x", "y", "theta", "pose_t"))
            self._log.write(json.dumps(dict(t=t, pwm_l=pwm_l, pwm_r=pwm_r, sent=sent, **p)) + "\n")
        if self.pose is None or t + DT - self.pose.t > LOST_S:
            await self._abort("pose lost")
        if not self.inside(self.pose.x, self.pose.y):
            await self._abort(f"left the floor inset at ({self.pose.x:.2f}, {self.pose.y:.2f})")
        return fresh

    async def _abort(self, why: str):
        await self.stop()
        raise RigAbort(why)

    async def stop(self):
        raise NotImplementedError

    async def close(self):
        if self._log:
            self._log.close()
            self._log = None


class Rig(BaseRig):
    def __init__(self, car: Car, cap, tracker: PoseTracker, floor, **kw):
        super().__init__(floor, **kw)
        self.car, self._cap, self._tracker = car, cap, tracker
        self._latest: Pose | None = None
        self._frame: np.ndarray | None = None
        self._next = 0.0
        self._running = True
        self._thread = threading.Thread(target=self._camera, daemon=True)
        self._thread.start()

    @classmethod
    async def open(cls, floor, camera: int = 0, marker_h: float = 0.0, cam_h: float = 1.8,
                   address: str | None = None, timeout: float = 5.0, **kw) -> "Rig":
        cap = open_camera(camera)
        rig = cls(await Car.connect(address), cap, PoseTracker(floor_layout(*floor), marker_h, cam_h), floor, **kw)
        end = time.monotonic() + timeout
        while rig._latest is None:
            if time.monotonic() > end:
                await rig.close()
                raise RigAbort("no car pose: check that markers 0-4 are visible")
            await asyncio.sleep(0.05)
        rig.pose = rig._latest
        return rig

    def _camera(self):
        while self._running:
            if not self._cap.grab():
                continue
            t = time.monotonic()
            ok, frame = self._cap.retrieve()
            if not ok:
                continue
            pose = self._tracker.update(frame, t)
            self._frame = frame
            if pose:
                self._latest = pose

    @property
    def H(self):
        return self._tracker.H

    def frame(self) -> np.ndarray | None:
        return self._frame

    async def step(self, pwm_l: int, pwm_r: int) -> Pose | None:
        now = time.monotonic()
        sent = self.car.drive(pwm_l, pwm_r)
        self._next = max(self._next + DT, now + DT / 2)
        await asyncio.sleep(self._next - time.monotonic())
        latest = self._latest
        return await self._tick(now, pwm_l, pwm_r, sent, latest if latest is not self.pose else None)

    async def stop(self):
        await self.car.stop()
        self._cmd_t.append(time.monotonic())
        self._cmd.append((0, 0))

    async def close(self):
        self._running = False
        try:
            await self.car.close()
        finally:
            self._thread.join(timeout=1)
            self._cap.release()
            await super().close()

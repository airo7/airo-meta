"""20 Hz drive loop over the car and the overhead camera, with a safety envelope and a per-tick jsonl log."""

import asyncio
import contextlib
import json
import math
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from airo.body import BODY_PATH, SIM_BODY_PATH, Segment
from airo.camera import Pose, PoseTracker, detect, floor_layout, open_camera
from airo.car import Car, CarLost

HZ = 20
DT = 1 / HZ
INSET = 0.15
LOST_S = 0.3
MJ_BODY_PATH = Path("data/body_mj.json")


class RigAbort(RuntimeError):
    pass


def add_args(p):
    p.add_argument("--sim", action="store_true", help="simulated car and camera")
    p.add_argument("--mujoco", action="store_true", help="MuJoCo twin: contact physics and a rendered camera")
    p.add_argument("--floor", nargs=2, type=float, default=(2.0, 1.4), metavar=("W", "H"),
                   help="corner marker centre distances in metres")
    p.add_argument("--camera", default=os.getenv("AIRO_CAMERA", os.getenv("AIRO_CAMERA_INDEX", "0")),
                   help="camera index, or a stream URL (a phone on Wi-Fi: http://<phone>:8080/video)")
    p.add_argument("--marker-h", type=float, default=0.0, help="car marker height above floor (m)")
    p.add_argument("--marker-offset", nargs=2, type=float, default=(0.0, 0.0), metavar=("FWD", "LEFT"),
                   help="turning centre from the marker centre (m); scripts/centre_check.py measures it")
    p.add_argument("--cam-h", type=float, default=1.8, help="camera height above floor (m)")
    p.add_argument("--address", help="BLE address; default scans for the HM-10 by name")
    p.add_argument("--dry", action="store_true", help="camera and pose only: no BLE, nothing drives")


def body_path(args) -> Path:
    return MJ_BODY_PATH if args.mujoco else SIM_BODY_PATH if args.sim else BODY_PATH


async def open_rig(args, **sim):
    """The hardware rig, or with --sim a SimRig on the KEY body, or with --mujoco an MjRig (keyword args go to
    the simulated rig)."""
    if args.mujoco:
        from airo.mjsim import MjRig

        return MjRig(floor=tuple(args.floor), source="aruco", realtime=True, bumper=True, **sim)
    if args.sim:
        from airo.body import KEY
        from airo.sim import SimRig

        return SimRig(KEY, floor=tuple(args.floor), realtime=True, **sim)
    return await Rig.open(tuple(args.floor), args.camera, args.marker_h, args.cam_h, args.address, args.dry,
                          offset=tuple(args.marker_offset))


class BaseRig:
    """step() sends a command, waits one tick and returns the pose that arrived during it, if any."""

    H: np.ndarray | None = None
    safety = True
    overrun = 0.0

    def __init__(self, floor: tuple[float, float], log_dir: str | None = "data/runs", inset: float = INSET):
        self.floor = floor
        self.inset = inset
        self.pose: Pose | None = None
        self._cmd_t: list[float] = []
        self._cmd: list[tuple[int, int]] = []
        self._ticks: list[tuple[float, Pose | None]] = []
        self._log = None
        self.log_path: Path | None = None
        if log_dir:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            self.log_path = Path(log_dir) / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
            self._log = open(self.log_path, "w")

    @property
    def history(self) -> tuple[np.ndarray, np.ndarray]:
        return np.array(self._cmd_t), np.array(self._cmd, float).reshape(-1, 2)

    def inside(self, x: float, y: float, margin: float = 0.0) -> bool:
        m = self.inset + margin
        return m <= x <= self.floor[0] - m and m <= y <= self.floor[1] - m

    def segment(self, t0: float = -math.inf) -> Segment:
        """Commands so far and one pose row per tick since t0, NaN for ticks without a fresh pose."""
        rows = np.array([(p.t, p.x, p.y, p.theta) if p else (t, math.nan, math.nan, math.nan)
                         for t, p in self._ticks]).reshape(-1, 4)
        rows = rows[rows[:, 0] >= t0]
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
        if not self.safety:
            return fresh
        if self.pose is None or t + DT - self.pose.t > LOST_S:
            await self._abort("pose lost")
        if not self.inside(self.pose.x, self.pose.y, -self.overrun):
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
    """With car=None (dry run) nothing is sent and the safety checks are off."""

    # goals stay in the inset, but a real car overshoots it; the stop waits this much further so it can drive back
    overrun = 0.1

    def __init__(self, car: Car | None, cap, tracker: PoseTracker, floor, **kw):
        super().__init__(floor, **kw)
        self.car, self._cap, self._tracker = car, cap, tracker
        self.safety = car is not None
        self._latest: Pose | None = None
        self._frame: np.ndarray | None = None
        self._next = 0.0
        self._t0 = time.monotonic()
        self._running = True
        self._thread = threading.Thread(target=self._camera, daemon=True)
        self._thread.start()

    @classmethod
    async def open(cls, floor, camera: int | str = 0, marker_h: float = 0.0, cam_h: float = 1.8,
                   address: str | None = None, dry: bool = False, timeout: float = 5.0,
                   offset: tuple[float, float] = (0.0, 0.0), **kw) -> "Rig":
        """Checks the camera, then the markers, then BLE, so each failure names its own fix."""
        cap = open_camera(camera)
        if not cap.isOpened() or not cap.read()[0]:
            cap.release()
            raise RigAbort(f"camera {camera} gives no frames: try --camera 1 (or set AIRO_CAMERA), for a phone "
                           "on Wi-Fi check its stream URL opens in a browser, and allow camera access for this terminal")
        rig = cls(None, cap, PoseTracker(floor_layout(*floor), marker_h, cam_h, offset=offset), floor, **kw)
        try:
            await rig._first_pose(timeout)
            if not dry:
                rig.car = await Car.connect(address)
                rig.safety = True
        except CarLost as e:
            await rig.close()
            raise RigAbort(str(e)) from e
        except BaseException:
            await rig.close()
            raise
        return rig

    async def _first_pose(self, timeout: float):
        end = time.monotonic() + timeout
        while self._latest is None:
            if time.monotonic() > end:
                seen = sorted(detect(cv2.cvtColor(self._frame, cv2.COLOR_BGR2GRAY))) if self._frame is not None else []
                missing = [i for i in (0, 1, 2, 3, 4) if i not in seen]
                raise RigAbort(f"no car pose after {timeout:.0f} s: markers seen {seen or 'none'}, missing {missing} "
                               "(1-4 are the floor corners, 0 is the car)")
            await asyncio.sleep(0.05)
        self.pose = self._latest

    def _camera(self):
        while self._running:
            if not self._cap.grab():
                continue
            t = self.clock()
            ok, frame = self._cap.retrieve()
            if not ok:
                continue
            pose = self._tracker.update(frame, t)
            self._frame = frame
            if pose:
                self._latest = pose

    def clock(self) -> float:
        """Seconds since the rig opened: the dashboard shows rig time as a mission clock."""
        return time.monotonic() - self._t0

    @property
    def H(self):
        return self._tracker.H

    def frame(self) -> np.ndarray | None:
        return self._frame

    async def step(self, pwm_l: int, pwm_r: int) -> Pose | None:
        now = self.clock()
        try:
            sent = self.car is not None and self.car.drive(pwm_l, pwm_r)
        except CarLost as e:
            raise RigAbort(f"{e}; the firmware stops the car within 500 ms") from e
        self._next = max(self._next + DT, now + DT / 2)
        await asyncio.sleep(self._next - self.clock())
        latest = self._latest
        return await self._tick(now, pwm_l, pwm_r, sent, latest if latest is not self.pose else None)

    async def stop(self):
        if self.car is None:
            return
        with contextlib.suppress(CarLost):
            await self.car.stop()
        self._cmd_t.append(self.clock())
        self._cmd.append((0, 0))

    async def close(self):
        self._running = False
        try:
            if self.car:
                with contextlib.suppress(CarLost):
                    await self.car.close()
        finally:
            self._thread.join(timeout=1)
            self._cap.release()
            await super().close()

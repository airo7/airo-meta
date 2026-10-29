"""MuJoCo twin of the rig: the Elegoo car on the marker floor. The truth comes from contact physics and a
DC-motor model per wheel; the pose comes from ground truth or from the real ArUco pipeline on rendered frames."""

import asyncio
import math
import time
from collections import deque
from pathlib import Path

import cv2
import mujoco
import numpy as np

from airo import meta_api
from airo.body import wrap
from airo.camera import CORNER_MARKER_M, DICT, Pose, PoseTracker, floor_layout
from airo.rig import DT, BaseRig
from airo.planner import PARCEL, STATIONS
from airo.sim import world

ASSETS = Path(__file__).parent / "assets"
DELAY = 0.17
WEIGHT_KG = 0.8
IMBALANCE = 0.009
CAM_H = 1.8
MARKER_H = 0.1
SIDES = {"l": (0, 1), "r": (2, 3)}
KINETIC = 0.05
BREAKAWAY = 0.15
LOAD_DAMPING = 2.6e-4
GEOM = mujoco.mjtGeom
GHOST = (0.38, 0.84, 1.0)


def _marker(i: int, cell: int = 40) -> np.ndarray:
    """Marker i with a one-cell white quiet zone: a geom of side 8/6 m carries a marker of side m."""
    img = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(DICT, i, 6 * cell), *[cell] * 4, cv2.BORDER_CONSTANT,
                             value=255)
    return np.repeat(img[..., None], 3, axis=2)


def _yaw(q) -> float:
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def _quat(theta: float) -> list[float]:
    return [math.cos(theta / 2), 0, 0, math.sin(theta / 2)]


def _rgba(bgr, a=1.0) -> list[float]:
    return [bgr[2] / 255, bgr[1] / 255, bgr[0] / 255, a]


def _rounded(spec, name: str, half) -> str:
    spec.add_mesh(name=name, scale=list(half)).make_supersphere(16, 0.12, 0.08)
    return name


def scene(floor, objects) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(ASSETS / "world.xml"))
    for i in range(5):
        img = _marker(i)
        spec.add_texture(name=f"aruco{i}", type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1],
                         height=img.shape[0], nchannel=3, data=img.ravel())
    wb = spec.worldbody
    side = CORNER_MARKER_M * 8 / 6
    for i, (x, y) in floor_layout(*floor).items():
        spec.add_material(name=f"aruco{i}", textures=["", f"aruco{i}"])
        wb.add_geom(type=GEOM.mjGEOM_BOX, size=[0.105, 0.1485, 0.0002], pos=[x, y, 0.0002], material="paper")
        wb.add_geom(type=GEOM.mjGEOM_PLANE, size=[side / 2, side / 2, 1], pos=[x, y, 0.0005],
                    material=f"aruco{i}", contype=0, conaffinity=0)
    for i, o in enumerate(objects):
        sx, sy = np.array(o.size) / 2
        if o.concept in STATIONS.values():
            spec.add_material(name=f"sheet{i}", rgba=_rgba(o.colour), specular=0.15, shininess=0.2)
            wb.add_geom(name=f"thing{i}", type=GEOM.mjGEOM_BOX, size=[sx, sy, 0.0002], pos=[*o.xy, 0.0002],
                        material=f"sheet{i}")
            continue
        h = float(np.clip(0.6 * max(o.size), 0.06, 0.15)) / 2
        b = wb.add_body(name=f"thing{i}", pos=[*o.xy, h])
        b.add_freejoint()
        if o.concept == PARCEL:
            spec.add_material(name=f"box{i}", rgba=_rgba(o.colour), specular=0.1, shininess=0.15)
        b.add_geom(type=GEOM.mjGEOM_MESH, meshname=_rounded(spec, f"thing{i}", (sx, sy, h)),
                   material=f"box{i}" if o.concept == PARCEL else "cardboard",
                   mass=0.12 if o.concept == PARCEL else 0.35, friction=[0.45, 0.005, 0.0001])
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx + 0.0004, 0.024, h + 0.0004], material="tape", contype=0,
                   conaffinity=0, mass=0)
        if o.concept == PARCEL:
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx * 0.55, sy * 0.4, 0.0003], pos=[-sx * 0.1, sy * 0.4, h],
                       material="paper", contype=0, conaffinity=0, mass=0)
    cam = spec.camera("overhead")
    cam.pos = [floor[0] / 2, floor[1] / 2, CAM_H]
    cam.fovy = math.degrees(2 * math.atan(max(floor[1] / 2 + 0.2, (floor[0] / 2 + 0.2) * 9 / 16) / CAM_H))
    return spec


class MjRig(BaseRig):
    """Same interface as sim.SimRig. source="aruco" renders the overhead camera and runs camera.PoseTracker on it;
    source="truth" reads the physics state plus noise. realtime=False advances instantly, for tests."""

    def __init__(self, floor=(2.0, 1.2), pose0=None, objects=None, source: str = "truth",
                 noise=(0.002, math.radians(0.3)), delay: float = DELAY, realtime: bool = False, seed: int = 0,
                 size=(1920, 1080), bumper: bool = False, weight: float = 0.0, tow=None, patch=None,
                 flip: str | None = None, **kw):
        super().__init__(floor, **kw)
        self.objects = list(world() if objects is None else objects)
        self.noise, self.delay, self.realtime, self.size = noise, delay, realtime, size
        self.t = 0.0
        self.ghosts: list = []
        self._rng = np.random.default_rng(seed)
        self._queue: deque = deque()
        self._sign = {"l": 1, "r": 1}
        self._renderers: dict = {}
        self._frame: tuple[float, np.ndarray] | None = None
        self._chase: mujoco.MjvCamera | None = None
        self._wall0 = None
        self.paused, self.speed = False, 1.0
        self.changes: list[str] = []
        self.spec = scene(floor, self.objects)
        self.m = self.d = None
        x, y, th = pose0 or (floor[0] / 2, floor[1] / 2, 0.0)
        car = self.spec.body("car")
        car.pos, car.quat = [x, y, 0.0005], _quat(th)
        if bumper:
            self.add_bumper()
        if weight:
            self.add_weight(weight)
        if tow:
            self.attach_tow(*(() if tow is True else (tow,)))
        if patch:
            self.slippery_patch(patch)
        if flip:
            self.flip_motor(flip)
        self._compile()
        self.tracker = PoseTracker(floor_layout(*floor), MARKER_H, CAM_H) if source == "aruco" else None
        self.pose = self._measure()

    def _compile(self):
        if self.m is None:
            self.m = self.spec.compile()
            self.d = mujoco.MjData(self.m)
            mujoco.mj_forward(self.m, self.d)
        else:
            self.m, self.d = self.spec.recompile(self.m, self.d)
            mujoco.mj_forward(self.m, self.d)
            self._renderers.clear()
        self._dofs = [self.m.joint(j).dofadr[0] for j in ("fl", "rl", "fr", "rr")]
        self._stiction = self.m.dof_frictionloss[self._dofs].copy()
        gains = self.m.actuator_gainprm[:, 0]
        gains[:] = np.abs(gains) * np.repeat([1 + IMBALANCE, 1 - IMBALANCE], 2)
        for side, sign in self._sign.items():
            gains[list(SIDES[side])] *= sign

    # body changes, as scene options or between steps

    def add_weight(self, kg: float = WEIGHT_KG):
        """A steel block strapped between the decks in front of the battery; more stack on the rear deck."""
        n = sum(b.name.startswith("weight") for b in self.spec.bodies)
        w = kg / 7850 / 0.045 / 0.022
        pos = [0.036, 0, 0.0581] if n == 0 else [-0.088, 0, 0.0861 + (n - 1) * 0.0225]
        self.changes.append("weight")
        b = self.spec.body("car").add_body(name=f"weight{n}", pos=pos)
        b.add_geom(type=GEOM.mjGEOM_MESH, meshname=_rounded(self.spec, f"weight{n}", (0.0225, w / 2, 0.011)),
                   material="iron", contype=0, conaffinity=0, mass=kg, group=1)
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.007, w / 2 + 0.0006, 0.0112], material="velcro", contype=0,
                   conaffinity=0, mass=0, group=1)
        self._rebuild()

    def add_bumper(self):
        """The V-shaped cardboard pusher on the nose; it cups the parcel toward the centre line."""
        b = self.spec.body("car").add_body(name="bumper", pos=[0.142, 0, 0.042])
        for side in (1, -1):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.045, 0.0015, 0.025], pos=[0, side * 0.04, 0],
                       euler=[0, 0, side * 63.4], material="cardboard", mass=0.015, group=1)
        self._rebuild()

    def attach_tow(self, box=(0.12, 0.1, 0.08), mass: float = 0.15, length: float = 0.25):
        """A cardboard box on a string from the rear hitch, dragging on the floor."""
        if "tow" in self.changes:
            return
        self.changes.append("tow")
        x, y, th = self.true_pose if self.m is not None else (*self.spec.body("car").pos[:2],
                                                                _yaw(self.spec.body("car").quat))
        back = 0.118 + length * 0.95 + box[0] / 2
        b = self.spec.worldbody.add_body(name="tow", pos=[x - back * math.cos(th), y - back * math.sin(th),
                                                          box[2] / 2], quat=_quat(th))
        b.add_freejoint()
        b.add_geom(type=GEOM.mjGEOM_MESH, meshname=_rounded(self.spec, "tow", np.array(box) / 2),
                   material="cardboard", mass=mass, friction=[0.5, 0.005, 0.0001])
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[box[0] / 2 + 0.0004, 0.02, box[2] / 2 + 0.0004], material="tape",
                   contype=0, conaffinity=0, mass=0)
        b.add_site(name="tow", pos=[box[0] / 2, 0, box[2] * 0.1])
        t = self.spec.add_tendon(name="string", limited=True, range=[0, length], width=0.0012,
                                 material="string")
        t.wrap_site("hitch")
        t.wrap_site("tow")
        self._rebuild()

    def slippery_patch(self, rect=(1.2, 0.35, 1.6, 0.85), friction: float = 0.12):
        """A glossy low-friction floor zone (x0, y0, x1, y1) that overrides the tyre friction."""
        if "slippery" in self.changes:
            return
        self.changes.append("slippery")
        x0, y0, x1, y1 = rect
        size, pos = [(x1 - x0) / 2, (y1 - y0) / 2, 4e-4], [(x0 + x1) / 2, (y0 + y1) / 2, 4e-4]
        wb = self.spec.worldbody
        wb.add_geom(name="patch", type=GEOM.mjGEOM_BOX, size=size, pos=pos, priority=1, condim=6,
                    friction=[friction, 0.001, 0.0001], group=3, rgba=[0, 0, 0, 0])
        wb.add_geom(type=GEOM.mjGEOM_PLANE, size=size, pos=[*pos[:2], 8.5e-4], material="glossy", contype=0,
                    conaffinity=0)
        self._rebuild()

    def flip_motor(self, side: str):
        """A hidden fault: that side's motors run backwards."""
        self._sign[side] *= -1
        self.changes.append("flip")
        self._rebuild()

    def _rebuild(self):
        if self.m is not None:
            self._compile()

    # rig interface

    @property
    def true_pose(self) -> np.ndarray:
        q = self.d.qpos
        return np.array([q[0], q[1], _yaw(q[3:7])])

    @property
    def H(self) -> np.ndarray:
        if self.tracker is not None and self.tracker.H is not None:
            return self.tracker.H
        w, h = self.size
        s = 2 * CAM_H * math.tan(math.radians(self.m.cam_fovy[0]) / 2) / h
        cx, cy = self.floor[0] / 2, self.floor[1] / 2
        return np.array([[s, 0, cx - s * (w - 1) / 2], [0, -s, cy + s * (h - 1) / 2], [0, 0, 1]])

    def _measure(self) -> Pose | None:
        if self.tracker is not None:
            return self.tracker.update(self.frame(), self.t)
        x, y, th = self.true_pose
        n = self._rng.normal(0, 1, 3) * (self.noise[0], self.noise[0], self.noise[1])
        return Pose(float(x + n[0]), float(y + n[1]), float(wrap(th + n[2])), self.t)

    def _physics(self, until: float):
        while self.d.time < until - 1e-9:
            while self._queue and self._queue[0][0] <= self.d.time + 1e-9:
                _, l, r = self._queue.popleft()
                self.d.ctrl[:] = (l, l, r, r)
            rolling = np.abs(self.d.qvel[self._dofs]) > BREAKAWAY
            self.m.dof_frictionloss[self._dofs] = self._stiction * np.where(rolling, KINETIC, 1)
            self.m.dof_damping[self._dofs] = LOAD_DAMPING * self.d.sensordata[:4]
            mujoco.mj_step(self.m, self.d)
        for i, o in enumerate(self.objects):
            if o.concept not in STATIONS.values():
                o.xy = tuple(self.d.body(f"thing{i}").xpos[:2])

    async def step(self, pwm_l: int, pwm_r: int) -> Pose | None:
        now = self.t
        self._queue.append((now + self.delay, pwm_l, pwm_r))
        self.t = now + DT
        self._physics(self.t)
        if self.realtime:
            while self.paused:
                await asyncio.sleep(0.05)
                self._wall0 = None
            wall = time.monotonic()
            if self._wall0 is None or wall - (self._wall0 + now / self.speed) > 0.25:
                self._wall0 = wall - now / self.speed
            await asyncio.sleep(max(0.0, self._wall0 + self.t / self.speed - wall))
        else:
            await asyncio.sleep(0)
        return await self._tick(now, pwm_l, pwm_r, True, self._measure())

    async def stop(self):
        self._queue.append((self.t + self.delay, 0, 0))
        self._cmd_t.append(self.t)
        self._cmd.append((0, 0))

    async def close(self):
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()
        await super().close()

    async def sam(self, image: np.ndarray, concept: str) -> list[meta_api.Segment]:
        """Stand-in for meta_api.segment: exact masks from a segmentation render of the overhead camera."""
        r = self._renderer(image.shape[1], image.shape[0])
        r.enable_segmentation_rendering()
        r.update_scene(self.d, "overhead")
        seg = r.render()
        r.disable_segmentation_rendering()
        out = []
        for i, o in enumerate(self.objects):
            if o.concept != concept:
                continue
            ids = self._geoms(f"thing{i}")
            mask = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], ids)
            if mask.any():
                ys, xs = np.nonzero(mask)
                out.append(meta_api.Segment((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                                            mask, 0.9))
        return out

    def _geoms(self, name: str) -> list[int]:
        try:
            return [self.m.geom(name).id]
        except KeyError:
            b = self.m.body(name).id
            return list(np.flatnonzero(self.m.geom_bodyid == b))

    # rendering

    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        if (w, h) not in self._renderers:
            self._renderers[w, h] = mujoco.Renderer(self.m, h, w, max_geom=4000)
        return self._renderers[w, h]

    def frame(self) -> np.ndarray:
        """The overhead camera: what the phone sees, without ghosts."""
        if self._frame is None or self._frame[0] != self.t:
            r = self._renderer(*self.size)
            r.update_scene(self.d, "overhead")
            self._frame = (self.t, cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        return self._frame[1]

    def render_ghosts(self, poses, alpha: float = 0.25):
        """Ghost cars drawn in view(): poses (k, 3) are ghosts, (k, n, 3) are k predicted paths ending in ghosts."""
        self.ghosts = [(np.asarray(poses, float), alpha)] if len(poses) else []

    def set_speed(self, speed: float):
        self.speed, self._wall0 = speed, None

    def view(self, width: int = 1280, height: int = 720, camera: str | mujoco.MjvCamera = "chase") -> np.ndarray:
        r = self._renderer(width, height)
        r.update_scene(self.d, self._follow() if camera == "chase" else camera)
        for poses, alpha in self.ghosts:
            _draw_ghosts(r.scene, poses, alpha)
        return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)

    def select(self, u: float, v: float, width: int = 1280, height: int = 720):
        """(body, geom, world point) under image fraction (u, v) of the last view() at that size, or None."""
        r = self._renderer(width, height)
        point, geom, flex, skin = np.zeros(3), *(np.array([-1], np.int32) for _ in range(3))
        body = mujoco.mjv_select(self.m, self.d, mujoco.MjvOption(), width / height, u, 1 - v, r.scene, point, geom,
                                 flex, skin)
        return (int(body), int(geom[0]), point) if body >= 0 and geom[0] >= 0 else None

    def _follow(self) -> mujoco.MjvCamera:
        """A trailing 3/4 camera that eases toward the car on the sim clock."""
        x, y, th = self.true_pose
        target = np.array([x + 0.25 * math.cos(th), y + 0.25 * math.sin(th), 0.03])
        az = math.degrees(th) + 30
        c = self._chase
        if c is None:
            c = self._chase = mujoco.MjvCamera()
            c.lookat[:], c.azimuth, c.distance, c.elevation = target, az, 1.15, -27
            self._chase_t = self.t
        k = 1 - math.exp(-(self.t - self._chase_t) / 0.6)
        self._chase_t = self.t
        c.lookat[:] += k * (target - c.lookat)
        c.azimuth += k * ((az - c.azimuth + 180) % 360 - 180)
        return c


_PARTS = [  # ghost silhouette: kind, half size, offset in the car frame
    (GEOM.mjGEOM_BOX, (0.115, 0.056, 0.0016), (0, 0, 0.0455)),
    (GEOM.mjGEOM_BOX, (0.115, 0.056, 0.0016), (0, 0, 0.0735)),
    (GEOM.mjGEOM_BOX, (0.006, 0.0225, 0.0105), (0.108, 0, 0.109)),
] + [(GEOM.mjGEOM_CYLINDER, (0.0325, 0.013, 0), (sx * 0.065, sy * 0.0725, 0.0325)) for sx in (1, -1) for sy in (1, -1)]
_WHEEL = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]], float)
_EYE = np.eye(3)


def _draw_ghosts(scn, poses: np.ndarray, alpha: float):
    paths = poses if poses.ndim == 3 else poses[:, None]
    end, n = paths[:, -1], paths.shape[1]
    # ghosts fade as they converge on the car, so a parked car is not tinted by its own cloud
    fade = np.clip(np.linalg.norm(end[:, :2] - paths[:, 0, :2], axis=1) / 0.08, 0, 1) if n > 1 else np.ones(len(end))
    for p in paths if n > 1 else ():
        for j in range(n - 1):
            if g := _geom(scn, GEOM.mjGEOM_CAPSULE, min(1.0, 2 * alpha) * (0.25 + 0.75 * j / n), 0.9):
                mujoco.mjv_connector(g, GEOM.mjGEOM_CAPSULE, 0.0014, np.r_[p[j, :2], 0.002], np.r_[p[j + 1, :2], 0.002])
    if n > 1 and len(end) > 2:
        w, v = np.linalg.eigh(np.cov(end[:, :2].T) + np.eye(2) * 1e-6)
        rot = np.array([[v[0, 1], v[0, 0], 0], [v[1, 1], v[1, 0], 0], [0, 0, 1]])
        _geom(scn, GEOM.mjGEOM_ELLIPSOID, 0.5 * alpha * fade.mean(), 0.45,
              (2 * w[1] ** 0.5 + 0.12, 2 * w[0] ** 0.5 + 0.08, 6e-4), (*end[:, :2].mean(axis=0), 0.0012), rot)
    for pose, f in zip(end, fade):
        _ghost(scn, pose, _PARTS[:2], alpha * f)
    if len(end) > 1:
        mean = (*end[:, :2].mean(axis=0), math.atan2(np.sin(end[:, 2]).mean(), np.cos(end[:, 2]).mean()))
        _ghost(scn, mean, _PARTS, min(0.5, 2.5 * alpha) * fade.mean())


def _ghost(scn, pose, parts, alpha: float):
    x, y, th = pose
    c, s = math.cos(th), math.sin(th)
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    for kind, size, off in parts:
        mat = rot @ _WHEEL if kind == GEOM.mjGEOM_CYLINDER else rot
        _geom(scn, kind, alpha, 0.45, size, rot @ off + (x, y, 0), mat)


def _geom(scn, kind, alpha: float, emission: float, size=(0, 0, 0), pos=(0, 0, 0), mat=_EYE):
    if scn.ngeom >= scn.maxgeom:
        return None
    g = scn.geoms[scn.ngeom]
    scn.ngeom += 1
    mujoco.mjv_initGeom(g, kind, np.asarray(size, float), np.asarray(pos, float), np.asarray(mat, float).ravel(),
                        np.array([*GHOST, alpha], np.float32))
    g.emission, g.specular, g.shininess, g.category = emission, 0.6, 0.6, mujoco.mjtCatBit.mjCAT_DECOR
    return g

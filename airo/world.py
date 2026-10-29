"""One MuJoCo world for a site and every car on it: fixed-rate physics with the wheel motor model, place changes,
and the cameras (overhead frame, SAM stand-in, hero view with ghost cars)."""

import asyncio
import math
import time

import cv2
import mujoco
import numpy as np

from airo import meta_api, scene
from airo.rig import DT
from airo.site import HOME, Site

KINETIC = 0.05
BREAKAWAY = 0.15
LOAD_DAMPING = 2.6e-4
WATCHDOG = 0.5
LOD_M = 1.6
GEOM = mujoco.mjtGeom
OBJ_GEOM = int(mujoco.mjtObj.mjOBJ_GEOM)  # comparing arrays to the pybind enum goes element by element in Python
GHOST = (0.38, 0.84, 1.0)


class World:
    """One physics world for a site and every car on it, stepped at the fixed rate. With realtime a ticker task paces
    it by the wall clock (pause and speed live here) and each car's step waits for the next tick; otherwise the cars'
    own steps advance it, in lockstep when they are stepped together. Recompiles after a change keep every car's
    state."""

    def __init__(self, site: Site = HOME, objects=None, realtime: bool = False, size=None):
        self.site, self.floor = site, site.floor
        self.objects = list(site.objects() if objects is None else objects)
        self.realtime, self.size = realtime, tuple(size or site.frame)
        self.cars: list = []
        self.changes: list[str] = []
        self.zone: tuple | None = None
        self.zone_t: float | None = None
        self.paused, self.speed = False, 1.0
        self.spec = scene.build(site, self.objects)
        self.m = self.d = None
        self._renderers: dict = {}
        self._near, self._far = mujoco.MjvOption(), mujoco.MjvOption()
        self._far.geomgroup[2] = 0
        self._frame = self._seg = None
        self._ticker: asyncio.Task | None = None
        self._waiters: list[tuple[float, asyncio.Future]] = []
        self._wall0 = None

    @property
    def t(self) -> float:
        return 0.0 if self.d is None else self.d.time

    def compile(self):
        if self.m is None:
            self.m = self.spec.compile()
            self.d = mujoco.MjData(self.m)
        else:
            self.m, self.d = self.spec.recompile(self.m, self.d)
            for r in self._renderers.values():
                r.close()
            self._renderers.clear()
        mujoco.mj_forward(self.m, self.d)
        self._frame = self._seg = None
        for car in self.cars:
            car._bind()
        self._dofs = np.concatenate([c._dofs for c in self.cars])
        self._touch = np.concatenate([c._touch for c in self.cars])
        self._stiction = self.m.dof_frictionloss[self._dofs].copy()
        self._free = [(o, b.id) for i, o in enumerate(self.objects)
                      if (b := _find(self.m.body, f"thing{i}")) is not None and self.m.body_dofnum[b.id]]
        for car in self.cars:
            if car.pose is None:
                car.pose = car._measure()

    def rebuild(self):
        if self.m is not None:
            self.compile()

    # place changes

    def spill(self, rect=None, friction: float = 0.12):
        """A slippery zone (x0, y0, x1, y1) on the floor, default the site's: it changes the place, not a car."""
        if "slippery" in self.changes:
            return
        self.changes.append("slippery")
        self.zone, self.zone_t = tuple(rect or self.site.spill), self.t
        self._clean = scene.add_spill(self.spec, self.zone, friction,
                                      self.floor if self.site.scene == "factory" else None)
        self.rebuild()

    def clean(self):
        if "slippery" not in self.changes:
            return
        self.changes.remove("slippery")
        self.zone = None
        scene.remove_spill(self.spec, self._clean)
        self.rebuild()

    def restock(self):
        """Every tote back where the site put it, as if new work arrived at the pick zone."""
        m, d = self.m, self.d
        for i, (o, home) in enumerate(zip(self.objects, self.site.objects())):
            if o.kind != "tote" or (b := _find(m.body, f"thing{i}")) is None or not m.body_dofnum[b.id]:
                continue
            q, v = m.jnt_qposadr[m.body_jntadr[b.id]], m.jnt_dofadr[m.body_jntadr[b.id]]
            d.qpos[q:q + 7] = (*home.xy, d.qpos[q + 2], 1, 0, 0, 0)
            d.qvel[v:v + 6] = 0
            o.xy = home.xy
        mujoco.mj_forward(m, d)
        self._frame = self._seg = None

    # time

    def _physics(self, until: float):
        m, d = self.m, self.d
        for c in self.cars:
            if not c._queue and d.time - c._sent > WATCHDOG:
                d.ctrl[c._act] = 0
        while d.time < until - 1e-9:
            for c in self.cars:
                while c._queue and c._queue[0][0] <= d.time + 1e-9:
                    _, l, r = c._queue.popleft()
                    d.ctrl[c._act] = (l, l, r, r)
            rolling = np.abs(d.qvel[self._dofs]) > BREAKAWAY
            m.dof_frictionloss[self._dofs] = self._stiction * np.where(rolling, KINETIC, 1)
            m.dof_damping[self._dofs] = LOAD_DAMPING * d.sensordata[self._touch]
            mujoco.mj_step(m, d)
        for o, b in self._free:
            o.xy = tuple(d.xpos[b, :2])

    async def until(self, target: float):
        """Return once the world has reached time target."""
        if not self.realtime:
            if self.t < target - 1e-9:
                self._physics(target)
            return
        loop = asyncio.get_running_loop()
        if self._ticker is None or self._ticker.done() or self._ticker.get_loop() is not loop:
            self._ticker = loop.create_task(self._tick())
        if self.t < target - 1e-9:
            fut = loop.create_future()
            self._waiters.append((target, fut))
            await fut

    async def _tick(self):
        while True:
            while self.paused:
                await asyncio.sleep(0.05)
                self._wall0 = None
            wall = time.monotonic()
            if self._wall0 is None or wall - (self._wall0 + self.t / self.speed) > 0.25:
                self._wall0 = wall - self.t / self.speed
            await asyncio.sleep(max(0.0, self._wall0 + (self.t + DT) / self.speed - wall))
            if self.paused:
                continue
            self._physics(self.t + DT)
            waiting = []
            for target, fut in self._waiters:
                if fut.done():
                    continue
                if target <= self.t + 1e-9:
                    fut.set_result(None)
                else:
                    waiting.append((target, fut))
            self._waiters = waiting

    def set_speed(self, speed: float):
        self.speed, self._wall0 = speed, None

    def close(self):
        if self._ticker:
            self._ticker.cancel()
        for r in self._renderers.values():
            r.close()
        self._renderers.clear()

    # cameras

    @property
    def H(self) -> np.ndarray:
        """Overhead image pixels to floor metres, from the camera's pinhole model."""
        w, h = self.size
        fovy = self.m.cam_fovy[self.m.camera("overhead").id]
        s = 2 * self.site.cam_h * math.tan(math.radians(fovy) / 2) / h
        cx, cy = self.floor[0] / 2, self.floor[1] / 2
        return np.array([[s, 0, cx - s * (w - 1) / 2], [0, -s, cy + s * (h - 1) / 2], [0, 0, 1]])

    def _renderer(self, w: int, h: int) -> mujoco.Renderer:
        if (w, h) not in self._renderers:
            self._renderers[w, h] = mujoco.Renderer(self.m, h, w, max_geom=8000)
        return self._renderers[w, h]

    def frame(self) -> np.ndarray:
        """The overhead camera: what the phone sees, without ghosts."""
        if self._frame is None or self._frame[0] != self.t:
            r = self._renderer(*self.size)
            r.update_scene(self.d, "overhead", self._far)
            self._frame = (self.t, cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR))
        return self._frame[1]

    def view(self, width: int, height: int, camera) -> np.ndarray:
        """Tyre lugs (geom group 2) are left out beyond LOD_M: hundreds of draw calls for sub-pixel detail."""
        r = self._renderer(width, height)
        far = isinstance(camera, str) or camera.distance > LOD_M
        r.update_scene(self.d, camera, self._far if far else self._near)
        for car in self.cars:
            for poses, alpha in car.ghosts:
                _draw_ghosts(r.scene, poses, alpha)
        return cv2.cvtColor(r.render(), cv2.COLOR_RGB2BGR)

    def select(self, u: float, v: float, width: int = 1280, height: int = 720):
        """(body, geom, world point) under image fraction (u, v) of the last view() at that size, or None."""
        r = self._renderer(width, height)
        point, geom, flex, skin = np.zeros(3), *(np.array([-1], np.int32) for _ in range(3))
        body = mujoco.mjv_select(self.m, self.d, mujoco.MjvOption(), width / height, u, 1 - v, r.scene, point, geom,
                                 flex, skin)
        return (int(body), int(geom[0]), point) if body >= 0 and geom[0] >= 0 else None

    def car_of(self, body: int):
        """The fleet car a body belongs to, by name prefix; the tow box hangs off the world but carries its prefix."""
        name = self.m.body(body).name
        return next((c for c in self.cars if c.prefix and name.startswith(c.prefix)), None)

    async def sam(self, image: np.ndarray, concept: str) -> list[meta_api.Segment]:
        """Stand-in for meta_api.segment: exact masks from a segmentation render of the overhead camera."""
        h, w = image.shape[:2]
        if self._seg is None or self._seg[0] != (self.t, w, h):
            r = self._renderer(w, h)
            r.enable_segmentation_rendering()
            r.update_scene(self.d, "overhead")
            seg = r.render()
            r.disable_segmentation_rendering()
            self._seg = ((self.t, w, h), np.where(seg[..., 1] == OBJ_GEOM, seg[..., 0], -1))
        geoms = self._seg[1]
        out = []
        for i, o in enumerate(self.objects):
            # like SAM, "yellow box" also finds a "yellow box with a blue tag"
            if o.concept != concept and not o.concept.startswith(concept + " "):
                continue
            mask = np.isin(geoms, self._geoms(f"thing{i}"))
            if mask.any():
                ys, xs = np.nonzero(mask)
                out.append(meta_api.Segment((int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1),
                                            mask, 0.9))
        return out

    def _geoms(self, name: str) -> list[int]:
        if (g := _find(self.m.geom, name)) is not None:
            return [g.id]
        return list(np.flatnonzero(self.m.geom_bodyid == self.m.body(name).id))


def _find(get, name: str):
    try:
        return get(name)
    except KeyError:
        return None


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
    for p, f in zip(paths, fade) if n > 1 else ():
        if f < 0.02:
            continue
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

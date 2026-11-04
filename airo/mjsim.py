"""MuJoCo twin of the rig: Elegoo cars on a site floor, all in one physics world (airo.world). The truth comes from
contact physics and a DC-motor model per wheel; the pose comes from ground truth or from the real ArUco pipeline on
rendered frames."""

import asyncio
import math
from collections import deque
from dataclasses import replace

import mujoco
import numpy as np

from airo import meta_api, scene
from airo.body import wrap
from airo.camera import Pose, PoseTracker, floor_layout
from airo.rig import DT, BaseRig
from airo.site import HOME, Site, robot
from airo.world import GEOM, WATCHDOG, World

DELAY = 0.17
WEIGHT_KG = 0.8
IMBALANCE = 0.009
MARKER_H = 0.1
WHEELS = ("fl", "rl", "fr", "rr")
SIDES = {"l": (0, 1), "r": (2, 3)}


def _yaw(q) -> float:
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def hex_rgba(colour: str) -> list[float]:
    return [int(colour[i:i + 2], 16) / 255 for i in (1, 3, 5)] + [1.0]


def fleet(site: Site, n: int, realtime: bool = False, seed: int = 0, log_dir: str | None = None,
          **kw) -> list["MjRig"]:
    """n cars on the site in one world, each with its accent colour, its own marker id and log directory."""
    world = World(site, realtime=realtime)
    rigs = [MjRig(world=world, prefix=f"airo{k + 1}/", pose0=site.starts[k], seed=seed + k, name=robot(k)["name"],
                  marker=5 + k, accent=hex_rgba(robot(k)["color"]),
                  log_dir=log_dir and f"{log_dir}/{robot(k)['id']}", **kw) for k in range(n)]
    world.compile()
    return rigs


class MjRig(BaseRig):
    """One car in a World, with the same interface as sim.SimRig. On its own it builds a single-car world on the home
    site (floor and objects overridable). source="aruco" renders the overhead camera and runs camera.PoseTracker on
    it; source="truth" reads the physics state plus noise. realtime=False advances instantly, for tests. marker_at
    (forward, left) metres tapes the car's marker off its turning centre."""

    def __init__(self, floor=(2.0, 1.2), pose0=None, objects=None, source: str = "truth",
                 noise=(0.002, math.radians(0.3)), delay: float = DELAY, realtime: bool = False, seed: int = 0,
                 size=(1920, 1080), bumper: bool = False, weight: float = 0.0, tow=None, patch=None,
                 flip: str | None = None, world: World | None = None, prefix: str = "", name: str = "AIRO",
                 marker: int = 0, accent=None, marker_at=(0.0, 0.0), **kw):
        self._own = world is None
        if world is None:
            world = World(replace(HOME, floor=tuple(floor), pose=source, frame=tuple(size)), objects, realtime)
        super().__init__(world.floor, **kw)
        self.world, self.prefix, self.name = world, prefix, name
        self.noise, self.delay = noise, delay
        self.t = world.t
        self.ghosts: list = []
        self._changes: list[str] = []
        self._rng = np.random.default_rng(seed)
        self._queue: deque = deque()
        self._sent = world.t
        self._sign = {"l": 1, "r": 1}
        self._chase: mujoco.MjvCamera | None = None
        self.tracker = PoseTracker(floor_layout(*self.floor), MARKER_H, world.site.cam_h) if source == "aruco" else None
        scene.add_car(world.spec, prefix, pose0 or (self.floor[0] / 2, self.floor[1] / 2, 0.0), marker, accent)
        world.spec.body(prefix + "marker_card").pos = [*marker_at, 0]
        world.cars.append(self)
        if world.site.mode == "carry":
            self.add_tray()
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
        if self._own:
            world.compile()

    def _bind(self):
        m, p = self.world.m, self.prefix
        self.root = m.body(p + "car").id
        self._qadr = m.jnt_qposadr[m.joint(p + "car").id]
        self._dofs = np.array([m.joint(p + w).dofadr[0] for w in WHEELS])
        self._act = np.array([m.actuator(p + w).id for w in WHEELS])
        self._touch = np.array([m.sensor(p + w).adr[0] for w in WHEELS])
        gains = np.abs(m.actuator_gainprm[self._act, 0]) * np.repeat([1 + IMBALANCE, 1 - IMBALANCE], 2)
        for side, sign in self._sign.items():
            gains[list(SIDES[side])] *= sign
        m.actuator_gainprm[self._act, 0] = gains

    @property
    def m(self) -> mujoco.MjModel:
        return self.world.m

    @property
    def d(self) -> mujoco.MjData:
        return self.world.d

    @property
    def spec(self) -> mujoco.MjSpec:
        return self.world.spec

    @property
    def objects(self) -> list:
        return self.world.objects

    @property
    def realtime(self) -> bool:
        return self.world.realtime

    @property
    def paused(self) -> bool:
        return self.world.paused

    @paused.setter
    def paused(self, value: bool):
        self.world.paused = value

    @property
    def speed(self) -> float:
        return self.world.speed

    def set_speed(self, speed: float):
        self.world.set_speed(speed)

    @property
    def changes(self) -> list[str]:
        """This car's body changes, then the place changes of its world."""
        return self._changes + self.world.changes

    # body changes, as constructor options or between steps

    def add_weight(self, kg: float = WEIGHT_KG):
        """A steel block strapped between the decks in front of the battery; more stack on the rear deck."""
        n = len(self._weights())
        w = kg / 7850 / 0.045 / 0.022
        pos = [0.036, 0, 0.0581] if n == 0 else [-0.088, 0, 0.0861 + (n - 1) * 0.0225]
        self._changes.append("weight")
        name = f"{self.prefix}weight{n}"
        b = self.spec.body(self.prefix + "car").add_body(name=name, pos=pos)
        b.add_geom(type=GEOM.mjGEOM_MESH, meshname=scene.rounded(self.spec, name, (0.0225, w / 2, 0.011)),
                   material="iron", contype=0, conaffinity=0, mass=kg, group=1)
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.007, w / 2 + 0.0006, 0.0112], material="velcro", contype=0,
                   conaffinity=0, mass=0, group=1)
        self._rebuild()

    def remove_weight(self) -> bool:
        """Unstrap the last weight added."""
        if not (names := self._weights()):
            return False
        self.spec.delete(self.spec.body(names[-1]))
        self.spec.delete(self.spec.mesh(names[-1]))
        self._changes.remove("weight")
        self._rebuild()
        return True

    def _weights(self) -> list[str]:
        return sorted(b.name for b in self.spec.bodies if b.name.startswith(self.prefix + "weight"))

    def add_bumper(self):
        """The V-shaped cardboard pusher on the nose; it cups the parcel toward the centre line."""
        b = self.spec.body(self.prefix + "car").add_body(name=self.prefix + "bumper", pos=[0.142, 0, 0.042])
        for side in (1, -1):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.045, 0.0015, 0.025], pos=[0, side * 0.04, 0],
                       euler=[0, 0, side * 63.4], material="cardboard", mass=0.015, group=1)
        self._rebuild()

    def add_tray(self):
        """A shallow tray on the top deck, its lip in the robot's colour: totes ride in it."""
        from airo.warehouse import DECK, TRAY

        hx, hy = TRAY
        b = self.spec.body(self.prefix + "car").add_body(name=self.prefix + "tray", pos=list(DECK))
        kw = {"contype": 1, "conaffinity": 1, "group": 1, "friction": [0.9, 0.005, 0.0001]}
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[hx + 0.004, hy + 0.004, 0.0015], pos=[0, 0, -0.0015], material="tray",
                   mass=0.02, **kw)
        accent = self.prefix + "accent" if self.prefix else "accent"
        for sx, sy, size in ((1, 0, (0.002, hy + 0.004)), (-1, 0, (0.002, hy + 0.004)), (0, 1, (hx + 0.004, 0.002)),
                             (0, -1, (hx + 0.004, 0.002))):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[*size, 0.0045], pos=[sx * (hx + 0.002), sy * (hy + 0.002), 0.0045],
                       material=accent, mass=0.003, **kw)
        self.world.cell.add_latches(self.spec, self.prefix)

    def attach_tow(self, box=(0.12, 0.1, 0.08), mass: float = 0.15, length: float = 0.25):
        """A cardboard box on a string from the rear hitch, dragging on the floor."""
        if "tow" in self._changes:
            return
        self._changes.append("tow")
        car = self.spec.body(self.prefix + "car")
        x, y, th = self.true_pose if self.m is not None else (*car.pos[:2], _yaw(car.quat))
        back = 0.118 + length * 0.95 + box[0] / 2
        name = self.prefix + "tow"
        b = self.spec.worldbody.add_body(name=name, pos=[x - back * math.cos(th), y - back * math.sin(th),
                                                         box[2] / 2], quat=scene.quat(th))
        b.add_freejoint()
        b.add_geom(type=GEOM.mjGEOM_MESH, meshname=scene.rounded(self.spec, name, np.array(box) / 2),
                   material="cardboard", mass=mass, friction=[0.5, 0.005, 0.0001])
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[box[0] / 2 + 0.0004, 0.02, box[2] / 2 + 0.0004], material="tape",
                   contype=0, conaffinity=0, mass=0)
        b.add_site(name=name, pos=[box[0] / 2, 0, box[2] * 0.1])
        t = self.spec.add_tendon(name=self.prefix + "string", limited=True, range=[0, length], width=0.0012,
                                 material="string")
        t.wrap_site(self.prefix + "hitch")
        t.wrap_site(name)
        self._rebuild()

    def detach_tow(self) -> bool:
        """Cut the string; the box is cleared away with it."""
        if "tow" not in self._changes:
            return False
        self.spec.delete(self.spec.tendon(self.prefix + "string"))
        self.spec.delete(self.spec.body(self.prefix + "tow"))
        self.spec.delete(self.spec.mesh(self.prefix + "tow"))
        self._changes.remove("tow")
        self._rebuild()
        return True

    def slippery_patch(self, rect=None, friction: float = 0.12):
        """The world's spill: a place change, kept here for the single-car API."""
        self.world.spill(rect, friction)

    def flip_motor(self, side: str):
        """A hidden fault: that side's motors run backwards. Flipping again rewires it."""
        self._sign[side] *= -1
        if self._sign[side] < 0:
            self._changes.append("flip")
        else:
            self._changes.remove("flip")
        self._rebuild()

    def _rebuild(self):
        self.world.rebuild()

    # rig interface

    @property
    def true_pose(self) -> np.ndarray:
        q = self.world.d.qpos[self._qadr:self._qadr + 7]
        return np.array([q[0], q[1], _yaw(q[3:7])])

    @property
    def H(self) -> np.ndarray:
        if self.tracker is not None and self.tracker.H is not None:
            return self.tracker.H
        return self.world.H

    def _measure(self) -> Pose | None:
        if self.tracker is not None:
            return self.tracker.update(self.frame(), self.t)
        x, y, th = self.true_pose
        n = self._rng.normal(0, 1, 3) * (self.noise[0], self.noise[0], self.noise[1])
        return Pose(float(x + n[0]), float(y + n[1]), float(wrap(th + n[2])), self.t)

    def _now(self) -> float:
        """World time in real time; the car's own clock in lockstep, unless it has fallen behind the others."""
        w = self.world
        return w.t if w.realtime or self.t < w.t - DT - 1e-9 else self.t

    def _hold(self, now: float):
        """The motors hold the last command between steps until the watchdog stops them. body.command_at
        interpolates the history linearly, so without this record a pause between steps would read as a ramp."""
        if not self._cmd_t or now - self._cmd_t[-1] < 1.5 * DT:
            return
        t, last = self._cmd_t[-1], self._cmd[-1]
        if any(last) and now - t > WATCHDOG + DT:
            self._cmd_t += [t + WATCHDOG, t + WATCHDOG + DT]
            self._cmd += [last, (0, 0)]
            last = (0, 0)
        self._cmd_t.append(now - DT)
        self._cmd.append(last)

    async def step(self, pwm_l: int, pwm_r: int) -> Pose | None:
        now = self._now()
        self._hold(now)
        self._queue.append((now + self.delay, pwm_l, pwm_r))
        self._sent = now
        await self.world.until(now + DT)
        self.t = self.world.t if self.realtime else now + DT
        pose = self._measure()
        if not self.realtime:
            # measured before yielding: in lockstep another car's step may advance the world next
            await asyncio.sleep(0)
        return await self._tick(now, pwm_l, pwm_r, True, pose)

    async def stop(self):
        now = self._now()
        self._hold(now)
        self._queue.append((now + self.delay, 0, 0))
        self._sent = now
        self._cmd_t.append(now)
        self._cmd.append((0, 0))

    async def close(self):
        if self._own:
            self.world.close()
        await super().close()

    async def sam(self, image: np.ndarray, concept: str) -> list[meta_api.Segment]:
        return await self.world.sam(image, concept)

    def frame(self) -> np.ndarray:
        return self.world.frame()

    def clear(self, path) -> bool:
        """Whether this car can drive the poses in path without hitting anything in its world."""
        return self.world.clear(path, self)

    def render_ghosts(self, poses, alpha: float = 0.25):
        """Ghost cars drawn in view(): poses (k, 3) are ghosts, (k, n, 3) are k predicted paths ending in ghosts."""
        self.ghosts = [(np.asarray(poses, float), alpha)] if len(poses) else []

    def view(self, width: int = 1280, height: int = 720, camera: str | mujoco.MjvCamera = "chase") -> np.ndarray:
        return self.world.view(width, height, self._follow() if camera == "chase" else camera)

    def select(self, u: float, v: float, width: int = 1280, height: int = 720):
        return self.world.select(u, v, width, height)

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

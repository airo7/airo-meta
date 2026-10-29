"""Loaders: robot arms that put totes on cars and take them off. Motion is kinematic and scripted: joint keyframes
from inverse kinematics, eased between, written into the physics state every substep. A gripped tote rides the tool
rigidly; released, it settles under physics.

The arm model is swappable: a Loader needs `joints`, `fingers`, `tcp`, `home`, `ik(xyz, yaw, near)`, `grip(open)`,
`finger_q(grip)`, `hold(tote half size)` and the HOP and LIFT clearances from it. `Arm` is our procedural six-axis
industrial arm with a parallel gripper; cobot.UR5e is the Menagerie one."""

import asyncio
import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

GEOM = mujoco.mjtGeom
STATIC = {"contype": 0, "conaffinity": 0}


def smooth(s: float) -> float:
    """Minimum-jerk easing: zero velocity and acceleration at both ends."""
    s = min(max(s, 0.0), 1.0)
    return s * s * s * (10 - 15 * s + 6 * s * s)


def _wrap(a: float) -> float:
    return (a + math.pi) % math.tau - math.pi


def yaw_quat(theta: float) -> np.ndarray:
    return np.array([math.cos(theta / 2), 0, 0, math.sin(theta / 2)])


class Arm:
    """A six-axis arm on a pedestal: base yaw, shoulder and elbow pitch, forearm roll, wrist pitch, flange roll, and
    a two-finger gripper. At zero the upper arm stands up and the forearm points along +x of the base frame."""

    MODEL = None
    # what a car could hit: the turret, not the flat base plate it would roll over
    FOOTING = 0.052
    HOP, LIFT = 0.09, 0.12
    SHOULDER = (0.02, 0.135)
    L1, L2 = 0.22, 0.22
    WRIST = 0.03
    TCP = 0.08
    OPEN, CLOSED = 0.066, 0.0525

    def __init__(self, name: str, base: tuple[float, float], yaw: float, paint: str = "robot_orange"):
        self.name, self.base, self.yaw, self.paint = name, np.array(base, float), yaw, paint
        self.joints = [f"{name}/j{k}" for k in range(1, 7)]
        self.fingers = [f"{name}/finger_l", f"{name}/finger_r"]
        self.tcp = f"{name}/tcp"
        self.home = self.ik(self.local(0.12, 0.0, 0.34), yaw)

    def local(self, fwd: float, left: float, z: float) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.base[0] + fwd * c - left * s, self.base[1] + fwd * s + left * c, z])

    @property
    def reach(self) -> float:
        return self.L1 + self.L2

    def ik(self, xyz, yaw: float, near: np.ndarray | None = None) -> np.ndarray:
        """Joint angles that put the tool point at xyz pointing straight down, the fingers closing across yaw.
        Elbow up; near picks the base turn and flange roll closest to a previous pose."""
        rel = np.asarray(xyz, float) - (*self.base, 0)
        q1 = _wrap(math.atan2(rel[1], rel[0]) - self.yaw)
        if near is not None:
            q1 = near[0] + _wrap(q1 - near[0])
        r = math.hypot(rel[0], rel[1]) - self.SHOULDER[0]
        z = rel[2] + self.WRIST + self.TCP - self.SHOULDER[1]
        d2 = r * r + z * z
        c = (d2 - self.L1 ** 2 - self.L2 ** 2) / (2 * self.L1 * self.L2)
        if c > 1:
            raise ValueError(f"{self.name}: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) is out of reach")
        g = math.acos(max(-1.0, c))
        a = math.atan2(z, r) + math.atan2(self.L2 * math.sin(g), self.L1 + self.L2 * math.cos(g))
        b = a - g
        q2 = math.pi / 2 - a
        q3 = -b - q2
        q5 = math.pi / 2 + b
        # the fingers close across the arm's plane, turned by -q6 about the downward tool axis
        q6 = _wrap(self.yaw + q1 - yaw)
        q6 = (q6 + math.pi / 2) % math.pi - math.pi / 2
        if near is not None:
            q6 = near[5] + _wrap(2 * (q6 - near[5])) / 2
        return np.array([q1, q2, q3, 0.0, q5, q6])

    def grip(self, open_: bool) -> float:
        return self.OPEN if open_ else self.CLOSED

    def finger_q(self, g: float) -> float:
        return g

    def hold(self, half) -> np.ndarray:
        """The tool point on a tote in its grasp frame: centred, the fingers across its middle."""
        return np.array([0.0, 0.0, half[2] - 0.015])

    def build(self, spec, label: str):
        """The arm into spec's world, every part in the arm's name space so a pick names the loader."""
        n, p = self.name, self.paint
        ped = spec.worldbody.add_body(name=n, pos=[*self.base, 0], quat=yaw_quat(self.yaw).tolist())
        ped.add_geom(type=GEOM.mjGEOM_BOX, size=[0.075, 0.075, 0.004], pos=[0, 0, 0.004], material="arm_dark",
                     **STATIC)
        for sx in (1, -1):
            for sy in (1, -1):
                ped.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.006, 0.003], pos=[sx * 0.06, sy * 0.06, 0.009],
                             material="steel", **STATIC)
        ped.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.05, 0.016], pos=[0, 0, 0.024], material="arm_dark", **STATIC)
        ped.add_geom(type=GEOM.mjGEOM_BOX, size=[0.03, 0.0005, 0.006], pos=[0, -0.0503, 0.026], material="arm_plate",
                     **STATIC)

        def link(parent, k, pos, axis):
            b = parent.add_body(name=f"{n}/l{k}", pos=pos, gravcomp=1)
            b.add_joint(name=f"{n}/j{k}", type=mujoco.mjtJoint.mjJNT_HINGE, axis=axis)
            return b

        cyl = lambda b, size, pos, mat, axis=None: b.add_geom(
            type=GEOM.mjGEOM_CYLINDER, size=size, pos=pos, material=mat, **STATIC,
            **({"zaxis": axis} if axis else {}))
        box = lambda b, size, pos, mat: b.add_geom(type=GEOM.mjGEOM_BOX, size=size, pos=pos, material=mat, **STATIC)
        mesh = lambda b, half, pos, mat: b.add_geom(type=GEOM.mjGEOM_MESH, pos=pos, material=mat, **STATIC,
                                                    meshname=_rounded(spec, f"{b.name}/{len(b.geoms)}", half))
        turret = link(ped, 1, [0, 0, 0.04], [0, 0, 1])
        cyl(turret, [0.052, 0.005], [0, 0, 0.005], "arm_dark")
        mesh(turret, (0.05, 0.045, 0.04), [0.005, 0, 0.05], p)
        cyl(turret, [0.017, 0.026], [-0.05, 0, 0.055], "arm_motor")
        cyl(turret, [0.0175, 0.004], [-0.05, 0, 0.083], "arm_dark")
        upper = link(turret, 2, [self.SHOULDER[0], 0, self.SHOULDER[1] - 0.04], [0, 1, 0])
        cyl(upper, [0.036, 0.04], [0, 0, 0], p, [0, 1, 0])
        cyl(upper, [0.026, 0.0045], [0, 0.044, 0], "arm_dark", [0, 1, 0])
        cyl(upper, [0.019, 0.022], [0, -0.062, 0], "arm_motor", [0, 1, 0])
        mesh(upper, (0.03, 0.03, 0.055), [0, 0, 0.06], p)
        mesh(upper, (0.022, 0.025, 0.075), [0, 0, 0.14], p)
        cyl(upper, [0.011, 0.045], [-0.036, 0.022, 0.07], "arm_dark")
        cyl(upper, [0.006, 0.03], [-0.036, 0.022, 0.135], "steel")
        cyl(upper, [0.0055, 0.075], [0.03, -0.03, 0.11], "cable")
        elbow = link(upper, 3, [0, 0, self.L1], [0, 1, 0])
        cyl(elbow, [0.03, 0.034], [0, 0, 0], p, [0, 1, 0])
        cyl(elbow, [0.021, 0.0045], [0, 0.038, 0], "arm_dark", [0, 1, 0])
        mesh(elbow, (0.03, 0.026, 0.028), [-0.045, 0, 0.004], "arm_motor")
        mesh(elbow, (0.075, 0.021, 0.023), [0.06, 0, 0], p)
        box(elbow, [0.04, 0.0005, 0.008], [0.06, -0.0215, 0], "arm_plate")
        cyl(elbow, [0.005, 0.06], [0.05, 0, 0.027], "cable", [1, 0, 0])
        roll = link(elbow, 4, [0.13, 0, 0], [1, 0, 0])
        cyl(roll, [0.018, 0.04], [0.04, 0, 0], p, [1, 0, 0])
        cyl(roll, [0.0185, 0.003], [0.002, 0, 0], "arm_dark", [1, 0, 0])
        wrist = link(roll, 5, [self.L2 - 0.13, 0, 0], [0, 1, 0])
        cyl(wrist, [0.019, 0.022], [0, 0, 0], p, [0, 1, 0])
        cyl(wrist, [0.012, 0.0035], [0, 0.024, 0], "arm_dark", [0, 1, 0])
        flange = link(wrist, 6, [self.WRIST, 0, 0], [1, 0, 0])
        cyl(flange, [0.016, 0.004], [-0.006, 0, 0], "steel", [1, 0, 0])
        cyl(flange, [0.013, 0.006], [0.004, 0, 0], "arm_dark", [1, 0, 0])
        box(flange, [0.016, 0.045, 0.017], [0.026, 0, 0], "gripper")
        box(flange, [0.0165, 0.02, 0.0005], [0.026, 0, 0.0172], "arm_plate")
        for side, fname in ((1, self.fingers[0]), (-1, self.fingers[1])):
            f = flange.add_body(name=fname, pos=[0, 0, 0], gravcomp=1)
            f.add_joint(name=fname, type=mujoco.mjtJoint.mjJNT_SLIDE, axis=[0, side, 0])
            box(f, [0.005, 0.003, 0.016], [0.045, 0, 0], "gripper")
            box(f, [0.022, 0.0025, 0.014], [0.07, 0, 0], "alu")
            box(f, [0.016, 0.001, 0.011], [0.074, -side * 0.0032, 0], "rubber")
        flange.add_body(name=self.tcp, pos=[self.TCP, 0, 0])
        return ped


def _rounded(spec, name: str, half) -> str:
    spec.add_mesh(name=name, scale=list(half)).make_supersphere(16, 0.18, 0.12)
    return name


@dataclass
class _Segment:
    t0: float
    t1: float
    q0: np.ndarray
    q1: np.ndarray
    event: tuple | None = None


@dataclass
class Loader:
    """One arm at a cell. Moves are queued as keyframe segments on the loader's clock, which is world time less
    the time spent paused (a person in the light curtain); act(t) poses the arm each physics substep and fires grip
    and release as their segments end. load and unload hold the car still until done."""
    title: str
    arm: Arm
    cell: object
    doing: str = "idle"
    paused: bool = False
    _plan: list[_Segment] = field(default_factory=list)
    _q: np.ndarray | None = None
    _lag: float = 0.0
    _last: float = 0.0

    def bind(self, m: mujoco.MjModel):
        adr = lambda j: m.jnt_qposadr[m.joint(j).id]
        self._adr = np.array([adr(j) for j in self.arm.joints])
        self._fadr = np.array([adr(f) for f in self.arm.fingers])
        self._dofs = np.array([m.jnt_dofadr[m.joint(j).id] for j in self.arm.joints + self.arm.fingers])
        self.tcp_id = m.body(self.arm.tcp).id
        self.root_id = m.body(self.arm.name).id
        if self._q is None:
            self._q = np.r_[self.arm.home, self.arm.grip(True)]
        self._pose(self.cell.world.d, self._q)

    @property
    def busy(self) -> bool:
        return bool(self._plan)

    def _pose(self, d, q):
        d.qpos[self._adr] = q[:6]
        d.qpos[self._fadr] = self.arm.finger_q(q[6])
        d.qvel[self._dofs] = 0

    def act(self, t: float):
        d = self.cell.world.d
        if self.paused:
            self._lag += t - self._last
        self._last = t
        t -= self._lag
        while self._plan and t >= self._plan[0].t1:
            seg = self._plan.pop(0)
            self._q = seg.q1
            if seg.event:
                self.cell.event(self, *seg.event)
        if self._plan and t >= self._plan[0].t0:
            seg = self._plan[0]
            self._q = seg.q0 + (seg.q1 - seg.q0) * smooth((t - seg.t0) / (seg.t1 - seg.t0))
        self._pose(d, self._q)

    def _queue(self, keys: list[tuple]):
        """keys: (arm joints or None to hold, fingers open, seconds, event at the end)."""
        self._last = self.cell.world.t
        t = self._last - self._lag
        self.cell.wake()
        q = self._q if not self._plan else self._plan[-1].q1
        for joints, open_, secs, event in keys:
            q1 = np.r_[q[:6] if joints is None else joints, self.arm.grip(open_)]
            self._plan.append(_Segment(t, t + secs, q, q1, event))
            t, q = t + secs, q1

    def _grasp(self, centre, yaw: float) -> tuple[np.ndarray, float]:
        """The tool point for a tote centred at centre and turned yaw, and the grasp yaw: of the tote's two ways
        round, the one that puts the arm's hold nearer the arm."""
        off = self.arm.hold(self.cell.tote)
        ways = []
        for g in (yaw, yaw + math.pi):
            c, s = math.cos(g), math.sin(g)
            ways.append((np.asarray(centre, float) + (c * off[0] - s * off[1], s * off[0] + c * off[1], off[2]), g))
        far = [np.linalg.norm(p[:2] - self.arm.base) for p, _ in ways]
        return ways[1] if far[1] < far[0] - 1e-9 else ways[0]

    def _keys(self, at, yaw, start, gripping: bool, tote, near) -> tuple[list, np.ndarray]:
        """Hover, descend, close or open, rise: the keyframes around one grasp or placement."""
        above = self.arm.ik(at + (0, 0, self.arm.HOP), yaw, near)
        down = self.arm.ik(at, yaw, above)
        up = self.arm.ik(at + (0, 0, self.arm.LIFT), yaw, down)
        keys = [(above, not gripping, start, None), (down, not gripping, 0.6, None),
                (None, gripping, 0.35, ("grip" if gripping else "release", tote)), (up, gripping, 0.55, None)]
        return keys, up

    async def load(self, car, tote) -> None:
        """Take tote from where it waits, put it on the car's deck, return home."""
        self.doing = f"loading {car.name}"
        try:
            pick, up = self._keys(*self._grasp(*self.cell.tote_pose(tote)), 1.1, True, tote, self._q)
            dst, dst_yaw = self.cell.deck_pose(car)
            place, _ = self._keys(*self._grasp(dst, dst_yaw), 1.4, False, tote, up)
            self._queue(pick + place + [(self.arm.home, True, 1.0, None)])
            await self._wait(car)
        finally:
            self.doing = "idle"

    async def unload(self, car, slot) -> None:
        """Take the tote off the car and set it down at slot: (x, y, z of the surface, yaw)."""
        self.doing = f"unloading {car.name}"
        try:
            tote = self.cell.on(car)
            self.cell.reserve(slot, tote)
            pick, up = self._keys(*self._grasp(*self.cell.tote_pose(tote)), 1.1, True, tote, self._q)
            dst, dst_yaw = self.cell.slot_pose(slot)
            place, _ = self._keys(*self._grasp(dst, dst_yaw), 1.4, False, tote, up)
            self._queue(pick + place + [(self.arm.home, True, 1.0, None)])
            await self._wait(car)
        finally:
            self.doing = "idle"

    async def _wait(self, car):
        """The car stands still, its rig ticking, while the arm works; the world clock paces the motion."""
        while self.busy:
            await car.step(0, 0)
            await asyncio.sleep(0)

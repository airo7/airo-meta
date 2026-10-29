"""Humanoid workers: Unitree G1s from MuJoCo Menagerie at the hall's 1:4 scale, animated kinematically.

Each worker is the G1's link tree without joints under a mocap pelvis; a pose is written into the model's body
quaternions, so the workers cost the physics nothing. The gait is joint keyframes per phase (hip, knee and ankle
with the legs half a cycle apart, arm swing against the legs, a little waist counter-turn); the pelvis height comes
from forward kinematics so the lower sole touches the floor, and the phase advances with the ground speed so the
stance foot stays planted. A worker follows a plan of steps: walk a polyline (turning on the way, slowing into
corners), face a heading, hold a posture for a while, pick up or set down a carton."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cache

import mujoco
import numpy as np

from airo import menagerie
from airo.menagerie import G1, SCALE, graft

GEOM = mujoco.mjtGeom
STATIC = {"contype": 0, "conaffinity": 0}
SPEED = 0.16
TURN = 2.4
EASE = 0.18
SIDES = ("left", "right")
LEG = ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll", "wrist_pitch", "wrist_yaw")
JOINTS = ([f"left_{j}_joint" for j in LEG] + [f"right_{j}_joint" for j in LEG]
          + ["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"]
          + [f"left_{j}_joint" for j in ARM] + [f"right_{j}_joint" for j in ARM])
J = {n.removesuffix("_joint"): i for i, n in enumerate(JOINTS)}

# Postures: hip pitch and knee per leg (the ankle keeps the foot flat), waist pitch, and the arms. Pitch is positive
# backward, so a forward reach is negative; elbow 0 holds the forearm level, about 1.4 lets it hang.
STAND = {"hip": -0.04, "knee": 0.1, "waist": 0.0, "shoulder": 0.18, "roll": 0.12, "elbow": 1.2}
POSTURES = {
    "stand": STAND,
    "carry": {**STAND, "shoulder": -0.42, "roll": 0.06, "elbow": 0.38},
    "reach_low": {"hip": -0.75, "knee": 1.25, "waist": 0.42, "shoulder": -1.05, "roll": 0.1, "elbow": 0.35},
    "reach_mid": {**STAND, "waist": 0.12, "shoulder": -1.15, "roll": 0.08, "elbow": 0.15},
    "work": {**STAND, "hip": -0.12, "knee": 0.25, "waist": 0.22, "shoulder": -0.6, "roll": 0.1, "elbow": 0.75},
}


def available() -> bool:
    return menagerie.available(G1)


@cache
def rig() -> "Rig":
    return Rig()


class Rig:
    """The G1's kinematics at scale: a jointed copy for forward kinematics, the hinge axes and rest quaternions of
    the links they turn, and the sole points of each foot."""

    def __init__(self):
        spec = mujoco.MjSpec()
        graft(spec, G1, "pelvis", spec.worldbody, "", pos=(0, 0, 0), quat=(1, 0, 0, 0), geoms=False)
        self.m = spec.compile()
        self.d = mujoco.MjData(self.m)
        src = menagerie._src(G1)
        self.links = [src.body(int(src.jnt_bodyid[src.joint(n).id])).name for n in JOINTS]
        self.axes = np.array([src.joint(n).axis for n in JOINTS])
        self.rest = np.array([src.body(b).quat for b in self.links])
        self.qadr = np.array([self.m.jnt_qposadr[self.m.joint(n).id] for n in JOINTS])
        self.feet = [self.m.body(f"{s}_ankle_roll_link").id for s in SIDES]
        self.soles = [np.array([src.geom_pos[g] * SCALE - (0, 0, src.geom_size[g][0] * SCALE)
                                for g in range(src.ngeom) if src.geom_bodyid[g] == src.body(f"{s}_ankle_roll_link").id
                                and src.geom_type[g] == GEOM.mjGEOM_SPHERE]) for s in SIDES]
        self.stride = self._stride()

    def soles_at(self, q: np.ndarray) -> list[np.ndarray]:
        """Sole points of each foot in the pelvis frame for joint angles q."""
        self.d.qpos[self.qadr] = q
        mujoco.mj_kinematics(self.m, self.d)
        return [self.d.xpos[f] + (self.d.xmat[f].reshape(3, 3) @ s.T).T for f, s in zip(self.feet, self.soles)]

    def height(self, q: np.ndarray) -> float:
        """Pelvis height that puts the lower sole on the floor."""
        return -min(s[:, 2].min() for s in self.soles_at(q))

    def _stride(self) -> float:
        """Ground travelled per gait cycle at full amplitude: how far the planted foot slides back under the
        pelvis, which the pelvis must cover forward."""
        n = 72
        xs, low = [], []
        for k in range(n):
            soles = self.soles_at(pose({"posture": STAND, "phase": k / n, "walk": 1.0}))
            z = [s[:, 2].min() for s in soles]
            low.append(int(np.argmin(z)))
            xs.append([s[:, 0].mean() for s in soles])
        xs = np.array(xs)
        back = [max(0.0, xs[k, low[k]] - xs[(k + 1) % n, low[k]]) for k in range(n)]
        return float(sum(back))


def _bump(phase: float, at: float, width: float) -> float:
    d = (phase - at + 0.5) % 1.0 - 0.5
    return math.exp(-0.5 * (d / width) ** 2)


def _leg(phase: float) -> tuple[float, float, float]:
    """(hip pitch, knee, toe) through one cycle from heel strike: stance to 0.6, then swing."""
    hip = -0.06 - 0.3 * math.cos(math.tau * phase)
    knee = 0.1 + 0.18 * _bump(phase, 0.12, 0.07) + 0.85 * _bump(phase, 0.72, 0.1)
    toe = 0.22 * _bump(phase, 0.56, 0.06) - 0.12 * _bump(phase, 0.85, 0.08)
    return hip, knee, toe


def pose(s: dict) -> np.ndarray:
    """Joint angles for a state: posture (a POSTURES entry, possibly blended), phase and walk (0 standing, 1 full
    stride), carry (arms hold the carton while walking), turn (stepping on the spot) and fidget (small hand work)."""
    p, ph, w = s["posture"], s.get("phase", 0.0), s.get("walk", 0.0)
    q = np.zeros(len(JOINTS))
    for k, side in enumerate(SIDES):
        hip, knee, toe = _leg((ph + 0.5 * k) % 1.0)
        hip, knee, toe = (p["hip"] + w * (hip + 0.04), p["knee"] + w * (knee - 0.1), w * toe)
        q[J[f"{side}_hip_pitch"]], q[J[f"{side}_knee"]] = hip, knee
        q[J[f"{side}_ankle_pitch"]] = -(hip + knee) + toe
        sway = 0.035 * w * math.sin(math.tau * ph)
        q[J[f"{side}_hip_roll"]], q[J[f"{side}_ankle_roll"]] = sway, -sway
        sign = 1 if side == "left" else -1
        swing = 0.0 if s.get("carry") else w * 0.32 * math.cos(math.tau * ((ph + 0.5 * k) % 1.0))
        q[J[f"{side}_shoulder_pitch"]] = p["shoulder"] + swing
        q[J[f"{side}_shoulder_roll"]] = sign * p["roll"]
        q[J[f"{side}_elbow"]] = p["elbow"] - 0.25 * w * (not s.get("carry")) + \
            0.12 * s.get("fidget", 0.0) * math.sin(math.tau * 1.3 * ph + k)
        q[J[f"{side}_shoulder_yaw"]] = -sign * 0.15 * (p["elbow"] < 0.5)
    q[J["waist_pitch"]] = p.get("waist", 0.0) + 0.05 * w
    q[J["waist_yaw"]] = 0.07 * w * math.cos(math.tau * ph)
    return q


@dataclass
class Step:
    """One thing a worker does: walk (to the points in order), face (a heading), hold (a posture for secs),
    take or put (the carton, while holding a posture for secs), wait (secs, standing); then runs when it is done."""
    kind: str
    points: list = field(default_factory=list)
    heading: float = 0.0
    posture: str = "stand"
    secs: float = 0.0
    then: Callable[[], None] | None = None


class Worker:
    """A G1 on the floor: pose (x, y, heading), a plan of Steps, and a doing line for pickers and the dashboard."""

    def __init__(self, k: int, role: str, pose0: tuple[float, float, float]):
        self.k, self.role = k, role
        self.id, self.name = f"worker-{k}", f"Worker {k}"
        self.prefix = f"worker{k}/"
        self.x, self.y, self.heading = pose0
        self.plan: list[Step] = []
        self.doing = "standing by"
        self.carrying = False
        self.phase, self.walk = 0.0, 0.0
        self.speed = 0.0
        self.blocked: object | None = None
        self.blocked_s = 0.0
        self._posture = dict(STAND)
        self._t: float | None = None
        self._fidget = 0.0

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.heading

    # building

    def build(self, spec, vest: str):
        """The G1 under a mocap pelvis, a hi-vis vest, and a carton it shows when carrying."""
        self.lod = menagerie.Lod(self.prefix + "torso_link")
        links = graft(spec, G1, "pelvis", spec.worldbody, self.prefix, pos=(self.x, self.y, 0.2), joints=False,
                      lod=self.lod)
        links["pelvis"].mocap = True
        torso = links["torso_link"]
        for x in (0.0205, -0.0165):
            torso.add_geom(type=GEOM.mjGEOM_BOX, size=[0.0016, 0.019, 0.021], pos=[x, 0, 0.037], material=vest,
                           group=1, **STATIC)
            torso.add_geom(type=GEOM.mjGEOM_BOX, size=[0.0004, 0.0192, 0.0025], pos=[x + math.copysign(0.0018, x), 0,
                                                                                     0.031],
                           material="reflective", group=1, **STATIC)
        torso.add_geom(name=self.prefix + "carton", type=GEOM.mjGEOM_BOX, size=[0.03, 0.04, 0.026],
                       pos=[0.075, 0, 0.01], material="carton", group=4, **STATIC)
        torso.add_geom(name=self.prefix + "carton_tape", type=GEOM.mjGEOM_BOX, size=[0.0302, 0.008, 0.0262],
                       pos=[0.075, 0, 0.01], material="tape", group=4, **STATIC)

    def bind(self, m: mujoco.MjModel):
        r = rig()
        self._ids = np.array([m.body(self.prefix + b).id for b in r.links])
        self._mocap = m.body_mocapid[m.body(self.prefix + "pelvis").id]
        self._carton = [m.geom(self.prefix + n).id for n in ("carton", "carton_tape")]

    def apply(self, m: mujoco.MjModel, d: mujoco.MjData):
        """Write the current pose into the model: link rotations, carton shown or not, pelvis on the floor."""
        r = rig()
        q = pose(self.state())
        half = 0.5 * q[:, None]
        turn = np.c_[np.cos(half), np.sin(half) * r.axes]
        m.body_quat[self._ids] = _qmul(r.rest, turn)
        m.geom_group[self._carton] = 1 if self.carrying else 4
        d.mocap_pos[self._mocap] = (self.x, self.y, r.height(q))
        d.mocap_quat[self._mocap] = (math.cos(self.heading / 2), 0, 0, math.sin(self.heading / 2))

    # moving

    def update(self, t: float, cars) -> None:
        dt = 0.0 if self._t is None else min(t - self._t, 0.1)
        self._t = t
        want, walking, turning = "stand", 0.0, 0.0
        step = self.plan[0] if self.plan else None
        if step is not None:
            want, walking, turning = self._do(step, dt, cars)
        target = POSTURES["carry"] if self.carrying and want == "stand" else POSTURES[want]
        k = 1 - math.exp(-dt / EASE) if dt else 1.0
        for key, v in target.items():
            self._posture[key] += k * (v - self._posture[key])
        stride = max(walking, turning)
        self.walk += (1 - math.exp(-dt / 0.15) if dt else 1.0) * (stride - self.walk)
        if self.walk > 0.02:
            rate = (self.speed / (rig().stride * max(self.walk, 0.35)) if walking else 0.0) + 1.3 * turning
            self.phase = (self.phase + rate * dt) % 1.0
        elif self.phase % 0.5 > 0.02:
            self.phase = min(self.phase + 1.5 * dt, math.floor(self.phase / 0.5) * 0.5 + 0.5) % 1.0
        self._fidget = 1.0 if want == "work" else 0.0

    def _do(self, step: Step, dt: float, cars) -> tuple[str, float, float]:
        if step.kind == "walk":
            return self._walk(step, dt, cars)
        if step.kind == "face":
            err = _wrap(step.heading - self.heading)
            turn = math.copysign(min(abs(err), TURN * 0.6 * dt), err)
            self.heading = _wrap(self.heading + turn)
            if abs(err) < 0.03:
                self.plan.pop(0)
            return "stand", 0.0, 0.6 if abs(err) > 0.03 else 0.0
        step.secs -= dt
        if step.secs <= 0:
            self.plan.pop(0)
            if step.kind == "take":
                self.carrying = True
            elif step.kind == "put":
                self.carrying = False
            if step.then:
                step.then()
        return (step.posture if step.kind != "wait" else "stand"), 0.0, 0.0

    def _walk(self, step: Step, dt: float, cars) -> tuple[str, float, float]:
        tx, ty = step.points[0]
        dx, dy = tx - self.x, ty - self.y
        dist = math.hypot(dx, dy)
        if dist < 0.02:
            step.points.pop(0)
            if not step.points:
                self.plan.pop(0)
            return "stand", self.walk, 0.0
        err = _wrap(math.atan2(dy, dx) - self.heading)
        self.heading = _wrap(self.heading + math.copysign(min(abs(err), TURN * dt), err))
        ahead = self._car_ahead(cars)
        if ahead is not None:
            self.blocked_s += dt
            if self.blocked_s > 4.0:
                self._detour(ahead, step)
            self.speed = 0.0
            return "stand", 0.0, 0.0
        self.blocked, self.blocked_s = None, 0.0
        corner = len(step.points) > 1 and dist < 0.12
        v = SPEED * max(0.0, math.cos(err)) ** 2 * (0.7 if corner else 1.0)
        v = min(v, dist / max(dt, 1e-3))
        self.speed = v
        self.x += v * dt * math.cos(self.heading)
        self.y += v * dt * math.sin(self.heading)
        return "stand", v / SPEED, 0.5 * (abs(err) > 0.5)

    def _car_ahead(self, cars):
        """A car standing in the next step or two: people do not walk into robots either."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        for car in cars:
            x, y = car[0] - self.x, car[1] - self.y
            if 0 < x * c + y * s < 0.3 and abs(-x * s + y * c) < 0.2:
                self.blocked = car
                return car
        self.blocked = None
        return None

    def _detour(self, car, step: Step):
        """Step round a robot that stays put: a waypoint beside it, on the side away from the walking line."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        side = 1 if (car[0] - self.x) * -s + (car[1] - self.y) * c < 0 else -1
        step.points.insert(0, (car[0] + side * 0.3 * -s, car[1] + side * 0.3 * c))
        self.blocked_s = 0.0

    def state(self) -> dict:
        return {"posture": self._posture, "phase": self.phase, "walk": self.walk, "carry": self.carrying,
                "fidget": self._fidget}


def _qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = a.T
    w2, x2, y2, z2 = b.T
    return np.c_[w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                 w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def _wrap(a: float) -> float:
    return (a + math.pi) % math.tau - math.pi

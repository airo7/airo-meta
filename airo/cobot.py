"""The station arm: a Universal Robots UR5e with a Robotiq 2F-85 gripper on a pedestal, from MuJoCo Menagerie at
the hall's 1:4 scale. Same contract as loader.Arm (joints, fingers, tcp, home, ik, grip, build), so the Loader's
load and unload are shared. Its IK is numerical: damped least squares on mj_jacBody in a small model of the arm on
its own. It holds a tote by pinching the rim of the long wall nearer to it, the way a 2F-85 takes a Euro tote."""

import math
from functools import cache

import mujoco
import numpy as np

from airo import menagerie
from airo.menagerie import GRIPPER, SCALE, UR5E, graft

GEOM = mujoco.mjtGeom
STATIC = {"contype": 0, "conaffinity": 0}
JOINTS = ("shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_1_joint", "wrist_2_joint",
          "wrist_3_joint")
FINGERS = ("right_driver_joint", "right_coupler_joint", "right_spring_link_joint", "right_follower_joint",
           "left_driver_joint", "left_coupler_joint", "left_spring_link_joint", "left_follower_joint")
# The 2F-85's four-bar linkage, closed: coupler and spring link follow the driver like this (measured on the model).
LINKAGE = np.array([1.0, 0.0, 1.0, -0.96] * 2)
KEY = np.array([-math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0])
PEDESTAL = 0.21


def available() -> bool:
    return menagerie.available(UR5E, GRIPPER)


def _assemble(spec, parent, prefix: str, geoms: bool = True, lod=None):
    """UR5e on top of parent at PEDESTAL, the 2F-85 on its tool flange, a tcp body at the gripper's pinch point:
    z out of the gripper, y the closing axis."""
    ur = graft(spec, UR5E, "base", parent, prefix, pos=(0, 0, PEDESTAL), geoms=geoms, lod=lod)
    m, g = menagerie._src(UR5E), menagerie._src(GRIPPER)
    site = m.site("attachment_site").id
    flange = ur["wrist_3_link"].add_body(name=prefix + "flange", pos=(m.site_pos[site] * SCALE).tolist(),
                                         quat=m.site_quat[site].tolist(), gravcomp=1)
    hand = graft(spec, GRIPPER, "base_mount", flange, prefix + "hand/", geoms=geoms, lod=lod)
    pinch = g.site("pinch").id
    hand["base"].add_body(name=prefix + "tcp", pos=(g.site_pos[pinch] * SCALE).tolist())
    return ur


@cache
def _solver() -> tuple[mujoco.MjModel, mujoco.MjData, int]:
    spec = mujoco.MjSpec()
    _assemble(spec, spec.worldbody.add_body(name="pedestal"), "ur/", geoms=False)
    m = spec.compile()
    return m, mujoco.MjData(m), m.body("ur/tcp").id


class UR5e:
    """At zero yaw the arm faces +x of its frame; base is its pedestal's floor position."""

    MODEL = "UR5e"
    FOOTING = 0.045
    HOP, LIFT = 0.05, 0.075
    OPEN, CLOSED = 0.05, 0.62
    RIM = 0.006

    def __init__(self, name: str, base: tuple[float, float], yaw: float):
        self.name, self.base, self.yaw = name, np.array(base, float), yaw
        self.joints = [f"{name}/{j}" for j in JOINTS]
        self.fingers = [f"{name}/hand/{j}" for j in FINGERS]
        self.tcp = f"{name}/tcp"
        self.home = self.ik(self.local(0.05, 0.0, PEDESTAL + 0.12), yaw + math.pi / 2, KEY)

    def local(self, fwd: float, left: float, z: float) -> np.ndarray:
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        return np.array([self.base[0] + fwd * c - left * s, self.base[1] + fwd * s + left * c, z])

    @property
    def reach(self) -> float:
        m = menagerie._src(UR5E)
        return float(abs(m.body("forearm_link").pos[2]) + abs(m.body("wrist_1_link").pos[2])) * SCALE

    def grip(self, open_: bool) -> float:
        return self.OPEN if open_ else self.CLOSED

    def finger_q(self, g: float) -> np.ndarray:
        return LINKAGE * g

    def hold(self, half) -> np.ndarray:
        """Where the pinch point sits on a tote, in its grasp frame: on the rim of the long wall at +y."""
        return np.array([0.0, half[1] - 0.002, half[2] - self.RIM])

    def ik(self, xyz, yaw: float, near: np.ndarray | None = None) -> np.ndarray:
        """Joint angles that put the pinch point at xyz, the gripper pointing straight down and closing along the
        horizontal normal to yaw (its +y toward yaw + 90°). Seeded from near (or the Menagerie home pose) turned to
        face the target, and wrapped to the turns closest to near."""
        m, d, tcp = _solver()
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        rel = np.asarray(xyz, float) - (*self.base, 0)
        p = np.array([c * rel[0] + s * rel[1], -s * rel[0] + c * rel[1], rel[2]])
        psi = yaw - self.yaw
        y = np.array([-math.sin(psi), math.cos(psi), 0.0])
        z = np.array([0.0, 0.0, -1.0])
        want = np.column_stack([np.cross(y, z), y, z])
        ref = KEY if near is None else np.asarray(near[:6], float)
        q = ref.copy()
        d.qpos[:6] = q
        mujoco.mj_kinematics(m, d)
        q[0] += _wrap(math.atan2(p[1], p[0]) - math.atan2(d.xpos[tcp][1], d.xpos[tcp][0]))
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        for _ in range(300):
            d.qpos[:6] = q
            mujoco.mj_kinematics(m, d)
            mujoco.mj_comPos(m, d)
            have = d.xmat[tcp].reshape(3, 3)
            e = np.r_[p - d.xpos[tcp], 0.5 * sum(np.cross(have[:, k], want[:, k]) for k in range(3))]
            if np.abs(e[:3]).max() < 2e-5 and np.abs(e[3:]).max() < 2e-4:
                break
            mujoco.mj_jacBody(m, d, jp, jr, tcp)
            j = np.r_[jp[:, :6], jr[:, :6]]
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(6), e)
            q += dq * min(1.0, 0.4 / (np.abs(dq).max() + 1e-12))
        else:
            raise ValueError(f"{self.name}: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) is out of reach")
        q = ref + np.array([_wrap(a - r) for a, r in zip(q, ref)])
        if abs(q[2]) > math.pi:
            raise ValueError(f"{self.name}: ({xyz[0]:.2f}, {xyz[1]:.2f}, {xyz[2]:.2f}) is out of reach")
        return q

    def build(self, spec):
        """Pedestal, arm and gripper in the arm's name space, so a pick names the loader."""
        ped = spec.worldbody.add_body(name=self.name, pos=[*self.base, 0], quat=menagerie.yaw_quat(self.yaw))
        box = lambda size, pos, mat: ped.add_geom(type=GEOM.mjGEOM_BOX, size=size, pos=pos, material=mat, **STATIC)
        box([0.045, 0.045, 0.003], [0, 0, 0.003], "arm_dark")
        for sx in (1, -1):
            for sy in (1, -1):
                ped.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.004, 0.0025], pos=[sx * 0.036, sy * 0.036, 0.008],
                             material="steel", **STATIC)
        box([0.026, 0.026, (PEDESTAL - 0.012) / 2], [0, 0, PEDESTAL / 2], "arm_plate")
        box([0.0265, 0.0005, 0.03], [0, -0.0262, PEDESTAL * 0.62], "arm_dark")
        box([0.034, 0.034, 0.003], [0, 0, PEDESTAL - 0.003], "steel")
        self.lods = [menagerie.Lod(self.name + "/wrist_2_link")]
        _assemble(spec, ped, self.name + "/", lod=self.lods[0])
        return ped


def _wrap(a: float) -> float:
    return (a + math.pi) % math.tau - math.pi

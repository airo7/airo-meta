"""An autonomous counterbalance forklift at 1:4 (lidar on the roof, no driver), procedural like the hall: it shuttles a pallet between a rack bay on the back
aisle and a staging lane on the right of the hall. Kinematic: its pose is a mocap body, the fork carriage a link
whose height is written into the model, and it turns about the middle of its front axle the way a rear-steered truck
does. It stops for people in front of it."""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from airo.meshes import Batch

GEOM = mujoco.mjtGeom
STATIC = {"contype": 0, "conaffinity": 0}
SPEED = 0.26
TURN = 1.1
LIFT_V = 0.06
FORK = 0.2
LOW, CARRY = 0.006, 0.04
LENGTH, WIDTH = 0.56, 0.28


@dataclass
class Leg:
    """drive to xy (backward if reverse), turn to heading, lift the forks to z, wait secs, take or leave the
    pallet."""
    kind: str
    xy: tuple[float, float] = (0.0, 0.0)
    heading: float = 0.0
    z: float = 0.0
    secs: float = 0.0
    reverse: bool = False


class Forklift:
    def __init__(self, name: str, pose0: tuple[float, float, float], pallet0: tuple[float, float, float]):
        self.name, self.id = name, "forklift-1"
        self.x, self.y, self.heading = pose0
        self.fork = LOW
        self.pallet = list(pallet0)
        self.loaded = False
        self.plan: list[Leg] = []
        self.doing = "parked"
        self.waiting_for = None
        self._t: float | None = None

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.heading

    def footprint(self) -> np.ndarray:
        """Corners of the truck and its forks (and any pallet on them) on the floor."""
        c, s = math.cos(self.heading), math.sin(self.heading)
        front = FORK + 0.06 + (0.15 if self.loaded else 0.0)
        pts = np.array([[front, WIDTH / 2], [front, -WIDTH / 2], [-LENGTH + 0.1, -WIDTH / 2],
                        [-LENGTH + 0.1, WIDTH / 2]])
        return pts @ np.array([[c, s], [-s, c]]) + (self.x, self.y)

    # building

    def build(self, spec):
        body = spec.worldbody.add_body(name="forklift", pos=[self.x, self.y, 0], mocap=True)
        b = Batch("forklift")
        hw = WIDTH / 2 - 0.012
        b.box("forklift_body", (0.15, hw, 0.045), (-0.2, 0, 0.07))
        b.box("forklift_body", (0.07, hw, 0.03), (-0.02, 0, 0.055))
        b.cylinder("forklift_weight", 0.075, hw, (-0.36, 0, 0.09), axis=(0, 1, 0), sides=20)
        b.box("forklift_weight", (0.04, hw, 0.075), (-0.33, 0, 0.09))
        b.box("hazard_y", (0.002, hw * 0.8, 0.012), (-0.437, 0, 0.12))
        for y in (-1, 1):
            b.box("led_red", (0.003, 0.012, 0.006), (-0.437, y * (hw - 0.03), 0.15))
            b.box("lamp_glow", (0.003, 0.01, 0.006), (0.052, y * (hw - 0.02), 0.09))
            b.box("hazard_k", (0.12, 0.0015, 0.01), (-0.2, y * (hw + 0.001), 0.1))
        b.box("arm_dark", (0.05, hw - 0.03, 0.01), (-0.17, 0, 0.122))
        b.box("arm_dark", (0.028, 0.034, 0.022), (-0.2, 0, 0.143))
        b.box("arm_dark", (0.007, 0.034, 0.03), (-0.228, 0, 0.17))
        b.cylinder("arm_dark", 0.016, 0.003, (-0.11, 0, 0.16), axis=(0.6, 0, 1), sides=12)
        b.box("arm_dark", (0.004, 0.004, 0.025), (-0.12, 0, 0.14))
        for y in (-1, 1):
            b.box("arm_dark", (0.007, 0.007, 0.115), (-0.29, y * (hw - 0.01), 0.24))
            b.box("arm_dark", (0.007, 0.007, 0.12), (-0.035, y * (hw - 0.01), 0.235), euler=(0, -12, 0))
            b.box("arm_dark", (0.135, 0.006, 0.006), (-0.15, y * (hw - 0.01), 0.355))
        for x in np.linspace(-0.27, -0.05, 6):
            b.box("arm_dark", (0.004, hw - 0.01, 0.003), (x, 0, 0.358))
        b.cylinder("arm_dark", 0.018, 0.012, (-0.12, 0, 0.373), sides=16)
        b.cylinder("steel", 0.014, 0.006, (-0.12, 0, 0.391), sides=16)
        b.cylinder("led_amber", 0.009, 0.008, (-0.27, 0, 0.37), sides=12)
        for x, r in ((-0.0, 0.042), (-0.38, 0.034)):
            for y in (-1, 1):
                b.cylinder("rubber", r, 0.02, (x, y * (hw - 0.008), r), axis=(0, 1, 0), sides=18)
                b.cylinder("steel", r * 0.55, 0.0205, (x, y * (hw - 0.008), r), axis=(0, 1, 0), sides=12)
        for y in (-1, 1):
            b.box("forklift_mast", (0.012, 0.016, 0.25), (0.045, y * 0.075, 0.26), euler=(0, -3, 0))
        b.box("forklift_mast", (0.01, 0.075, 0.008), (0.032, 0, 0.5))
        b.box("forklift_mast", (0.01, 0.075, 0.008), (0.045, 0, 0.03))
        b.cylinder("steel", 0.006, 0.08, (0.02, 0.04, 0.2), axis=(0, 0, 1), sides=8)
        b.emit(spec, body)
        for r, z in ((0.045, 0.0006), (0.028, 0.0008)):
            body.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[r, 0.0002], pos=[0.5, 0, z], material="spot_blue",
                          group=1, **STATIC)
        carriage = body.add_body(name="forklift/carriage", pos=[0.065, 0, LOW])
        c = Batch("forklift_carriage")
        c.box("forklift_mast", (0.008, 0.085, 0.04), (0, 0, 0.045))
        c.box("arm_dark", (0.004, 0.08, 0.004), (0.006, 0, 0.088))
        for y in (-0.045, 0.045):
            c.box("forklift_fork", (FORK / 2, 0.012, 0.004), (FORK / 2 + 0.008, y, 0.004))
            c.box("forklift_fork", (0.006, 0.012, 0.045), (0.01, y, 0.045))
        c.emit(spec, carriage)
        load = spec.worldbody.add_body(name="forklift/pallet", pos=list(self.pallet[:2]) + [0], mocap=True)
        from airo.hall import _pallet

        p = Batch("forklift_pallet")
        _pallet(p, np.random.default_rng(11), (0, 0, 0))
        p.emit(spec, load)

    def bind(self, m: mujoco.MjModel):
        self._body = m.body_mocapid[m.body("forklift").id]
        self._pallet = m.body_mocapid[m.body("forklift/pallet").id]
        self._carriage = m.body("forklift/carriage").id

    def apply(self, m: mujoco.MjModel, d: mujoco.MjData):
        d.mocap_pos[self._body] = (self.x, self.y, 0)
        d.mocap_quat[self._body] = _yaw(self.heading)
        m.body_pos[self._carriage][2] = self.fork
        if self.loaded:
            c, s = math.cos(self.heading), math.sin(self.heading)
            self.pallet = [self.x + 0.2 * c, self.y + 0.2 * s, self.heading - math.pi / 2]
            d.mocap_pos[self._pallet] = (*self.pallet[:2], self.fork - LOW)
        else:
            d.mocap_pos[self._pallet] = (*self.pallet[:2], 0)
        d.mocap_quat[self._pallet] = _yaw(self.pallet[2])

    # moving

    def update(self, t: float, people) -> None:
        dt = 0.0 if self._t is None else min(t - self._t, 0.1)
        self._t = t
        if not self.plan:
            return
        leg = self.plan[0]
        self.waiting_for = self._person_ahead(people, leg)
        if self.waiting_for is not None:
            return
        if leg.kind == "drive":
            dx, dy = leg.xy[0] - self.x, leg.xy[1] - self.y
            dist = math.hypot(dx, dy)
            step = min(dist, SPEED * dt * min(1.0, 0.3 + dist / 0.15))
            if dist < 1e-3:
                self.x, self.y = leg.xy
                self.plan.pop(0)
            else:
                self.x += step * dx / dist
                self.y += step * dy / dist
        elif leg.kind == "turn":
            err = (leg.heading - self.heading + math.pi) % math.tau - math.pi
            self.heading += math.copysign(min(abs(err), TURN * dt), err)
            if abs(err) < 1e-3:
                self.heading = leg.heading
                self.plan.pop(0)
        elif leg.kind == "lift":
            err = leg.z - self.fork
            self.fork += math.copysign(min(abs(err), LIFT_V * dt), err)
            if abs(err) < 1e-4:
                self.plan.pop(0)
        else:
            leg.secs -= dt
            if leg.kind in ("take", "leave"):
                self.loaded = leg.kind == "take"
                self.plan.pop(0)
            elif leg.secs <= 0:
                self.plan.pop(0)

    def _person_ahead(self, people, leg: Leg):
        """Someone within a metre (at 1:4) of the way it is moving: it waits."""
        if leg.kind not in ("drive", "turn"):
            return None
        c, s = math.cos(self.heading), math.sin(self.heading)
        sign = -1 if leg.reverse else 1
        for who, (x, y) in people:
            dx, dy = x - self.x, y - self.y
            along, across = sign * (dx * c + dy * s), -dx * s + dy * c
            if leg.kind == "turn" and math.hypot(dx, dy) < 0.5:
                return who
            if -0.1 < along < 0.65 and abs(across) < 0.25:
                return who
        return None


def shuttle(rack: tuple[float, float], stage: tuple[float, float], lane: float, side: float, wait: float,
            outbound: bool) -> list[Leg]:
    """One trip of the pallet between rack (facing +y) and stage (facing -y): out of the rack if outbound, else back
    into it. lane is the back aisle's y, side the staging lane's x. The forks reach 0.2 m past the front axle."""
    up, down, left, right = math.pi / 2, -math.pi / 2, math.pi, 0.0
    at_rack, at_stage = (rack[0], rack[1] - FORK), (stage[0], stage[1] + FORK)
    rack_in, stage_in = (rack[0], lane), (side, stage[1] + 0.55)
    corner = (side, lane)
    to_rack = [Leg("turn", heading=up), Leg("drive", corner), Leg("turn", heading=left), Leg("drive", rack_in),
               Leg("turn", heading=up)]
    to_stage = [Leg("turn", heading=right), Leg("drive", corner), Leg("turn", heading=down), Leg("drive", stage_in)]
    rack_pick = [Leg("lift", z=LOW), Leg("drive", at_rack), Leg("take"), Leg("lift", z=CARRY),
                 Leg("drive", rack_in, reverse=True)]
    rack_drop = [Leg("drive", at_rack), Leg("lift", z=LOW), Leg("leave"), Leg("drive", rack_in, reverse=True)]
    stage_pick = [Leg("lift", z=LOW), Leg("drive", at_stage), Leg("take"), Leg("lift", z=CARRY),
                  Leg("drive", stage_in, reverse=True)]
    stage_drop = [Leg("drive", at_stage), Leg("lift", z=LOW), Leg("leave"), Leg("drive", stage_in, reverse=True)]
    rest = [Leg("wait", secs=wait)]
    if outbound:
        return to_rack + rack_pick + to_stage + stage_drop + rest
    return [Leg("turn", heading=down)] + stage_pick + [Leg("turn", heading=up), Leg("drive", corner)] + \
        [Leg("turn", heading=left), Leg("drive", rack_in), Leg("turn", heading=up)] + rack_drop + \
        [Leg("turn", heading=right), Leg("drive", corner), Leg("turn", heading=down), Leg("drive", stage_in)] + rest


def _yaw(theta: float) -> tuple[float, float, float, float]:
    return math.cos(theta / 2), 0.0, 0.0, math.sin(theta / 2)

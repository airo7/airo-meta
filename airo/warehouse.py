"""The carry site's logistics: a pool of totes, the inbound conveyor at the pick cell, a loader beside every bay,
the slots totes are set down in at each station, and the queues robots wait in for a bay.

Totes are free bodies. The conveyor and the arms move them kinematically; everywhere else (on a car's deck, a
station's table, the conveyor's end stop) physics holds them, so a tote's mass genuinely loads the car it rides."""

import math
from dataclasses import dataclass, field

import mujoco
import numpy as np

from airo.loader import Arm, Loader, yaw_quat
from airo.meshes import shell
from airo.scene import rounded
from airo.site import KINDS, Bay, Site

GEOM = mujoco.mjtGeom
STATIC = {"contype": 0, "conaffinity": 0}
TOTE = (0.07, 0.05, 0.0325)
GRASP = TOTE[2] - 0.015
DECK = (-0.02, 0.0, 0.1035)
TRAY = (0.073, 0.053)
ARM_D = 0.30
BACK = 0.27
BELT_Z = 0.11
BELT_GAP = 0.18
BELT_V = 0.3
BELT_MAX = 4
TABLE_Z = 0.12
POOL = 10
STOCK_X = -1.7
CLEAR = 0.32
TITLES = {"pick": "Pick cell arm", "assembly": "Assembly arm", "qa": "QA arm", "packing": "Packing arm",
          "dock": "Dock arm"}


@dataclass(frozen=True)
class Slot:
    station: str
    k: int
    xyz: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class Cell:
    """A loader cell beside a bay: the arm stands ARM_D to the bay's loader side and faces the car; its work (the
    conveyor end or a station's table) is BACK behind it. out points from the bay toward the arm, along the bay's
    heading."""
    name: str
    bay: Bay
    base: tuple[float, float]
    yaw: float
    out: tuple[float, float]
    along: tuple[float, float]

    def at(self, back: float, side: float = 0.0) -> np.ndarray:
        """The floor point back metres beyond the arm (negative: toward the bay), side metres along the bay."""
        return np.array(self.base) + back * np.array(self.out) + side * np.array(self.along)


def cell(name: str, bay: Bay) -> Cell:
    base = bay.beside(ARM_D)
    out = ((base[0] - bay.x) / ARM_D, (base[1] - bay.y) / ARM_D)
    return Cell(name, bay, base, math.atan2(-out[1], -out[0]), out, (math.cos(bay.theta), math.sin(bay.theta)))


def cells(site: Site) -> dict[str, Cell]:
    return {"pick": cell("pick", site.pick), **{n: cell(n, b) for n, b in site.drops.items()}}


def slots(c: Cell) -> list[Slot]:
    """Four places for totes behind a station's arm, each tote turned to point at the arm. The dock stacks them two
    by two on an outbound pallet; the others line them up on a table."""
    yaw = math.atan2(c.out[1], c.out[0])
    if c.name == "dock":
        spots = [(BACK + d, s, 0.026) for d in (-0.075, 0.075) for s in (-0.056, 0.056)]
    else:
        spots = [(BACK, s, TABLE_Z) for s in (-0.2, -0.067, 0.067, 0.2)]
    return [Slot(c.name, k, (*c.at(b, s), z), yaw) for k, (b, s, z) in enumerate(spots)]


def belt(c: Cell, floor_w: float, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """The inbound conveyor: (end stop, entry inside the wall), totes travelling from entry to end."""
    end = c.at(BACK)
    return end, np.array([-margin - 0.25, end[1]])


@dataclass(eq=False)
class Tote:
    k: int
    kind: str
    state: str = "stock"
    body: int = -1
    qadr: int = 0
    vadr: int = 0
    slide: tuple | None = None
    held: tuple | None = None
    slot: Slot | None = None
    t: float = 0.0
    claimed: bool = False
    latch: tuple | None = None

    @property
    def name(self) -> str:
        return f"tote{self.k}"


class Line:
    """Robots waiting for one bay, first come first served: position 0 has the bay, 1 the nearest queue spot. A
    robot moves up only once the one ahead has reached its own spot, so nobody cuts across a robot still on its
    way in."""

    def __init__(self, name: str, bay: Bay):
        self.name, self.bay, self.robots = name, bay, []
        self.at: dict = {}

    def join(self, who):
        if who not in self.robots:
            self.robots.append(who)

    def leave(self, who):
        if who in self.robots:
            self.robots.remove(who)
        self.at.pop(who, None)

    def arrive(self, who, k: int):
        self.at[who] = k

    def clear(self, who) -> bool:
        """Whether who may drive to its spot: the robot ahead is at its own."""
        k = self.robots.index(who)
        return k == 0 or self.at.get(self.robots[k - 1]) == k - 1

    def spot(self, who) -> tuple[int, tuple[float, float, float] | None]:
        """(place in line, where to wait): the bay itself for the first, then the queue spots, then wherever the
        robot stands."""
        k = self.robots.index(who)
        if k == 0:
            return 0, self.bay.pose
        return k, self.bay.wait[k - 1] if k - 1 < len(self.bay.wait) else None


class Warehouse:
    """Runtime of a carry site in a World: built into the spec with the scene, bound after every compile, acted on
    every physics substep (cheaply when nothing moves)."""

    def __init__(self, world):
        self.world, site = world, world.site
        self.site = site
        self.cells = cells(site)
        self.loaders = {n: Loader(TITLES.get(n, n.capitalize() + " arm"), Arm(f"arm_{n}", c.base, c.yaw), self)
                        for n, c in self.cells.items()}
        self.lines = {n: Line(n, c.bay) for n, c in self.cells.items()}
        self.slots = {n: slots(c) for n, c in self.cells.items() if n != "pick"}
        self.end, self.entry = belt(self.cells["pick"], site.floor[0], 1.0)
        self.dir = (self.end - self.entry) / np.linalg.norm(self.end - self.entry)
        self.totes = [Tote(k, ("light", "heavy")[k % 2]) for k in range(POOL)]
        self.belt: list[Tote] = []
        self.grasp = GRASP
        self._moving = False
        self._active: set[Tote] = set()
        self._pins: dict[Tote, np.ndarray] = {}
        self._pin_q = self._pin_v = np.zeros(0, int)
        self._pin_x = np.zeros(0)

    @property
    def pick(self) -> Loader:
        return self.loaders["pick"]

    # building

    def build(self, spec):
        for n, ld in self.loaders.items():
            ld.arm.build(spec, ld.title)
        for t in self.totes:
            self._tote(spec, t)
        c = self.cells["pick"]
        mid, half = (self.end + self.entry) / 2, np.linalg.norm(self.end - self.entry) / 2 + 0.02
        spec.worldbody.add_geom(name="belt_bed", type=GEOM.mjGEOM_BOX, size=[half, 0.06, 0.005],
                                pos=[*mid, BELT_Z - 0.005], quat=yaw_quat(math.atan2(*self.dir[::-1])).tolist(),
                                group=3, rgba=[0, 0, 0, 0], friction=[0.6, 0.005, 0.0001])
        for n, ss in self.slots.items():
            c = self.cells[n]
            centre = np.mean([s.xyz for s in ss], axis=0)
            spec.worldbody.add_geom(name=f"table_{n}", type=GEOM.mjGEOM_BOX, size=[0.16, 0.3, 0.005],
                                    pos=[*centre[:2], ss[0].xyz[2] - 0.005], quat=yaw_quat(ss[0].yaw).tolist(),
                                    group=3, rgba=[0, 0, 0, 0])

    def _tote(self, spec, t: Tote):
        b = spec.worldbody.add_body(name=t.name, pos=[STOCK_X - 0.2 * (t.k // 5), 0.25 + 0.3 * (t.k % 5),
                                                      TOTE[2] + 0.001])
        b.add_freejoint()
        self._mass(b, t.kind)
        b.add_geom(name=f"{t.name}/hull", type=GEOM.mjGEOM_BOX, size=list(TOTE), group=3, rgba=[0, 0, 0, 0],
                   friction=[0.6, 0.005, 0.0001], density=0)
        b.add_geom(type=GEOM.mjGEOM_MESH, meshname=shell(spec, f"{t.name}/shell", TOTE, 0.004, 0.0035),
                   material="tote", density=0, **STATIC)
        for sx in (1, -1):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.0006, 0.017, 0.006], pos=[sx * (TOTE[0] - 0.0028), 0, 0.017],
                       material="tote_slot", density=0, **STATIC)
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.025, 0.0005, 0.011], pos=[0.012, -TOTE[1] + 0.0028, 0.006],
                   material="label_card", density=0, **STATIC)
        g = lambda kind, **kw: b.add_geom(name=f"{t.name}/{kind}{len(b.geoms)}", density=0,
                                          group=1 if kind == t.kind else 4, **STATIC, **kw)
        g("heavy", type=GEOM.mjGEOM_MESH, meshname=rounded(spec, f"{t.name}/casting", (0.032, 0.027, 0.016)),
          pos=[-0.018, 0.004, -0.008], material="iron")
        g("heavy", type=GEOM.mjGEOM_CYLINDER, size=[0.02, 0.008], pos=[0.036, -0.014, -0.018], material="steel")
        g("heavy", type=GEOM.mjGEOM_CYLINDER, size=[0.009, 0.0081], pos=[0.036, -0.014, -0.018], material="iron")
        g("heavy", type=GEOM.mjGEOM_CYLINDER, size=[0.016, 0.007], pos=[0.038, 0.022, -0.02], material="steel")
        g("light", type=GEOM.mjGEOM_MESH, meshname=rounded(spec, f"{t.name}/box", (0.03, 0.022, 0.018)),
          pos=[-0.02, -0.012, -0.006], euler=[0, 0, 8], material="cardboard")
        g("light", type=GEOM.mjGEOM_BOX, size=[0.0301, 0.006, 0.0181], pos=[-0.02, -0.012, -0.006], euler=[0, 0, 8],
          material="tape")
        g("light", type=GEOM.mjGEOM_MESH, meshname=rounded(spec, f"{t.name}/bag", (0.028, 0.024, 0.01)),
          pos=[0.03, 0.016, -0.014], euler=[0, 0, -12], material="polybag")

    def add_latches(self, spec, prefix: str):
        """A car's tray latch for every tote, off until an arm sets that tote down on the tray. The twin's contacts
        are tuned soft for skid-steer tyres, so an unlatched tote would slide off in a pivot."""
        for t in self.totes:
            spec.add_equality(name=f"{prefix}latch{t.k}", type=mujoco.mjtEq.mjEQ_WELD, objtype=mujoco.mjtObj.mjOBJ_BODY,
                              name1=prefix + "tray", name2=t.name, active=False)

    @staticmethod
    def _mass(b, kind: str):
        kg = KINDS[kind]
        b.mass, b.ipos, b.explicitinertial = kg, [0, 0, 0], True
        b.inertia = [kg / 3 * (TOTE[1] ** 2 + TOTE[2] ** 2), kg / 3 * (TOTE[0] ** 2 + TOTE[2] ** 2),
                     kg / 3 * (TOTE[0] ** 2 + TOTE[1] ** 2)]

    def bind(self, m: mujoco.MjModel):
        for ld in self.loaders.values():
            ld.bind(m)
        for t in self.totes:
            t.body = m.body(t.name).id
            j = m.body_jntadr[t.body]
            t.qadr, t.vadr = m.jnt_qposadr[j], m.jnt_dofadr[j]
            if t.latch is not None:
                self._latch(t, *t.latch)
        self._index()
        self._moving = True

    # tote kinds and poses

    def set_kind(self, t: Tote, kind: str):
        """Swap what's in the tote: its mass for physics, its contents for the camera, in the spec and the model."""
        if kind == t.kind:
            return
        t.kind = kind
        spec, m = self.world.spec, self.world.m
        b = spec.body(t.name)
        self._mass(b, kind)
        m.body_mass[t.body], m.body_inertia[t.body] = b.mass, b.inertia
        # the solver scales constraint stiffness by each body's inverse weight, computed once at compile
        mujoco.mj_setConst(m, mujoco.MjData(m))
        for gs in b.geoms:
            if gs.name.startswith((f"{t.name}/heavy", f"{t.name}/light")):
                gs.group = 1 if gs.name.startswith(f"{t.name}/{kind}") else 4
                m.geom_group[m.geom(gs.name).id] = gs.group

    def _place(self, t: Tote, xyz, yaw: float):
        d = self.world.d
        d.qpos[t.qadr:t.qadr + 7] = (*xyz, *yaw_quat(yaw))
        d.qvel[t.vadr:t.vadr + 6] = 0

    def tote_pose(self, t: Tote) -> tuple[np.ndarray, float]:
        d = self.world.d
        q = d.xquat[t.body]
        return d.xpos[t.body].copy(), math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))

    def deck_pose(self, car) -> tuple[np.ndarray, float]:
        """Where a tote's centre sits on the car's tray, and the car's heading."""
        d, b = self.world.d, self.world.m.body(car.prefix + "tray").id
        return d.xpos[b] + d.xmat[b].reshape(3, 3) @ (0, 0, TOTE[2]), float(car.true_pose[2])

    def slot_pose(self, s: Slot) -> tuple[np.ndarray, float]:
        return np.array([*s.xyz[:2], s.xyz[2] + TOTE[2]]), s.yaw

    def on(self, car) -> Tote | None:
        """The tote riding on car, if any."""
        return next((t for t in self.totes if t.latch is not None and t.latch[0] == car.prefix), None)

    def _latch(self, t: Tote, prefix: str | None, flip: bool = False):
        """Seat t on the car's tray (turned half round if flip) and lock it there; prefix None unlocks it. A latched
        tote rides on the weld alone: its contacts with the tray would fight the lock."""
        m, d = self.world.m, self.world.d
        if t.latch is not None and prefix is None:
            d.eq_active[m.equality(f"{t.latch[0]}latch{t.k}").id] = 0
        t.latch = None if prefix is None else (prefix, flip)
        hull = m.geom(f"{t.name}/hull").id
        m.geom_contype[hull] = m.geom_conaffinity[hull] = prefix is None
        if prefix is None:
            return
        e = m.equality(f"{prefix}latch{t.k}").id
        m.eq_data[e, :10] = (0, 0, 0, 0, 0, TOTE[2], *yaw_quat(math.pi if flip else 0.0))
        d.eq_active[e] = 1

    # the inbound conveyor

    def order(self, kind: str) -> Tote:
        """The tote for a new order: the first unclaimed one on the conveyor, packed with kind. The conveyor holds a
        tote for every open order and one spare."""
        self._feed(extra=1)
        t = next(t for t in self.belt if not t.claimed)
        t.claimed = True
        self.set_kind(t, kind)
        self._feed()
        return t

    def stock(self):
        """At the start, two totes already waiting on the conveyor."""
        if self.belt:
            return
        for i in range(2):
            t = self._restock()
            t.state = "belt"
            self.belt.append(t)
            self._pin(t, (*(self.end - i * BELT_GAP * self.dir), BELT_Z + TOTE[2] + 0.0005,
                          *yaw_quat(self._belt_yaw)))
        mujoco.mj_forward(self.world.m, self.world.d)

    def ready(self, t: Tote) -> bool:
        """At the conveyor's end stop and still."""
        return bool(self.belt) and self.belt[0] is t and t.slide is None

    def _feed(self, extra: int = 0):
        """Keep the conveyor stocked: the head at the end stop, the rest queued behind it, more fetched from stock."""
        for i, t in enumerate(self.belt):
            want = self.end - i * BELT_GAP * self.dir
            if t.slide is None and np.linalg.norm(self._xy(t) - want) > 0.01:
                self._slide(t, want)
        while len(self.belt) < min(BELT_MAX, max(2, sum(t.claimed for t in self.belt) + 1 + extra)):
            start = self.entry
            if self.belt:
                start = self.entry + min(0.0, (self._xy(self.belt[-1]) - self.entry) @ self.dir - BELT_GAP) * self.dir
            t = self._restock()
            self.belt.append(t)
            t.state, t.claimed = "belt", False
            self._place(t, (*start, BELT_Z + TOTE[2] + 0.001), self._belt_yaw)
            self._slide(t, self.end - (len(self.belt) - 1) * BELT_GAP * self.dir)

    def _xy(self, t: Tote) -> np.ndarray:
        return self.world.d.qpos[t.qadr:t.qadr + 2].copy()

    @property
    def _belt_yaw(self) -> float:
        return math.atan2(self.dir[1], self.dir[0])

    def _restock(self) -> Tote:
        """A tote from stock, or failing that the one that has sat longest at a station."""
        stock = [t for t in self.totes if t.state == "stock"]
        if stock:
            return stock[0]
        t = min((t for t in self.totes if t.slot is not None), key=lambda t: t.t)
        self._stow(t)
        return t

    def _slide(self, t: Tote, to):
        self._pin(t, None)
        self._active.add(t)
        p0 = self._xy(t)
        dist = float(np.linalg.norm(to - p0))
        t.slide = (self.world.t, self.world.t + dist / BELT_V, p0, np.array(to, float))
        self._moving = True

    # stations

    def slot(self, station: str) -> Slot:
        """A free place at the station, clearing the longest-waiting tote back into stock if all are taken."""
        taken = {t.slot: t for t in self.totes if t.slot is not None}
        if free := [s for s in self.slots[station] if s not in taken]:
            return free[0]
        old = min((taken[s] for s in self.slots[station]), key=lambda t: t.t)
        s = old.slot
        self._stow(old)
        return s

    def _stow(self, t: Tote):
        self._pin(t, None)
        t.slot, t.state = None, "stock"
        self._place(t, (STOCK_X - 0.2 * (t.k // 5), 0.25 + 0.3 * (t.k % 5), TOTE[2] + 0.001), 0.0)

    def reserve(self, s: Slot, t: Tote):
        t.slot = s

    # time

    def event(self, loader: Loader, what: str, t: Tote):
        d = self.world.d
        if what == "grip":
            if t in self.belt:
                self.belt.remove(t)
            self._pin(t, None)
            self._latch(t, None)
            tp, tq = d.xpos[loader.tcp_id], d.xquat[loader.tcp_id]
            inv, rel_q = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(inv, tq)
            mujoco.mju_mulQuat(rel_q, inv, d.xquat[t.body])
            rel_p = d.xmat[loader.tcp_id].reshape(3, 3).T @ (d.xpos[t.body] - tp)
            t.state, t.held = "held", (loader.tcp_id, rel_p, rel_q)
            self._active.add(t)
        else:
            t.state, t.held, t.t = "free", None, self.world.t
            self._active.discard(t)
            d.qvel[t.vadr:t.vadr + 6] = 0
            if t.slot is not None:
                xyz, yaw = self.slot_pose(t.slot)
                self._pin(t, (*xyz, *yaw_quat(yaw)))
            elif car := min(self.world.cars, key=lambda c: np.linalg.norm(self.deck_pose(c)[0] - d.xpos[t.body])):
                _, yaw = self.tote_pose(t)
                self._latch(t, car.prefix, abs(math.remainder(yaw - car.true_pose[2], math.tau)) > math.pi / 2)
            self._feed()

    def _pin(self, t: Tote, pose):
        """Hold a waiting tote exactly where it is put: at rest on a conveyor or table, physics would let it creep
        under the soft contacts tuned for the tyres. Totes on a car's deck stay free."""
        if pose is None:
            self._pins.pop(t, None)
        else:
            self._pins[t] = np.asarray(pose, float)
            self._place(t, pose[:3], 0.0)
            self.world.d.qpos[t.qadr + 3:t.qadr + 7] = pose[3:]
        self._index()

    def _index(self):
        ts = list(self._pins)
        self._pin_q = np.array([t.qadr + i for t in ts for i in range(7)], int)
        self._pin_v = np.array([t.vadr + i for t in ts for i in range(6)], int)
        self._pin_x = np.concatenate([self._pins[t] for t in ts]) if ts else np.zeros(0)

    def wake(self):
        self._moving = True

    def act(self, t: float):
        """Before each physics substep: pinned totes held, arms to their keyframes, held totes to the tool, sliding
        totes along the conveyor."""
        d = self.world.d
        d.qpos[self._pin_q] = self._pin_x
        d.qvel[self._pin_v] = 0
        if not self._moving:
            return
        busy = False
        for ld in self.loaders.values():
            if ld.busy:
                ld.act(t)
                busy = True
        for tote in list(self._active):
            if tote.held is not None:
                body, rel_p, rel_q = tote.held
                q = np.zeros(4)
                mujoco.mju_mulQuat(q, d.xquat[body], rel_q)
                d.qpos[tote.qadr:tote.qadr + 7] = (*(d.xpos[body] + d.xmat[body].reshape(3, 3) @ rel_p), *q)
                d.qvel[tote.vadr:tote.vadr + 6] = 0
            elif tote.slide is not None:
                t0, t1, p0, p1 = tote.slide
                s = 1.0 if t1 <= t0 else min(1.0, (t - t0) / (t1 - t0))
                xyz = (*(p0 + (p1 - p0) * s), BELT_Z + TOTE[2] + 0.0005)
                self._place(tote, xyz, self._belt_yaw)
                if s >= 1:
                    tote.slide = None
                    self._active.discard(tote)
                    self._pin(tote, (*xyz, *yaw_quat(self._belt_yaw)))
        self._moving = busy or bool(self._active)

    def keepout(self) -> list[np.ndarray]:
        """Floor polygons around each loader cell, for the planner: the arm, its fence and its work behind it."""
        out = []
        for c in self.cells.values():
            a = np.array(c.out)
            along = np.array(c.along)
            near = np.array(c.base) - 0.17 * a
            far = np.array(c.base) + 0.6 * a
            w = 0.27 if c.name == "pick" else 0.3
            out.append(np.array([near - w * along, near + w * along, far + w * along, far - w * along]))
        return out

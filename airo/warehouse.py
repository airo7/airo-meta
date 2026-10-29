"""The carry site's logistics: a pool of totes, the inbound conveyor at the pick cell, a loader beside every bay,
the slots totes are set down in at each station, and the queues robots wait in for a bay.

Totes are free bodies without contacts, so nothing a cell moves can shove a car. The conveyor and the arms move them
kinematically, at rest they are pinned in place, and on a car's deck a weld holds them, so a tote's mass genuinely
loads the car it rides."""

import math
from dataclasses import dataclass

import mujoco
import numpy as np

from airo import cobot, geometry
from airo.loader import Arm, Loader, yaw_quat
from airo.meshes import Batch, shell
from airo.scene import rounded
from airo.site import KINDS, Bay, Site

GEOM = mujoco.mjtGeom
STATIC = {"contype": 0, "conaffinity": 0}
TOTE = (0.07, 0.05, 0.0325)
DECK = (-0.02, 0.0, 0.1035)
TRAY = (0.073, 0.053)
BACK = 0.27
BELT_Z = 0.11
BELT_GAP = 0.18
BELT_V = 0.3
BELT_MAX = 4
TABLE_Z = 0.12
TABLE_BACK = 0.13
PALLET_BACK = 0.155
PALLET_Z = 0.026
POOL = 10
STOCK_X = -1.7
CLEAR = 0.32
TITLES = {"pick": "Pick cell arm", "assembly": "Assembly arm", "qa": "QA arm", "packing": "Packing arm",
          "dock": "Dock arm"}


@dataclass(frozen=True)
class Layout:
    """A cell's footprint, in metres from the bay's centre toward the loader side: the arm's base, the guarding's
    open front and its rear; side is the guarding's half width along the bay."""
    arm: float
    front: float
    rear: float
    side: float = 0.33


# The pick cell is sized for our big arm and the conveyor end; station cells for a cobot that swings between the car
# and a table right behind it, which leaves a back aisle between their guarding and the racks.
HERO = Layout(arm=0.30, front=0.15, rear=0.77)
STATION = Layout(arm=0.225, front=0.12, rear=0.44)
DOCK = Layout(arm=0.225, front=0.12, rear=0.48)


@dataclass(frozen=True)
class Slot:
    station: str
    k: int
    xyz: tuple[float, float, float]
    yaw: float


@dataclass(frozen=True)
class Cell:
    """A loader cell beside a bay: the arm stands layout.arm to the bay's loader side and faces the car; its work
    (the conveyor end or a station's table) is behind it. out points from the bay toward the arm, along the bay's
    heading."""
    name: str
    bay: Bay
    base: tuple[float, float]
    yaw: float
    out: tuple[float, float]
    along: tuple[float, float]
    layout: Layout = HERO

    def at(self, back: float, side: float = 0.0) -> np.ndarray:
        """The floor point back metres beyond the arm (negative: toward the bay), side metres along the bay."""
        return np.array(self.base) + back * np.array(self.out) + side * np.array(self.along)

    @property
    def front(self) -> float:
        """The guarding's open front, from the arm's base (negative: toward the bay)."""
        return self.layout.front - self.layout.arm

    @property
    def rear(self) -> float:
        return self.layout.rear - self.layout.arm

    @property
    def side(self) -> float:
        return self.layout.side


def cell(name: str, bay: Bay) -> Cell:
    layout = HERO if name == "pick" else DOCK if name == "dock" else STATION
    base = bay.beside(layout.arm)
    out = ((base[0] - bay.x) / layout.arm, (base[1] - bay.y) / layout.arm)
    return Cell(name, bay, base, math.atan2(-out[1], -out[0]), out, (math.cos(bay.theta), math.sin(bay.theta)),
                layout)


def cells(site: Site) -> dict[str, Cell]:
    return {"pick": cell("pick", site.pick), **{n: cell(n, b) for n, b in site.drops.items()}}


def slots(c: Cell) -> list[Slot]:
    """Four places for totes on a table right behind a station's arm, each tote pointing at the arm. The dock sets
    them two by two on an outbound pallet, lengthwise, so the arm can reach the far row's near wall."""
    if c.name == "dock":
        yaw = math.atan2(c.along[1], c.along[0])
        spots = [(PALLET_BACK + d, s, PALLET_Z) for d in (-0.052, 0.052) for s in (-0.0725, 0.0725)]
    else:
        yaw = math.atan2(c.out[1], c.out[0])
        spots = [(TABLE_BACK, s, TABLE_Z) for s in (-0.1575, -0.0525, 0.0525, 0.1575)]
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


def _bar(a, b, half: float) -> np.ndarray:
    d = np.asarray(b) - a
    return geometry.rect(*(np.asarray(a) + d / 2), math.atan2(d[1], d[0]), np.linalg.norm(d) / 2 + half, half)


def _stock(t: Tote) -> tuple[float, float, float]:
    return STOCK_X - 0.2 * (t.k // 5), 0.25 + 0.3 * (t.k % 5), TOTE[2]


def _arm(name: str, c: Cell) -> tuple[str, object]:
    """(title, arm): our own arm at the pick cell, and at the stations the Menagerie UR5e when it's installed."""
    title = TITLES.get(name, name.capitalize() + " arm")
    if name != "pick" and cobot.available():
        station = name.upper() if len(name) <= 2 else name.capitalize()
        return f"Station arm · UR5e · {station}", cobot.UR5e(f"arm_{name}", c.base, c.yaw)
    return title, Arm(f"arm_{name}", c.base, c.yaw)


class Warehouse:
    """Runtime of a carry site in a World: built into the spec with the scene, bound after every compile, acted on
    every physics substep (cheaply when nothing moves)."""

    def __init__(self, world):
        self.world, site = world, world.site
        self.site = site
        self.cells = cells(site)
        self.loaders = {n: Loader(*_arm(n, c), self) for n, c in self.cells.items()}
        self.lines = {n: Line(n, c.bay) for n, c in self.cells.items()}
        self.slots = {n: slots(c) for n, c in self.cells.items() if n != "pick"}
        self.end, self.entry = belt(self.cells["pick"], site.floor[0], 1.0)
        self.dir = (self.end - self.entry) / np.linalg.norm(self.end - self.entry)
        self.totes = [Tote(k, ("light", "heavy")[k % 2]) for k in range(POOL)]
        self.belt: list[Tote] = []
        self.tote = TOTE
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

    def _tote(self, spec, t: Tote):
        b = spec.worldbody.add_body(name=t.name, pos=list(_stock(t)))
        b.add_freejoint()
        self._mass(b, t.kind)
        b.add_geom(name=f"{t.name}/hull", type=GEOM.mjGEOM_BOX, size=list(TOTE), group=3, rgba=[0, 0, 0, 0],
                   density=0, **STATIC)
        body = Batch(t.name)
        body.add("tote", shell((TOTE[0] - 0.003, TOTE[1] - 0.003, TOTE[2]), 0.0035, 0.003))
        hx, hy, hz = TOTE
        for sx, sy in ((1, 1), (1, -1), (-1, 1), (-1, -1)):
            body.box("tote", (0.004, 0.004, hz - 0.004), (sx * (hx - 0.006), sy * (hy - 0.006), -0.002))
        for k in np.linspace(-0.04, 0.04, 4):
            for sy in (1, -1):
                body.box("tote", (0.0025, 0.0015, hz - 0.01), (k, sy * (hy - 0.0035), -0.003))
        for sx in (1, -1):
            body.box("tote", (0.0015, 0.0025, hz - 0.01), (sx * (hx - 0.0035), 0.0, -0.003))
        for sx in (1, -1):
            body.box("tote", (0.002, hy, 0.003), (sx * (hx - 0.002), 0, hz - 0.003))
        for sy in (1, -1):
            body.box("tote", (hx, 0.002, 0.003), (0, sy * (hy - 0.002), hz - 0.003))
        body.box("tote_slot", (0.0006, 0.016, 0.0055), (hx - 0.0024, 0, hz - 0.014))
        body.box("tote_slot", (0.0006, 0.016, 0.0055), (-hx + 0.0024, 0, hz - 0.014))
        body.box("label_card", (0.024, 0.0005, 0.01), (0.014, -hy + 0.0009, 0.004))
        body.emit(spec, b, density=0)
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
            elif t.state == "stock" and t not in self._pins:
                self._pins[t] = np.array([*_stock(t), 1, 0, 0, 0])
        self._index()
        self._moving = True

    def label(self, body: str) -> tuple[str, str] | None:
        """A name for a picked part of the cell: "Pick cell arm · loading AIRO 2", "Tote · heavy"."""
        for ld in self.loaders.values():
            if body == ld.arm.name or body.startswith(ld.arm.name + "/"):
                if ld.arm.MODEL:
                    return ld.title, (f"{ld.doing.capitalize()}; Robotiq 2F-85 gripper, MuJoCo Menagerie model at 1:4, "
                                      "numerical IK")
                return f"{ld.title} · {ld.doing}", "six-axis arm, parallel gripper; scripted keyframes"
        t = next((t for t in self.totes if body == t.name), None)
        if t is None:
            return None
        where = (f"latched on {next(c.name for c in self.world.cars if c.prefix == t.latch[0])}'s tray" if t.latch
                 else f"set down at {t.slot.station}" if t.slot else "in the gripper" if t.held
                 else "waiting on the conveyor" if t.state == "belt" else "in stock")
        what = "steel parts" if t.kind == "heavy" else "small parcels"
        return f"Tote · {t.kind}", f"{KINDS[t.kind]:.1f} kg of {what}, {where}"

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
        # The solver scales constraint stiffness by each body's inverse weight, computed once at compile. mj_setConst
        # also resets every weld's relative pose to the one at qpos0, which would yank a latched tote, and its car,
        # back to where they stood at the start: keep the latches.
        latches = m.eq_data.copy()
        mujoco.mj_setConst(m, mujoco.MjData(m))
        m.eq_data[:] = latches
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

    def _latch(self, t: Tote, prefix: str | None, rel=None):
        """Lock t to the car's tray as it sits there (rel: its pose in the tray's frame, kept across recompiles);
        prefix None unlocks it. Locking it where the arm put it, rather than snapping it to the tray's centre, leaves
        the car unjolted."""
        m, d = self.world.m, self.world.d
        if t.latch is not None:
            d.eq_active[m.equality(f"{t.latch[0]}latch{t.k}").id] = 0
        if prefix is None:
            t.latch = None
            return
        if rel is None:
            tray = m.body(prefix + "tray").id
            q, inv = np.zeros(4), np.zeros(4)
            mujoco.mju_negQuat(inv, d.xquat[tray])
            mujoco.mju_mulQuat(q, inv, d.xquat[t.body])
            rel = (*(d.xmat[tray].reshape(3, 3).T @ (d.xpos[t.body] - d.xpos[tray])), *q)
        t.latch = (prefix, tuple(float(v) for v in rel))
        e = m.equality(f"{prefix}latch{t.k}").id
        m.eq_data[e, :10] = (0, 0, 0, *t.latch[1])
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
        t.slot, t.state = None, "stock"
        self._pin(t, (*_stock(t), 1, 0, 0, 0))

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
                self._latch(t, car.prefix)
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

    def structures(self) -> list[np.ndarray]:
        """Outlines of what a car can hit around the cells: the guarding's sides and rear, its front posts, and each
        arm's footing. Not the open front, which is a light curtain."""
        out = []
        t = 0.012
        for name, c in self.cells.items():
            out += [_bar(c.at(c.front, s), c.at(c.rear, s), t) for s in (-c.side, c.side)]
            out.append(_bar(c.at(c.rear, -c.side), c.at(c.rear, c.side), t))
            arm = self.loaders[name].arm
            out.append(geometry.rect(*c.base, c.yaw, arm.FOOTING, arm.FOOTING))
        return out

    def keepout(self) -> list[np.ndarray]:
        """Floor polygons around each loader cell, for the planner: the arm, its fence and its work behind it."""
        out = []
        for c in self.cells.values():
            a = np.array(c.out)
            along = np.array(c.along)
            bay = np.array((c.bay.x, c.bay.y))
            near, far = bay + 0.13 * a, bay + (c.layout.rear + 0.13) * a
            w = 0.27 if c.name == "pick" else 0.3
            out.append(np.array([near - w * along, near + w * along, far + w * along, far - w * along]))
        return out

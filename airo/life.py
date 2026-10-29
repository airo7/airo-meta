"""Life on the factory floor around the fleet: Unitree G1 workers at the racks, on the walkway and at the dock, a
forklift on the back aisle, stack lights and light curtains on every cell, and the events the robots must handle:
a worker crossing their lanes, a pallet or carton dropped in an aisle, a worker stepping into a cell to clear a jam.

A busy level (calm, normal, busy) sets how often things happen; commands make them happen now. Everything here is
kinematic and on world time, so a run is repeatable. The fleet's Guard stops for what hazards() returns; dropped
things are also ordinary obstacles to the planner's scan."""

import math
from collections.abc import Callable
from dataclasses import dataclass

import mujoco
import numpy as np

from airo import hall, scene, workers
from airo.forklift import Forklift, shuttle
from airo.planner import OBSTACLE
from airo.sim import Thing
from airo.workers import Step, Worker

GEOM = mujoco.mjtGeom
STATIC = {"contype": 0, "conaffinity": 0}
RATE = 1 / 60
HIDDEN = 5
WALKWAY = -0.76
LANE = 2.25
PARKED = (-4.0, -4.0, -1.0)
CLEAR_S = 45.0
PALLET, CARTON = (0.2, 0.16), (0.1, 0.08)
# How the dock worker gets to a cell's open front from the dock, around the guarding and the parked cars.
APPROACH = {"packing": [(3.2, 0.0), (3.2, 0.45), (2.72, 0.52)], "dock": [(2.6, -0.52), (1.65, -0.52), (1.65, 0.27)]}


@dataclass(frozen=True)
class Level:
    """Mean seconds between events; None never."""
    cross: float
    forklift: float
    jam: float | None
    drop: float | None


LEVELS = {"calm": Level(cross=110, forklift=40, jam=None, drop=None),
          "normal": Level(cross=45, forklift=15, jam=240, drop=None),
          "busy": Level(cross=20, forklift=4, jam=110, drop=80)}
DEFAULT = "normal"


@dataclass
class Hazard:
    """Something a car must not drive into: a person (stopped for at a distance, like another car) or a thing
    (stopped for just short of the bumper). points outline it on the floor."""
    name: str
    points: np.ndarray
    person: bool
    why: str

    def blocks(self, x: float, y: float, theta: float, direction: float) -> bool:
        # A thing only within the car's own width: the planner passes things with less clearance than a person gets.
        reach, width = (0.42, 0.2) if self.person else (0.22, 0.095)
        rel = self.points - (x, y)
        f = direction * (rel @ (math.cos(theta), math.sin(theta)))
        side = rel @ (-math.sin(theta), math.cos(theta))
        return bool(((f > 0) & (f < reach) & (np.abs(side) < width)).any())


@dataclass
class Drop:
    k: int
    kind: str
    thing: Thing
    body: str
    placed: bool = False
    t: float = 0.0
    yaw: float = 0.0
    clearing: bool = False

    def outline(self) -> np.ndarray:
        hx, hy = np.array(self.thing.size) / 2
        c, s = math.cos(self.yaw), math.sin(self.yaw)
        local = np.array([[sx * hx, sy * hy] for sx in (-1, 0, 1) for sy in (-1, 0, 1)])
        return local @ np.array([[c, s], [-s, c]]) + self.thing.xy


class Life:
    def __init__(self, world, level: str = DEFAULT, seed: int = 7):
        self.world, self.site = world, world.site
        self.level = level
        self.rng = np.random.default_rng(seed)
        self.workers = ([Worker(1, "walker", (-0.4, WALKWAY, 0.0)), Worker(2, "racks", (0.23, 2.5, math.pi / 2)),
                         Worker(3, "dock", (3.0, -0.4, math.pi))] if workers.available() else [])
        rack = (hall.FORKLIFT_SLOT, hall.rack_front(self.site) + hall.RACK_DEPTH / 2)
        self.rack, self.stage = rack, hall.STAGE
        self.forklift = Forklift("Autonomous forklift", (hall.STAGE[0], hall.STAGE[1] + 0.55, -math.pi / 2),
                                 (*hall.STAGE, -math.pi))
        self._outbound = False
        self.drops: list[Drop] = []
        self.paths: Callable[[], list] = lambda: []
        self.tripped: dict[str, bool] = {}
        self._t: float | None = None
        self._next: dict[str, float] = {}

    @property
    def walker(self) -> Worker | None:
        return next((w for w in self.workers if w.role == "walker"), None)

    # building

    def build(self, spec):
        for w in self.workers:
            w.build(spec, "hivis")
        self.forklift.build(spec)
        for k, kind in enumerate(("pallet", "pallet", "carton", "carton", "carton")):
            thing = Thing(OBSTACLE, PARKED[:2], PALLET if kind == "pallet" else CARTON, (60, 110, 160), kind=kind)
            name = f"thing{len(self.world.objects)}"
            self.world.objects.append(thing)
            body = spec.worldbody.add_body(name=name, pos=list(PARKED), mocap=True)
            sx, sy = np.array(thing.size) / 2
            if kind == "pallet":
                scene._pallet(spec, body, sx, sy, collide=True)
            else:
                body.add_geom(type=GEOM.mjGEOM_BOX, size=[sx, sy, 0.03], pos=[0, 0, 0.03], material="carton")
                body.add_geom(type=GEOM.mjGEOM_BOX, size=[sx + 0.0004, 0.012, 0.0304], pos=[0, 0, 0.03],
                              material="tape", **STATIC)
            self.drops.append(Drop(k, kind, thing, name))
        for name, c in self.world.cell.cells.items():
            at = c.at(c.rear, c.side)
            b = spec.worldbody.add_body(name=f"stack_{name}", pos=[*at, hall.FENCE_H])
            for k, colour in enumerate(("green", "amber", "red")):
                b.add_geom(name=f"stack_{name}_{colour}", type=GEOM.mjGEOM_CYLINDER, size=[0.011, 0.012],
                           pos=[0, 0, 0.03 + 0.025 * k], material=f"lamp_{colour}_off", **STATIC)
            for k, s in enumerate((-c.side, c.side)):
                b = spec.worldbody.add_body(name=f"curtain_{name}_{k}", pos=[*c.at(c.front, s), 0.385])
                b.add_geom(name=f"curtain_{name}_{k}", type=GEOM.mjGEOM_BOX, size=[0.0078, 0.0078, 0.008],
                           material="curtain_ok", **STATIC)

    def bind(self, m: mujoco.MjModel):
        for w in self.workers:
            w.bind(m)
        self.forklift.bind(m)
        mat = lambda n: m.material(n).id
        self._lamps = {(n, c): (m.geom(f"stack_{n}_{c}").id, mat(f"lamp_{c}_on"), mat(f"lamp_{c}_off"))
                       for n in self.world.cell.cells for c in ("green", "amber", "red")}
        self._curtains = {n: [m.geom(f"curtain_{n}_{k}").id for k in range(2)] for n in self.world.cell.cells}
        self._curtain_mats = mat("curtain_ok"), mat("curtain_tripped")
        for d in self.drops:
            b = m.body(d.body).id
            d.mocap, d.geoms = m.body_mocapid[b], np.flatnonzero(m.geom_bodyid == b)
            d.groups = m.geom_group[d.geoms].copy()
        self._t = None
        self._apply()

    # time

    def act(self, t: float):
        if self._t is not None and t - self._t < RATE:
            return
        first = self._t is None
        self._t = t
        if first:
            self._schedule(t)
        lv = LEVELS[self.level]
        if t >= self._next["cross"]:
            self.cross()
            self._next["cross"] = t + lv.cross * self.rng.uniform(0.7, 1.3)
        if lv.jam and t >= self._next["jam"]:
            self.jam()
            self._next["jam"] = t + lv.jam * self.rng.uniform(0.7, 1.3)
        if lv.drop and t >= self._next["drop"]:
            self.drop("carton")
            self._next["drop"] = t + lv.drop * self.rng.uniform(0.7, 1.3)
        for d in self.drops:
            if d.placed and d.kind == "carton" and not d.clearing and t - d.t > CLEAR_S:
                self._fetch(d)
        cars = [c.true_pose[:2] for c in self.world.cars]
        for w in self.workers:
            if not w.plan:
                self._program(w)
            w.update(t, cars)
        f = self.forklift
        if not f.plan:
            f.plan = shuttle(self.rack, self.stage, LANE, self.stage[0], lv.forklift * self.rng.uniform(0.7, 1.3),
                             self._outbound)
            f.doing = "taking a pallet to the rack" if not self._outbound else "bringing a pallet down from the rack"
            self._outbound = not self._outbound
        f.update(t, [(w, (w.x, w.y)) for w in self.workers])
        self._cells(t)
        self._apply()

    def _schedule(self, t: float):
        lv = LEVELS[self.level]
        self._next = {"cross": t + 0.4 * lv.cross, "jam": t + (lv.jam or 0) * 0.6,
                      "drop": t + (lv.drop or 0) * 0.6}

    def _apply(self):
        m, d = self.world.m, self.world.d
        for w in self.workers:
            w.apply(m, d)
        self.forklift.apply(m, d)
        for dr in self.drops:
            m.geom_group[dr.geoms] = dr.groups if dr.placed else HIDDEN
            d.mocap_pos[dr.mocap] = (*dr.thing.xy, 0) if dr.placed else PARKED
            d.mocap_quat[dr.mocap] = (math.cos(dr.yaw / 2), 0, 0, math.sin(dr.yaw / 2))

    def _cells(self, t: float):
        """Stack lights: green idle, amber while the arm works or a car stands in the bay, red (flashing) while
        someone is inside the guarding, which also pauses the arm."""
        m, cell = self.world.m, self.world.cell
        for name, c in cell.cells.items():
            people = [w for w in self.workers if _inside(c, (w.x, w.y))]
            tripped = bool(people)
            self.tripped[name] = tripped
            ld = cell.loaders[name]
            ld.paused = tripped
            bay = (c.bay.x, c.bay.y)
            working = ld.busy or any(math.dist(car.true_pose[:2], bay) < 0.1 for car in self.world.cars)
            on = "red" if tripped else "amber" if working else "green"
            flash = not tripped or (t * 2) % 1 < 0.6
            for colour in ("green", "amber", "red"):
                g, lit, dark = self._lamps[name, colour]
                m.geom_matid[g] = lit if colour == on and flash else dark
            for g in self._curtains[name]:
                m.geom_matid[g] = self._curtain_mats[tripped]

    def set_level(self, level: str):
        self.level = level
        if self._t is not None:
            self._schedule(self._t)

    # people

    def _program(self, w: Worker):
        """What a worker does when it has nothing else to do."""
        rng = self.rng
        if w.role == "walker":
            end = -0.7 if w.x > 1.5 else 3.6
            back = [(hall.CROSSING, -0.1), (hall.CROSSING, WALKWAY)] if w.y > WALKWAY + 0.3 else []
            w.plan = [Step("walk", back + [(end, WALKWAY)]), Step("hold", posture="work", secs=rng.uniform(2, 5))]
            w.doing = "walking the walkway"
        elif w.role == "racks":
            x = rng.choice([-0.715, -0.49, 0.23, 0.455])
            front = hall.rack_front(self.site) - 0.1
            cage = hall.CAGES[0]
            restock = rng.random() < 0.35
            w.plan = [Step("walk", [(x, front)]), Step("face", heading=math.pi / 2),
                      Step("put" if restock else "take", posture="reach_low", secs=2.2),
                      Step("walk", [(cage[0], cage[1] + 0.19)]), Step("face", heading=-math.pi / 2),
                      Step("take" if restock else "put", posture="work", secs=1.6),
                      Step("wait", secs=rng.uniform(1, 4))]
            if restock:
                w.plan = w.plan[3:6] + [Step("walk", [(x, front)]), Step("face", heading=math.pi / 2),
                                        Step("put", posture="reach_low", secs=2.2)]
            w.doing = "restocking the racks" if restock else "picking cartons from the racks"
        else:
            cage = hall.CAGES[1]
            w.plan = [Step("walk", [(3.16, -0.4)]), Step("face", heading=0.0),
                      Step("take", posture="reach_low", secs=2.0), Step("walk", [(cage[0] + 0.17, cage[1])]),
                      Step("face", heading=math.pi), Step("put", posture="work", secs=1.8),
                      Step("wait", secs=rng.uniform(2, 6))]
            w.doing = "loading cartons into a roll cage at the dock"

    def cross(self) -> str:
        """The walker takes an order from the rack cage to the dock, across the robots' lanes and back."""
        w = self.walker
        if w is None:
            return "no workers on this floor: the Unitree G1 model is not installed (uv run scripts/fetch_models.py)"
        x0 = hall.CROSSING
        route = [(x0, 0.62), (1.0, 1.05), (0.78, 1.42), (0.76, 1.88), (0.1, 1.9)]
        cage = hall.CAGES[1]
        w.plan = [Step("walk", [(x0, WALKWAY), (x0, -0.1)] + route), Step("face", heading=math.pi),
                  Step("take", posture="reach_mid", secs=1.6), Step("walk", route[::-1][1:] + [(x0, -0.1),
                                                                                             (x0, WALKWAY)]),
                  Step("walk", [(cage[0], WALKWAY + 0.15)]), Step("face", heading=math.pi / 2),
                  Step("put", posture="work", secs=1.6)]
        w.doing = "crossing the robots' floor to fetch an order"
        return f"{w.name} is crossing the floor"

    def jam(self, station: str = "packing") -> str:
        """The dock worker steps into a cell's guarding to clear a jammed carton: the curtain trips, the arm stops."""
        w = next((w for w in self.workers if w.role == "dock"), None)
        if w is None:
            return "no workers on this floor"
        c = self.world.cell.cells.get(station)
        if c is None or station not in APPROACH:
            return f"no way into the {station} cell"
        entry, inside = c.at(c.front - 0.07, 0.08 - c.side), c.at(c.front + 0.12, 0.1 - c.side)
        way = [*APPROACH[station], tuple(entry)]
        w.plan = [Step("walk", way + [tuple(inside)]), Step("face", heading=math.atan2(c.out[1], c.out[0])),
                  Step("hold", posture="work", secs=6.0), Step("walk", way[::-1])]
        w.doing = f"clearing a jam in the {station} cell"
        return f"{w.name} is clearing a jam at {station}"

    def _fetch(self, d: Drop):
        """A worker comes to carry a dropped carton away."""
        w = self.walker
        if w is None:
            return
        d.clearing = True
        x, y = d.thing.xy
        here = (w.x, w.y)
        to = np.array([x, y]) - 0.12 * (np.array([x, y]) - here) / max(1e-6, math.dist((x, y), here))
        back = [(hall.CROSSING, -0.1), (hall.CROSSING, WALKWAY)]
        w.plan = [Step("walk", [(hall.CROSSING, WALKWAY), (hall.CROSSING, -0.1), tuple(to)]),
                  Step("face", heading=math.atan2(y - to[1], x - to[0])),
                  Step("take", posture="reach_low", secs=1.8, then=lambda: self._park(d)), Step("walk", back)]
        w.doing = "clearing a dropped carton from the aisle"

    # things in the aisles

    def drop(self, kind: str = "pallet") -> str:
        """Something lands in an aisle, near a robot's path if one is driving, clear of cars, bays and people."""
        free = [d for d in self.drops if d.kind == kind and not d.placed]
        if not free:
            return f"no more {kind}s to drop; clear the aisle first"
        d = free[0]
        spot = self._spot(d)
        if spot is None:
            return "no clear spot to drop it"
        d.thing.xy, d.yaw = spot, float(self.rng.uniform(-0.4, 0.4))
        d.placed, d.t, d.clearing = True, self.world.t, False
        self._apply()
        return f"a {kind} dropped in the aisle at ({spot[0]:.2f}, {spot[1]:.2f})"

    def clear(self) -> str:
        n = sum(d.placed for d in self.drops)
        for d in self.drops:
            self._park(d)
        return f"cleared {n} thing{'s' * (n != 1)} from the aisles" if n else "the aisles are clear"

    def _park(self, d: Drop):
        d.placed, d.clearing, d.thing.xy = False, False, PARKED[:2]
        if self.world.m is not None:
            self._apply()

    def _spot(self, d: Drop) -> tuple[float, float] | None:
        site, cell = self.site, self.world.cell
        keep = [c.true_pose[:2] for c in self.world.cars]
        avoid = [(b.x, b.y) for b in (site.pick, *site.drops.values())]
        avoid += [s[:2] for b in (site.pick, *site.drops.values()) for s in b.wait] + [s[:2] for s in site.starts]
        avoid += [o.xy for o in self.world.objects if o.kind == "pallet" and o is not d.thing] + \
            [(w.x, w.y) for w in self.workers]
        paths = [np.asarray(p, float) for p in self.paths() if len(p) > 1]
        best, score = None, math.inf
        for x in np.arange(0.55, site.floor[0] - 0.45, 0.05):
            for y in np.arange(0.45, site.floor[1] - 0.4, 0.05):
                p = np.array([x, y])
                if any(np.linalg.norm(p - q) < 0.45 for q in keep) or any(np.linalg.norm(p - q) < 0.33 for q in avoid):
                    continue
                if any(_in_poly(p, poly, 0.12) for poly in cell.keepout()):
                    continue
                # on a robot's way if one is driving, else somewhere out of everyone's way
                s = min((_to_path(p, path) for path in paths), default=None)
                s = (s if s is not None else -min(np.linalg.norm(p - q) for q in avoid)) + 0.02 * self.rng.random()
                if s < score:
                    best, score = (float(x), float(y)), s
        return best

    # outputs

    def hazards(self) -> list[Hazard]:
        out = [Hazard(f"{w.name}", np.array([[w.x, w.y]]), True, "is crossing its path") for w in self.workers]
        out.append(Hazard("the forklift", self.forklift.footprint(), False, "is in its way"))
        out += [Hazard(f"a dropped {d.kind}", d.outline(), False, "is in its way") for d in self.drops if d.placed]
        return out

    def movers(self) -> list[dict]:
        out = [{"kind": "human", "id": w.id, "name": f"{w.name} · Unitree G1", "doing": w.doing,
                "pose": [round(w.x, 3), round(w.y, 3), round(w.heading, 3)]} for w in self.workers]
        f = self.forklift
        out.append({"kind": "forklift", "id": f.id, "name": f.name, "doing": f.doing,
                    "pose": [round(f.x, 3), round(f.y, 3), round(f.heading, 3)]})
        return out

    def label(self, body: str) -> tuple[str, str] | None:
        for w in self.workers:
            if body.startswith(w.prefix):
                return (f"{w.name} · Unitree G1", f"{w.doing.capitalize()}; MuJoCo Menagerie model at 1:4, kinematic "
                                                  "gait")
        if body.startswith("forklift"):
            f = self.forklift
            state = f"waiting for {f.waiting_for.name} to pass" if f.waiting_for else f.doing
            return f.name, f"{state.capitalize()}; counterbalance truck with a lidar, stops for people"
        if body.startswith(("stack_", "curtain_")):
            name = body.split("_")[1]
            title = name.upper() if len(name) <= 2 else name.capitalize()
            ld = self.world.cell.loaders[name]
            state = ("red: someone is inside the guarding, the arm has stopped" if self.tripped.get(name) else
                     "amber: the arm is working" if ld.busy else "green: idle")
            kind = "Stack light" if body.startswith("stack_") else "Light curtain"
            return f"{kind} · {title}", state
        return None


def _inside(c, xy, margin: float = 0.04) -> bool:
    rel = np.asarray(xy) - c.base
    out, along = rel @ np.array(c.out), rel @ np.array(c.along)
    return c.front - margin < out < c.rear and abs(along) < c.side


def _in_poly(p, poly: np.ndarray, pad: float) -> bool:
    import cv2

    return cv2.pointPolygonTest(poly.astype(np.float32), (float(p[0]), float(p[1])), True) > -pad


def _to_path(p, path: np.ndarray) -> float:
    a, b = path[:-1], path[1:]
    d = b - a
    t = np.clip(((p - a) * d).sum(axis=1) / np.maximum((d * d).sum(axis=1), 1e-9), 0, 1)
    return float(np.linalg.norm(a + t[:, None] * d - p, axis=1).min())

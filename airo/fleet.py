"""Fleet coordinator: robots on one site, each with its own mission and scientist. Routes commands ("airo 2 deliver
to packing", "select airo 2", body changes to the selected robot, "spill" to the place) and publishes the v3
dashboard state. Robots plan around each other's footprints and hold still rather than drive into one another."""

import asyncio
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from airo import geometry, planner
from airo.camera import Pose
from airo.life import LEVELS, Hazard
from airo.mission import Mission
from airo.server import topdown
from airo.site import Site, robot
from airo.voice import parse_intent

REACH = 0.38
WIDTH = 0.2
LANE = 0.12
HORIZON = 0.4
CONTACT = 0.03
PERSON = 0.07
PIVOT_PWM = 70
FOOTPRINT = geometry.CAR
WEIGHT_KG = 0.8
NUMBERS = {"one": 1, "two": 2, "to": 2, "too": 2, "three": 3, "four": 4}
_ROBOT = re.compile(r"^\s*(?:(select|switch to|follow|use)\s+)?(?:airo|aero|arrow|robot)[\s#-]*"
                    r"(\d+|one|two|to|too|three|four)\b[\s,:.!-]*(.*)$", re.IGNORECASE)
# "airo 1 …, airo 2 … and airo 3 …" is one command per robot; "and" alone stays inside a command
_SPLIT = re.compile(r"\s*(?:[,;]|\band\b)\s*(?=(?:airo|aero|arrow|robot)[\s#-]*(?:\d+|one|two|three|four)\b)",
                    re.IGNORECASE)
_ALL = re.compile(r"^\s*(?:all(?:\s+(?:robots|airos))?|everyone|every robot)[\s,:]+(.+)$", re.IGNORECASE)
NOW = {
    "idle": "Idle, waiting for a job.",
    "scanning": "Looking at the floor for its tote and the {station} station.",
    "delivering": "Driving behind its tote to push it to {station}.",
    "pushing": "Pushing its tote to {station}.",
    "surprised": "Stopped: its motion left the prediction cloud, so its body has changed.",
    "relearning": "Running probes to relearn its body before it carries on.",
    "learning": "Running probes to learn how its body moves.",
    "stopped": "Stopped on command.",
    "aborted": "Stopped by a safety check.",
}


@dataclass
class Robot:
    id: str
    name: str
    color: str
    rig: object
    mission: Mission
    sci: object
    tote: str
    tote_xy: tuple[float, float] | None = None
    real: bool = False
    guard: "Guard | None" = None


class Guard:
    """A robot's rig as its mission drives it: it holds still rather than drive into another car within REACH ahead
    (behind, when reversing), or into a hazard: a person in the same reach, or a thing (the forklift, a dropped
    pallet) just short of its bumper. Turning in place is always allowed."""

    def __init__(self, rig, others: Callable[[], list], hazards: Callable[[], list] = list,
                 guards: dict | None = None, rank: int = 0):
        self.rig, self._others, self._hazards = rig, others, hazards
        self.blocked_by = None
        self._seen: dict = {}
        self._guards, self.rank = guards if guards is not None else {}, rank
        self._guards[rig] = self

    def should_yield(self) -> bool:
        """Whether to back off after standing too long for another car. Replanning from where it stands only finds
        paths that hug the other car and stop again. A car standing still (loading, feeling its body) never moves
        out of the way, so give way to it; two cars stopped for each other: the higher-numbered one gives way; and
        always to a person."""
        other = next((g for rig, g in self._guards.items() if rig is self.blocked_by), None)
        if other is None:
            # a person heading for where this car stands waits for it as it waits for them
            return getattr(self.blocked_by, "person", False)
        return other.blocked_by is not self.rig or self.rank > other.rank

    def __getattr__(self, name):
        return getattr(self.rig, name)

    async def step(self, pwm_l: int, pwm_r: int):
        # a pivot from the inverse model is rarely exactly balanced: a small drive within a turn is still a turn
        drive = pwm_l + pwm_r
        self.blocked_by = self._ahead(math.copysign(1, drive)) if abs(drive) > 0.25 * abs(pwm_r - pwm_l) else None
        if self.blocked_by is not None:
            # Against a thing that does not move, keep the turn and drop the drive: an arc that would clip it
            # becomes a pivot, which swings the car away instead of leaving it nose to nose for good.
            turn = (pwm_r - pwm_l) / 2 if getattr(self.blocked_by, "person", True) is False else 0
            turn = math.copysign(max(abs(turn), PIVOT_PWM), turn) if turn else 0
            pwm_l, pwm_r = -int(turn), int(turn)
        return await self.rig.step(pwm_l, pwm_r)

    def _ahead_of(self, rig):
        """rig's pose HORIZON from now at its current speed and turn rate, from its last two poses."""
        p = rig.pose
        before, now = self._seen.get(rig, (None, None))
        if now is None or p.t != now.t:
            before, now = now, p
            self._seen[rig] = before, now
        if before is None or now.t <= before.t:
            return now
        k = HORIZON / (now.t - before.t)
        return Pose(now.x + k * (now.x - before.x), now.y + k * (now.y - before.y),
                    now.theta + k * math.remainder(now.theta - before.theta, math.tau), now.t)

    def people_near(self, r: float) -> bool:
        p = self.rig.pose
        return p is not None and any(h.person and np.hypot(*(h.points[0] - (p.x, p.y))) < r for h in self._hazards())

    def _ahead(self, direction: float):
        p = self.rig.pose
        if p is None:
            return None
        c, s = math.cos(p.theta), math.sin(p.theta)
        for other in self._others():
            if (q := other.pose) is None:
                continue
            dx, dy = q.x - p.x, q.y - p.y
            if 0 < direction * (dx * c + dy * s) < REACH and abs(-dx * s + dy * c) < WIDTH:
                return other
            # a car whose centre is off to the side can still have its nose in this one's lane
            rel = footprint(q) - (p.x, p.y)
            f, side = direction * (rel @ (c, s)), rel @ (-s, c)
            if ((f > 0) & (f < REACH) & (np.abs(side) < LANE)).any():
                return other
            # two cars turning into each other: where both will be once the motors have caught up with a stop
            if geometry.overlap(footprint(self._ahead_of(self.rig), CONTACT), footprint(self._ahead_of(other), CONTACT)):
                return other
        return next((h for h in self._hazards() if h.blocks(p.x, p.y, p.theta, direction)), None)


def footprint(pose, pad: float = 0.02) -> np.ndarray:
    hx, hy = FOOTPRINT[0] + pad, FOOTPRINT[1] + pad
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    return corners @ np.array([[c, s], [-s, c]]) + (pose.x, pose.y)


def parse_robot(text: str) -> tuple[int | None, bool, str]:
    """(robot index, select?, the rest): "airo 2 deliver to qa" -> (1, False, "deliver to qa")."""
    if not (m := _ROBOT.match(text)):
        return None, text.strip().lower().startswith("select"), text.strip()
    n = int(m[2]) if m[2].isdigit() else NUMBERS[m[2].lower()]
    return n - 1, bool(m[1]) or not m[3].strip(), m[3].strip()


def crew(site: Site, rigs: list, scientist: Callable, mode: str = "sim", body_path=None,
         real: bool = False) -> list[Robot]:
    """A robot per rig: on a shared floor each mission drives through a Guard and plans around the other cars and
    totes. scientist(rig, robot_id) builds a robot's scientist on the rig its mission drives."""
    robots: list[Robot] = []
    carry = site.mode == "carry"
    life = getattr(getattr(rigs[0], "world", None), "life", None)
    hazards = life.hazards if life is not None else list
    guards: dict = {}
    for k, rig in enumerate(rigs):
        guard = Guard(rig, lambda rig=rig: [r for r in rigs if r is not rig], hazards, guards, k) \
            if len(rigs) > 1 or life is not None else None
        driven = guard or rig
        sci = scientist(driven, robot(k)["id"])
        scan = _scanner(robots, k, carry, life) if len(rigs) > 1 or life is not None else None
        mission = Mission(driven, sci, scan=scan, mode=mode, body_path=body_path, stations=site.stations,
                          cell=rig.world.cell if carry else None, dock=site.starts[k] if carry else None)
        tote = site.totes[k % len(site.totes)] if site.totes else ""
        robots.append(Robot(**robot(k), rig=rig, mission=mission, sci=sci, tote=tote,
                            tote_xy=next((o.xy for o in site.objects() if o.concept == tote), None), real=real,
                            guard=guard))
    return robots


def _scanner(robots: list[Robot], k: int, carry: bool = False, life=None):
    async def scan() -> planner.Scene:
        me = robots[k]
        rig, frame = me.rig, me.rig.frame()
        scene = await planner.scan(frame, rig.H, rig.floor, stations=me.mission.stations,
                                   parcel=None if carry else planner.PARCEL, near=me.tote_xy)
        me.tote_xy = scene.parcel or me.tote_xy
        for other in robots:
            if other is not me and other.rig.pose is not None:
                planner.occupy(scene, footprint(other.rig.pose), "robot")
        for h in life.hazards() if life is not None else ():
            if h.person:
                x, y = h.points[0]
                planner.occupy(scene, np.array([[x - PERSON, y - PERSON], [x + PERSON, y - PERSON],
                                                [x + PERSON, y + PERSON], [x - PERSON, y + PERSON]]), "person")
            else:
                # the guard stops for the forklift and dropped goods: a plan through them only stops again
                planner.occupy(scene, cv2.convexHull(h.points.astype(np.float32))[:, 0], h.name)
        return scene
    return scan


class Fleet:
    """robots[selected] takes commands without a robot prefix and body changes; the world takes place changes."""

    def __init__(self, site: Site, robots: list[Robot], mode: str = "sim", dry: bool = False):
        self.site, self.robots, self.mode, self.dry = site, robots, mode, dry
        self.world = getattr(robots[0].rig, "world", None)
        self.life = getattr(self.world, "life", None)
        if self.life is not None:
            self.life.paths = lambda: [r.mission.path for r in self.robots if r.mission.busy and r.mission.path]
        self.selected = 0

    @property
    def robot(self) -> Robot:
        return self.robots[self.selected]

    def route(self, text: str) -> str:
        """Act on one command, or one per robot ("airo 1 …, airo 2 …", "all robots …"); returns what was done."""
        if len(parts := [p for p in _SPLIT.split(text) if p.strip()]) > 1:
            return "; ".join(self.route(p) for p in parts)
        if m := _ALL.match(text):
            return "; ".join(self.route(f"{r.name} {m[1]}") for r in self.robots)
        k, select, rest = parse_robot(text)
        if k is not None and not 0 <= k < len(self.robots):
            return f"no robot {k + 1}"
        if select:
            if k is None:
                return "select which robot?"
            self.selected = k
            return f"selected {self.robots[k].name}"
        target = self.robots[self.selected if k is None else k]
        if self.dry:
            return f"dry run, not driving: {text}"
        what = rest.lower().strip(" .!")
        try:
            if done := self._place(what) or self._body(target, what):
                return done
        except Exception as e:  # a bad change must not take the dashboard down with it
            return f"{what}: {e}"
        target.mission.inbox.put_nowait(rest)
        return f"{target.name}: {rest}"

    def _place(self, what: str) -> str | None:
        w = self.world
        if w is None:
            return None
        if what in ("spill", "oil spill", "slippery", "spill oil"):
            w.spill()
            return f"{self.site.spill_label.lower()} on the floor"
        if what in ("clean", "clean up", "mop", "clean the spill", "mop the spill"):
            w.clean()
            return "floor cleaned"
        if what in ("restock", "new totes", "reset totes"):
            if w.cell is not None:
                return "the conveyor brings totes by itself"
            w.restock()
            for r in self.robots:
                r.tote_xy = next(o.xy for o in w.objects if o.concept == r.tote)
            return "new totes in the pick zone"
        if self.life is not None and (done := self._life(what)):
            return done
        if what in ("pause", "resume"):
            w.paused = what == "pause"
            return what + "d"
        if what.startswith("speed "):
            try:
                w.set_speed(float(np.clip(float(what[6:]), 0.1, 4)))
            except ValueError:
                return None
            return what
        return None

    def _life(self, what: str) -> str | None:
        """The floor's life: "drop a pallet", "drop a carton", "clear the aisle", "send a worker across", "jam",
        and the busy level: "calm", "normal", "busy" (also "busy level calm")."""
        life = self.life
        level = what.removeprefix("busy level ").removeprefix("set busy ").strip()
        if level in LEVELS:
            life.set_level(level)
            return f"the floor is {level}"
        if m := re.fullmatch(r"(?:drop|dropped|spill)\s+(?:a\s+)?(pallet|carton|box)", what):
            return life.drop("carton" if m[1] == "box" else m[1])
        if what in ("clear the aisle", "clear the aisles", "clear the pallet", "clear the carton", "pick up the pallet",
                    "clear aisle", "clear"):
            return life.clear()
        if re.fullmatch(r"(?:send\s+)?(?:a\s+)?(?:worker|person|human|someone)\s+(?:across|crossing)|cross", what):
            return life.cross()
        if m := re.fullmatch(r"jam(?:\s+(?:at|in)\s+(?:the\s+)?(\w+))?", what):
            station = m[1] or "packing"
            return life.jam(station) if station in self.site.drops else f"no {station} cell"
        return None

    def _body(self, r: Robot, what: str) -> str | None:
        rig = r.rig
        if hasattr(rig, "add_weight"):
            change = {"weight": lambda: rig.add_weight(WEIGHT_KG), "add weight": lambda: rig.add_weight(WEIGHT_KG),
                      "remove weight": rig.remove_weight, "unweight": rig.remove_weight, "unload": rig.remove_weight,
                      "tow": rig.attach_tow, "untow": rig.detach_tow, "detach tow": rig.detach_tow,
                      "cut tow": rig.detach_tow, "flip": lambda: rig.flip_motor("l")}.get(what)
        elif hasattr(rig, "truth"):
            b = rig.truth
            change = {"weight": lambda: setattr(rig, "truth", replace(b, g_l=b.g_l * 0.65, g_r=b.g_r * 0.65,
                                                                        eta=b.eta * 0.7)),
                      "tow": lambda: setattr(rig, "truth", replace(b, g_l=b.g_l * 0.85, g_r=b.g_r * 0.85,
                                                                     lag=b.lag * 2))}.get(what)
        else:
            change = None
        if change is None:
            return None
        change()
        return f"{r.name}: {what}"

    # dashboard state, v3

    def state(self) -> dict:
        w, rig = self.world, self.robots[0].rig
        objects = w.objects if w is not None else getattr(rig, "objects", None)
        stations = self.site.station_xy(objects) if objects else \
            next((r.mission.scene.stations for r in self.robots if r.mission.scene), {})
        zones = [{"kind": "slippery", "polygon": _rect(w.zone), "label": self.site.spill_label, "found_by": None,
                  "t": w.zone_t}] if w is not None and w.zone else []
        floor = rig.floor if w is None else w.floor
        return {
            "t": w.t if w is not None else max((r.rig.pose.t for r in self.robots if r.rig.pose), default=0.0),
            "mode": self.mode, "dry": self.dry, "camera": _camera(rig) if self.mode == "real" else None,
            "site": {"name": self.site.name, "floor": {"w": floor[0], "h": floor[1]},
                     "stations": {k: [round(v, 3) for v in xy] for k, xy in stations.items()}, "zones": zones,
                     "movers": self.life.movers() if self.life is not None else [],
                     "busy": self.life.level if self.life is not None else None},
            "selected": self.robot.id,
            "robots": [{**r.mission.state(), "id": r.id, "name": r.name, "color": r.color, "real": r.real,
                        "now": self._now(r)} for r in self.robots],
        }

    def _now(self, r: Robot) -> str:
        m = r.mission
        if r.guard is not None and (b := r.guard.blocked_by) is not None and m.busy:
            if isinstance(b, Hazard):
                return f"Waiting: {b.name} {b.why}." if b.person else f"Stopped: {b.name} {b.why}; replanning."
            other = next((o.name for o in self.robots if o.rig is b), "another robot")
            return f"Waiting: {other} is in its way."
        if m.cell is not None or m.status in ("driving", "patrolling"):
            return m.now()
        intent = parse_intent(m.command, m.stations) if m.command else None
        station = intent.target if intent and intent.target else "its"
        return NOW.get(m.status, m.status.capitalize()).format(station=station)

    async def run(self, dash, hz: float = 10):
        """Route commands from the dashboard and terminal, publish the state and the top-down floor view."""
        while True:
            while not dash.commands.empty():
                print(self.route(dash.commands.get_nowait()))
            dash.publish(self.state())
            dash.portrait = self.robot.mission.portrait
            rig = self.robots[0].rig
            if (frame := rig.frame()) is not None and rig.H is not None:
                dash.set_frame(topdown(frame, rig.H, rig.floor))
                if not hasattr(rig, "view"):
                    dash.set_view(frame)
            await asyncio.sleep(1 / hz)


def _camera(rig) -> dict | None:
    """What the dashboard needs to draw over the raw phone frame and turn a click on it into a floor point."""
    frame = rig.frame()
    if rig.H is None or frame is None:
        return None
    return {"size": frame.shape[1::-1], "to_floor": rig.H, "to_image": np.linalg.inv(rig.H), "inset": rig.inset}


def _rect(r) -> list[list[float]]:
    x0, y0, x1, y1 = r
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

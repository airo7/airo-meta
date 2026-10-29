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

from airo import planner
from airo.mission import Mission
from airo.server import topdown
from airo.site import Site, robot
from airo.voice import parse_intent

REACH = 0.38
WIDTH = 0.2
FOOTPRINT = (0.13, 0.08)
WEIGHT_KG = 0.8
NUMBERS = {"one": 1, "two": 2, "to": 2, "too": 2, "three": 3, "four": 4}
_ROBOT = re.compile(r"^\s*(?:(select|switch to|follow|use)\s+)?(?:airo|aero|arrow|robot)[\s#-]*"
                    r"(\d+|one|two|to|too|three|four)\b[\s,:.!-]*(.*)$", re.IGNORECASE)
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
    (behind, when reversing). Turning in place is always allowed."""

    def __init__(self, rig, others: Callable[[], list]):
        self.rig, self._others = rig, others
        self.blocked_by = None

    def __getattr__(self, name):
        return getattr(self.rig, name)

    async def step(self, pwm_l: int, pwm_r: int):
        self.blocked_by = self._ahead(math.copysign(1, pwm_l + pwm_r)) if pwm_l + pwm_r else None
        if self.blocked_by is not None:
            pwm_l = pwm_r = 0
        return await self.rig.step(pwm_l, pwm_r)

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
        return None


def footprint(pose, pad: float = 0.02) -> np.ndarray:
    hx, hy = FOOTPRINT[0] + pad, FOOTPRINT[1] + pad
    c, s = math.cos(pose.theta), math.sin(pose.theta)
    corners = np.array([[hx, hy], [hx, -hy], [-hx, -hy], [-hx, hy]])
    return corners @ np.array([[c, s], [-s, c]]) + (pose.x, pose.y)


def _occupy(scene: planner.Scene, poly: np.ndarray, label: str):
    grid = scene.grid.astype(np.uint8)
    cv2.fillPoly(grid, [np.round(poly / scene.res - 0.5).astype(np.int32)], 1)
    scene.grid = grid > 0
    scene.polygons.append((label, poly))


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
    for k, rig in enumerate(rigs):
        guard = Guard(rig, lambda rig=rig: [r for r in rigs if r is not rig]) if len(rigs) > 1 else None
        driven = guard or rig
        sci = scientist(driven, robot(k)["id"])
        scan = _scanner(robots, k) if len(rigs) > 1 else None
        mission = Mission(driven, sci, scan=scan, mode=mode, body_path=body_path, stations=site.stations)
        tote = site.totes[k % len(site.totes)]
        robots.append(Robot(**robot(k), rig=rig, mission=mission, sci=sci, tote=tote,
                            tote_xy=next((o.xy for o in site.objects() if o.concept == tote), None), real=real,
                            guard=guard))
    return robots


def _scanner(robots: list[Robot], k: int):
    async def scan() -> planner.Scene:
        me = robots[k]
        rig, frame = me.rig, me.rig.frame()
        scene = await planner.scan(frame, rig.H, rig.floor, stations=me.mission.stations, parcel=planner.PARCEL,
                                   near=me.tote_xy)
        me.tote_xy = scene.parcel or me.tote_xy
        for other in robots:
            if other is not me and other.rig.pose is not None:
                _occupy(scene, footprint(other.rig.pose), "robot")
        return scene
    return scan


class Fleet:
    """robots[selected] takes commands without a robot prefix and body changes; the world takes place changes."""

    def __init__(self, site: Site, robots: list[Robot], mode: str = "sim", dry: bool = False):
        self.site, self.robots, self.mode, self.dry = site, robots, mode, dry
        self.world = getattr(robots[0].rig, "world", None)
        self.selected = 0

    @property
    def robot(self) -> Robot:
        return self.robots[self.selected]

    def route(self, text: str) -> str:
        """Act on one command; returns what was done, in a few words."""
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
            w.restock()
            for r in self.robots:
                r.tote_xy = next(o.xy for o in w.objects if o.concept == r.tote)
            return "new totes in the pick zone"
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
        w = self.world
        objects = w.objects if w is not None else getattr(self.robots[0].rig, "objects", None)
        stations = self.site.station_xy(objects) if objects else \
            next((r.mission.scene.stations for r in self.robots if r.mission.scene), {})
        zones = [{"kind": "slippery", "polygon": _rect(w.zone), "label": self.site.spill_label, "found_by": None,
                  "t": w.zone_t}] if w is not None and w.zone else []
        floor = self.site.floor if w is None else w.floor
        return {
            "t": w.t if w is not None else max((r.rig.pose.t for r in self.robots if r.rig.pose), default=0.0),
            "mode": self.mode,
            "site": {"name": self.site.name, "floor": {"w": floor[0], "h": floor[1]},
                     "stations": {k: [round(v, 3) for v in xy] for k, xy in stations.items()}, "zones": zones},
            "selected": self.robot.id,
            "robots": [{**r.mission.state(), "id": r.id, "name": r.name, "color": r.color, "real": r.real,
                        "now": self._now(r)} for r in self.robots],
        }

    def _now(self, r: Robot) -> str:
        m = r.mission
        if r.guard is not None and r.guard.blocked_by is not None and m.busy:
            other = next((o.name for o in self.robots if o.rig is r.guard.blocked_by), "another robot")
            return f"Waiting: {other} is in its way."
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


def _rect(r) -> list[list[float]]:
    x0, y0, x1, y1 = r
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]

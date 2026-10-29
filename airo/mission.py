"""Mission loop: commands in, deliveries out. Watches surprise while driving and relearns when reality leaves the
cloud, then replans from where the car stands."""

import asyncio
import math
import re
import time
import traceback
from collections import deque
from pathlib import Path
from contextlib import suppress

import numpy as np

from airo import planner, portrait
from airo.body import KEY, THRESHOLD, Body, cloud, save, surprise, wrap
from airo.control import V_MAX, face, follow
from airo.rig import DT, BaseRig, RigAbort
from airo.server import topdown
from airo.voice import parse_intent

WINDOW = 0.5
QUIET_S = 2.0
RESCAN_S = 2.0
DRIFT_M = 0.08
RUN_IN = 0.4
NEAR_PARCEL = 0.35
RUN_IN_WIDTH = 0.08
RUN_IN_HEADING = math.radians(30)
# The bumper's arm tips are 0.18 m from the centre and the parcel is 0.1 m square: a pivot closer than this can sweep
# it. No wider: the alignment pivot at the push pose (~0.27 m) is where a heavier body shows first.
CONTACT = 0.24
STANDOFF = 0.22
PORTRAITS = Path("data/portraits")
PROBE_REACH = 0.6
RELEARN_PROBES = 5
FAULT_Z = 5.0
CAUTION = 0.6
BLOCKED_S = 2.0
PERSON_S = 8.0
PEOPLE_CLEAR = 0.6
GIVE_WAY_S = 1.2
GIVE_WAY_PWM = 90
WAIT_S = 1.0
CLEAR = 0.3
BAY_RUN = 0.3
DOCK_RUN = 0.38
KINDS = ("heavy", "light")
PHASES = {"delivering": ("drive", "drive to the parcel"), "pushing": ("drive", "push to the station"),
          "fetching": ("drive", "drive to the pick cell"), "loading": ("drive", "loaded by the arm"),
          "unloading": ("drive", "unloaded by the arm"), "returning": ("drive", "back to the dock"),
          "surprised": ("surprise", "reality left the cloud"), "relearning": ("relearn", "relearn"),
          "learning": ("probe", "learn the body")}
PARAMS = [("g_l", "Gain left", "mm/s/PWM", 1000, 0, 5), ("g_r", "Gain right", "mm/s/PWM", 1000, 0, 5),
          ("dz", "Dead zone", "PWM", 1, 0, 80), ("eta", "Turn efficiency", "", 1, 0, 1),
          ("tau", "Delay", "ms", 1000, 0, 400), ("lag", "Wheel lag", "ms", 1000, 0, 400),
          ("kappa", "Drift", "rad/m", 1, -0.3, 0.3)]
GAUGE_UNITS = {"forward_cm": " cm", "lateral_cm": " cm", "turn_deg": "°"}


def _params(b: Body, key: bool = False) -> list[dict]:
    return [{"name": nm, "value": getattr(b, f) * k, "key": getattr(KEY, f) * k if key else None, "unit": u,
             "lo": lo, "hi": hi} for f, nm, u, k, lo, hi in PARAMS]


def _gauge(entry: dict) -> dict:
    """The scientist records outcomes per quantity; the dashboard gauge shows the one with the largest prediction."""
    p, m = entry.get("predicted"), entry.get("measured")
    if not (isinstance(p, dict) and isinstance(m, dict)):
        return entry
    q = max(p, key=lambda k: abs(p[k][0]))
    return {**entry, "predicted": round(p[q][0], 1), "sd": round(p[q][1], 1), "measured": round(m[q], 1),
            "unit": GAUGE_UNITS[q]}


class Surprised(Exception):
    pass


class Blocked(Exception):
    pass


class Stuck(Exception):
    """Another robot has stood in the way for a while: rescan and plan around it."""


class _Watched:
    """The rig as control sees it during a delivery: every step also feeds the surprise monitor."""

    def __init__(self, mission: "Mission"):
        self._m = mission

    def __getattr__(self, name):
        return getattr(self._m.rig, name)

    async def step(self, pwm_l: int, pwm_r: int):
        pose = await self._m.rig.step(pwm_l, pwm_r)
        self._m.watch()
        if self._m.cell is not None:
            self._m.moved()
        return pose


class Mission:
    """sci follows the scientist contract: best, members, n, name, notebook, learn(), relearn(seg) -> seconds, and
    optionally check_body(cause) -> {result, n, name, seconds}. With a cell (a carry site's Warehouse) deliveries
    fetch a tote from the pick cell instead of pushing a parcel, and end at dock."""

    def __init__(self, rig: BaseRig, sci, inbox: asyncio.Queue | None = None, scan=None, threshold=THRESHOLD,
                 watch_push: bool = False, mode: str = "real", body_path=None, portraits: Path | None = PORTRAITS,
                 stations: dict[str, str] = planner.STATIONS, cell=None, dock=None):
        self.rig, self.sci = rig, sci
        self.cell, self.dock = cell, dock
        self.kind: str | None = None
        self.loaded = False
        self.felt: dict | None = None
        self._line = None
        self._leaving = None
        self._blocked_t: float | None = None
        self.stations = stations
        self.watch_push = watch_push
        self.mode = mode
        self.body_path = body_path
        self.portraits = portraits
        self.portrait: bytes | None = None
        self.portrait_n = 0
        self._drawing: asyncio.Task | None = None
        self.phases: list[dict] = []
        self._relearn_t: float | None = None
        self._relearn_end: float | None = None
        self.inbox = inbox or asyncio.Queue()
        parcel = None if cell is not None else planner.PARCEL
        self.scan = scan or (lambda: planner.scan(rig.frame(), rig.H, rig.floor, stations=stations, parcel=parcel))
        self.threshold = threshold
        self._status, self.command, self.station = "idle", "", ""
        self.scene: planner.Scene | None = None
        self.path: list = []
        self.notes: list[dict] = []
        self.t0 = time.time()
        self.relearned_s: float | None = None
        self.probing_s: float | None = None
        self._relearn_rig0 = 0.0
        self.values: deque[float] = deque(maxlen=120)
        self._recent: deque[float | None] = deque(maxlen=3)
        self._next_window = -math.inf
        self._quiet_until = -math.inf
        self._near: tuple[float, float, float] | None = None
        self._task: asyncio.Task | None = None

    @property
    def status(self) -> str:
        return self._status

    @status.setter
    def status(self, value: str):
        if value != self._status and value in PHASES:
            kind, label = PHASES[value]
            self.phases.append({"t": time.time() - self.t0, "kind": kind, "label": label})
        self._status = value

    @property
    def body(self) -> Body:
        return self.sci.best

    @property
    def v_max(self) -> float:
        """Slower after a relearn the scientist could not confirm."""
        return V_MAX * (1 if getattr(self.sci, "verified", True) else CAUTION)

    def note(self, kind: str, text: str):
        self.notes.append({"t": time.time(), "kind": kind, "text": text})

    def watch(self):
        """Score the last window; raise Surprised when the median of three unsuppressed windows passes threshold."""
        now = self.rig.pose.t
        if now < self._next_window:
            return
        self._next_window = now + WINDOW
        seg = self.rig.segment(now - WINDOW - DT / 2)
        cmd = np.abs(seg.cmd[seg.cmd_t >= seg.t[0]])
        p = self.rig.pose
        quiet = (now < self._quiet_until or not len(cmd) or cmd.max(axis=1).min() <= self.body.dz
                 or self._running_in(p) or self._touching(p))
        s = [] if quiet else surprise(self.sci.members, seg, WINDOW)
        score = None if not len(s) or np.isnan(s[-1]) else float(s[-1])
        self.values.append(score or 0.0)
        self._recent.append(score)
        if len(self._recent) == 3 and None not in self._recent and (m := np.median(self._recent)) > self.threshold:
            self._recent.clear()
            raise Surprised(f"surprise {m:.1f}")

    def _running_in(self, p) -> bool:
        """On the last straight stretch into the push pose, where parcel contact would read as surprise. Turns near
        the parcel stay watched: that is where a heavier body shows."""
        if self._near is None:
            return False
        x, y, th = self._near
        along = (p.x - x) * math.cos(th) + (p.y - y) * math.sin(th)
        across = -(p.x - x) * math.sin(th) + (p.y - y) * math.cos(th)
        return (-NEAR_PARCEL <= along <= 0.05 and abs(across) < RUN_IN_WIDTH
                and abs(wrap(p.theta - th)) < RUN_IN_HEADING)

    def _touching(self, p) -> bool:
        parcel = self.scene and self.scene.parcel
        return parcel is not None and math.dist((p.x, p.y), parcel) < CONTACT

    async def deliver(self, station: str, attempts: int = 5) -> bool:
        self.station = station
        if self.cell is not None:
            return await self._fetch(station)
        for _ in range(attempts):
            self.status = "scanning"
            self.scene = await self.scan()
            if station not in self.scene.stations or self.scene.parcel is None:
                self.note("alert", f"Can't see the parcel or the {station} station.")
                return False
            try:
                if await self._approach(station) and await self._push(station):
                    self.note("result", f"Parcel delivered to {station}.")
                    return True
            except Surprised as e:
                await self.relearn(str(e))
            except Blocked as e:
                self.note("alert", str(e))
                return False
            finally:
                self._near = None
        self.note("alert", f"Gave up on {station} after {attempts} attempts.")
        return False

    def _plan(self, goal, **kw) -> list:
        p = self.rig.pose
        path = planner.plan(self.scene, (p.x, p.y), goal, **kw)
        if path is None or not all(self.rig.inside(*q) for q in path[1:]):
            raise Blocked(f"No safe path to ({goal[0]:.2f}, {goal[1]:.2f}).")
        self.path = path
        return path

    async def _approach(self, station: str) -> bool:
        """Drive to a run-in point behind the parcel, shortened near the inset, then straight to the push pose."""
        x, y, th = planner.push_pose(self.scene.parcel, self.scene.stations[station], STANDOFF)
        if not self.rig.inside(x, y):
            raise Blocked(f"The push pose ({x:.2f}, {y:.2f}) is outside the floor inset.")
        run_in = next(((x - r * math.cos(th), y - r * math.sin(th)) for r in np.linspace(RUN_IN, 0.1, 7)
                       if self.rig.inside(x - r * math.cos(th), y - r * math.sin(th))), (x, y))
        self.status = "delivering"
        self._near = (x, y, th)
        self.path = self._plan(run_in) + [(x, y)]
        await follow(_Watched(self), self.body, self.path, v_max=self.v_max)
        return True

    async def _push(self, station: str) -> bool:
        target = np.array(self.scene.stations[station])
        d = target - self.scene.parcel
        goal = target - STANDOFF * d / np.linalg.norm(d)
        self._plan(tuple(goal), ignore=("parcel",))
        self.status = "pushing"
        rig = _Watched(self) if self.watch_push else self.rig
        drive = asyncio.create_task(follow(rig, self.body, self.path, v_max=self.v_max))
        try:
            while not (await asyncio.wait({drive}, timeout=RESCAN_S))[0]:
                self.scene = await self.scan()
                p = self.rig.pose
                front = (p.x + STANDOFF * math.cos(p.theta), p.y + STANDOFF * math.sin(p.theta))
                if self.scene.parcel is None or math.dist(front, self.scene.parcel) > DRIFT_M:
                    self.note("alert", "The parcel slipped off the bumper. Going around again.")
                    return False
            await drive
        finally:
            drive.cancel()
            with suppress(asyncio.CancelledError):
                await drive
        self.scene = await self.scan()
        return self.scene.parcel is not None and math.dist(self.scene.parcel, target) < 0.1

    # carrying: fetch a tote from the pick cell, have it loaded, take it to a station, have it unloaded

    async def _fetch(self, station: str) -> bool:
        """Queue for the pick bay, get loaded, feel the new body, drive to the station's drop bay, get unloaded, feel
        the body again, go back to the dock. A tote already on the deck is delivered without fetching another."""
        cell = self.cell
        self.kind = self.kind or cell.site.kinds.get(station, "light")
        pick, drop = cell.lines["pick"], cell.lines[station]
        tote = None
        try:
            if (on := cell.on(self.rig)) is not None:
                self.kind = on.kind
            else:
                tote = cell.order(self.kind)
                await self._bay(pick, "fetching")
                while not cell.ready(tote):
                    await self.rig.step(0, 0)
                self.status = "loading"
                await cell.pick.load(self.rig, tote)
                self.loaded = True
                await self._feel(f"loaded a {self.kind} tote")
                self._leaving = pick
            self.loaded = True
            await self._bay(drop, "delivering")
            self.status = "unloading"
            await cell.loaders[station].unload(self.rig, cell.slot(station))
            self.loaded = False
            await self._feel("unloaded")
            self.note("result", f"Delivered a {self.kind} tote to {_title(station)}.")
            if self.dock is not None:
                self._leaving = drop
                self.status = "returning"
                x, y, th = self.dock
                await self._go(self.dock, via=(x + DOCK_RUN * math.cos(th), y + DOCK_RUN * math.sin(th)))
            return True
        except (Blocked, ValueError, StopIteration) as e:
            self.note("alert", str(e) or "No tote on the conveyor.")
            return False
        finally:
            if tote is not None and tote.state == "belt":
                tote.claimed = False
            for line in (pick, drop):
                line.leave(self)
            self._line = self._leaving = None

    async def _bay(self, line, status: str):
        """Join the line for a bay; drive up the queue spots as the robots ahead leave, and into the bay."""
        line.join(self)
        self._line = line
        while True:
            k, spot = line.spot(self)
            self.status = status if k == 0 else "queued"
            while not line.clear(self):
                await self.rig.step(0, 0)
                self._release()
            k, spot = line.spot(self)
            self.status = status if k == 0 else "queued"
            if spot is not None:
                x, y, th = spot
                behind = (x - BAY_RUN * math.cos(th), y - BAY_RUN * math.sin(th))
                await self._go(spot, via=behind if k == 0 else None)
            line.arrive(self, k)
            if k == 0:
                return
            while line.spot(self)[0] == k:
                await self.rig.step(0, 0)
                self._release()

    async def _go(self, pose, via=None, attempts: int = 20):
        """Drive to pose (x, y, heading) around the loader cells and the other robots, through via if given and
        not already passed; relearn on surprise; when another robot stays in the way or blocks every path, wait a
        moment and plan again."""
        p = self.rig.pose
        if math.dist(pose[:2], (p.x, p.y)) < 0.05:
            await face(_Watched(self), self.body, pose[2])
            return
        for _ in range(attempts):
            self.scene = await self.scan()
            for poly in self.cell.keepout():
                planner.occupy(self.scene, poly, "loader cell")
            p = self.rig.pose
            try:
                if via is None or math.dist(via, (p.x, p.y)) < 0.1 or math.dist(via, pose[:2]) > \
                        math.dist((p.x, p.y), pose[:2]):
                    self._plan(pose[:2])
                else:
                    self.path = self._plan(via) + [tuple(pose[:2])]
                await follow(_Watched(self), self.body, self.path, v_max=self.v_max)
                await face(_Watched(self), self.body, pose[2])
                return
            except Surprised as e:
                await self.relearn(str(e))
            except (Stuck, Blocked):
                if getattr(self.rig, "should_yield", lambda: False)():
                    await self._give_way()
                for _ in range(round(WAIT_S / DT)):
                    await self.rig.step(0, 0)
        raise Blocked(f"Could not get to ({pose[0]:.2f}, {pose[1]:.2f}).")

    async def _give_way(self):
        """Back off a little, straight, so the robot this one is deadlocked with has room to pass; the guard still
        stops it for anything behind."""
        for _ in range(round(GIVE_WAY_S / DT)):
            await self.rig.step(-GIVE_WAY_PWM, -GIVE_WAY_PWM)
        await self.rig.stop()

    def moved(self):
        """After every watched step on a carry site: give up the bay just left once clear of it, and notice a robot
        that has been in the way too long."""
        self._release()
        blocker = getattr(self.rig, "blocked_by", None)
        if blocker is None:
            self._blocked_t = None
        elif self._blocked_t is None:
            self._blocked_t = self.rig.pose.t
        elif self.rig.pose.t - self._blocked_t > (PERSON_S if getattr(blocker, "person", False) else BLOCKED_S):
            # people walk on, so wait longer for them before planning around
            self._blocked_t = None
            raise Stuck()

    def _release(self):
        if self._leaving is not None:
            b, p = self._leaving.bay, self.rig.pose
            if math.dist((b.x, b.y), (p.x, p.y)) > CLEAR:
                self._leaving.leave(self)
                self._leaving = None

    async def _feel(self, cause: str):
        """Ask the scientist whether the body changed with the load, while parked at the cell. Skipped by
        scientists without check_body: the surprise monitor then finds a changed body on the way."""
        check = getattr(self.sci, "check_body", None)
        if check is None:
            return
        self.status = "feeling"
        near = getattr(self.rig, "people_near", None)
        while near is not None and near(PEOPLE_CLEAR):
            # the probe pivots and drives a little: not with someone standing next to the car
            await self.rig.step(0, 0)
        r = await check(cause)
        self.felt = {**r, "cause": cause, "at": time.time()}
        self._recent.clear()
        self._quiet_until = self.rig.pose.t + QUIET_S
        self.note("conclusion", f"{cause[0].upper()}{cause[1:]}: {_felt(r)}.")

    def _carry_now(self) -> str | None:
        s, kind = self.status, self.kind or "light"
        to = _title(self.station)
        if s == "fetching":
            return f"Driving to the pick cell for a {kind} tote for {to}."
        if s == "queued" and self._line is not None:
            k = self._line.spot(self)[0] if self in self._line.robots else 0
            if self._line.name == "pick":
                return f"Waiting in line for the pick cell arm ({_nth(k)} in line)."
            return f"Waiting for the {_title(self._line.name)} drop bay to clear."
        if s == "loading":
            return f"Waiting for the arm to load a {kind} tote."
        if s == "feeling":
            return (f"Feeling its new body with a {kind} tote on: a pivot or two." if self.loaded else
                    "Feeling its body again now the tote is off: a pivot or two.")
        if s == "delivering":
            f = self.felt
            if f and time.time() - f["at"] < 6 and f["result"] != "same":
                return f"{_felt(f)[0].upper()}{_felt(f)[1:]}. Driving to {to} with a {kind} tote."
            why = "" if getattr(self.sci, "verified", True) else f", slowly: Body #{self.sci.n} is not confirmed"
            return f"Driving to {to} with a {kind} tote{why}."
        if s == "unloading":
            return f"Unloading at {to}."
        if s == "returning":
            return "Heading back to its charging dock."
        if s == "idle":
            return f"Parked at its charging dock as Body #{self.sci.n} ({self.sci.name}), waiting for a job."
        return None

    async def relearn(self, why: str = ""):
        """Probes drive blind, so first leave the parcel's reach for the probe area if recentring is available."""
        await self.rig.stop()
        self._relearn_t, self._relearn_end, t0 = time.time(), None, self.rig.pose.t
        self._relearn_rig0 = t0
        self.status = "surprised"
        seg = self.rig.segment(self.rig.pose.t - 3 * WINDOW)
        self.status = "relearning"
        p, parcel = self.rig.pose, self.scene and self.scene.parcel
        recentre = getattr(self.sci, "recentre", None)
        if recentre and parcel and math.dist((p.x, p.y), parcel) < PROBE_REACH:
            self.note("alert", f"Reality left the cloud ({why}). Moving clear of the parcel to probe.")
            centre = (self.rig.floor[0] / 2, self.rig.floor[1] / 2)
            if path := planner.plan(self.scene, (p.x, p.y), centre):
                await follow(self.rig, self.body, path, v_max=CAUTION * V_MAX)
            await recentre(self.rig)
        t_probe = self.rig.pose.t
        await self.sci.relearn(seg, RELEARN_PROBES)
        self.relearned_s, self.probing_s = self.rig.pose.t - t0, self.rig.pose.t - t_probe
        self._relearn_end = time.time()
        z = [abs(v) for e in self.sci.notebook if e["t"] >= self._relearn_t and e["kind"] == "result"
             for v in e["z"].values()]
        if not getattr(self.sci, "verified", True) and z and max(z) > FAULT_Z:
            await self.rig.stop()
            self.note("alert", f"Hardware fault: no body I can fit explains these probes (|z| up to {max(z):.0f}). "
                               "Stopping; check the motors and wiring.")
            raise RigAbort("hardware fault")
        self._quiet_until = self.rig.pose.t + QUIET_S
        how = "recalled" if any(e["kind"] == "recall" for e in self.sci.notebook if e["t"] >= self._relearn_t) \
            else "relearned"
        self.note("conclusion", f"Body #{self.sci.n}, {how} in {self.relearned_s:.1f} s "
                                f"({self.probing_s:.1f} s probing).")
        known = self._memory and self._memory.get(self.sci.n)
        if how == "recalled" and known and known.portrait and Path(known.portrait).exists():
            self.portrait, self.portrait_n = Path(known.portrait).read_bytes(), self.portrait_n + 1
        elif getattr(self.sci, "verified", True):
            self._draw()

    async def learn(self):
        self.status = "learning"
        await self.sci.learn()
        self._draw()
        if self.body_path:
            save(self.sci.best, self.sci.members, self.sci.name, self.body_path)

    async def handle(self, text: str):
        intent = parse_intent(text, self.stations)
        if intent is None:
            self.note("alert", f'Not understood: "{text}"')
            return
        self.command = text
        if intent.kind == "stop":
            await self._cancel()
            await self.rig.stop()
            self.status = "stopped"
        elif intent.kind == "status":
            b = self.body
            self.note("result", f"Body #{self.sci.n} {self.sci.name}: gains {b.g_l * 1000:.2f}/{b.g_r * 1000:.2f} "
                                f"mm/s/PWM, dead zone {b.dz:.0f}, delay {b.tau * 1000:.0f} ms.")
        elif intent.kind == "learn" or intent.target in self.stations:
            await self._cancel()
            self.kind = next((k for k in KINDS if re.search(rf"\b{k}\b", text.lower())), None)
            work = self.learn() if intent.kind == "learn" else self.deliver(intent.target)
            self._task = asyncio.create_task(self._run(work))
        else:
            self.note("alert", f"Deliver where? Stations: {', '.join(self.stations)}.")

    async def _run(self, work):
        try:
            await work
            self.status = "idle"
        except RigAbort as e:
            self.status = "aborted"
            self.note("alert", f"Safety stop: {e}.")
        except Exception as e:  # otherwise the task ends silently and the robot keeps a stale status
            traceback.print_exc()
            self.status = "error"
            self.note("alert", f"Stopped on an internal error: {type(e).__name__}: {e}.")
            with suppress(Exception):
                await self.rig.stop()

    async def _cancel(self):
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task

    @property
    def busy(self) -> bool:
        return self._task is not None and not self._task.done()

    def state(self) -> dict:
        p = self.rig.pose
        pose = None if p is None else {"x": p.x, "y": p.y, "theta": p.theta}
        ghost = []
        if p is not None and self.busy:
            cmd_t, cmd = self.rig.history
            t = p.t + np.linspace(0, 1.5, 16)
            ghost = cloud(self.sci.members, t, cmd_t, cmd, (p.x, p.y, p.theta))[..., :2].round(3)
        b = self.body
        polygons = [{"label": lb, "points": pts.round(3)} for lb, pts in self.scene.polygons] if self.scene else []
        return {
            "t": p.t if p else 0.0, "status": self.status, "command": self.command,
            "floor": {"w": self.rig.floor[0], "h": self.rig.floor[1]},
            "pose": pose, "cloud": ghost, "path": self.path, "polygons": polygons,
            "body": {"n": self.sci.n, "name": self.sci.name, "relearned_s": self.relearned_s,
                     "probing_s": self.probing_s, "portrait": self.portrait_n or None,
                     "params": _params(b, self.sci.n <= 1)},
            "surprise": {"values": list(self.values), "threshold": self.threshold},
            "notebook": [_gauge({**n, "t": n["t"] - self.t0})
                         for n in sorted(self.notes + list(self.sci.notebook), key=lambda n: n["t"])],
            "mode": self.mode, "episode": {"phases": self._timeline()}, "relearn": self._relearn_state(),
            "memory": self._memory_state(), "now": self.now(),
        }

    def _memory_state(self) -> list[dict]:
        mem = self._memory
        return [{"n": e.n, "name": e.name, "cause": e.cause, "params": _params(e.body),
                 "portrait": f"/portraits/{Path(e.portrait).name}" if e.portrait else None,
                 "learned_t": e.learned_t - self.t0, "recalls": e.recalls, "active": e.n == self.sci.n,
                 "from": e.robot, "verified": mem.robot in e.tested} for e in (mem.entries if mem else [])]

    def now(self) -> str:
        """One plain sentence on what the car is doing and why, for the dashboard."""
        sci, s = self.sci, self.status
        if self.cell is not None and (line := self._carry_now()):
            return line
        to = f"the {self.station} station" if self.station else "the station"
        me = f"Body #{sci.n} ({sci.name})"
        if s == "relearning" and (e := getattr(sci, "recalling", None)):
            return f"This feels like Body #{e.n} ({e.name}) again: one probe to confirm before relearning."
        if s == "relearning":
            r = self._relearn_state()
            step = f" ({r['step']} of up to {r['steps']} probes run)" if r and r["step"] else ""
            return f"My body changed while driving, so I am testing what changed{step}."
        if s == "delivering":
            why = "" if getattr(sci, "verified", True) else f", slowly because {me} is not confirmed"
            return f"Driving behind the parcel to push it to {to}{why}."
        return {"idle": f"Waiting for a command as {me}.", "scanning": f"Looking for the parcel and {to}.",
                "pushing": f"Pushing the parcel to {to}.", "surprised": "I moved differently than predicted: stopping.",
                "learning": "Learning my body from scratch with probe drives.", "stopped": "Stopped on command.",
                "aborted": "Stopped for safety; waiting for a command."}.get(s, f"Stopped: {s}.")

    def _draw(self):
        """Self-portrait of the confirmed body, in the background so driving never waits for it."""
        seen = next((e["seen"] for e in reversed(self.sci.notebook) if e.get("seen")), None)
        self._drawing = asyncio.create_task(self._portrait(self.sci.n, self.sci.name, seen))

    async def _portrait(self, n: int, name: str, seen: str | None):
        if (png := await portrait.draw(name, seen)) is None:
            return
        self.portrait, self.portrait_n = png, self.portrait_n + 1
        if self.portraits:
            self.portraits.mkdir(parents=True, exist_ok=True)
            (path := self.portraits / f"body_{n}.png").write_bytes(png)
            if e := self._memory and self._memory.get(n):
                e.portrait = str(path)
                self._memory.save()

    @property
    def _memory(self):
        return getattr(self.sci, "memory", None)

    def _timeline(self) -> list[dict]:
        """Mission phases plus each probe and refit from the scientist's notebook, in time order."""
        probes = [{"t": e["t"] - self.t0, "kind": "probe", "label": e["probe"]}
                  for e in self.sci.notebook if e["kind"] == "hypothesis" and "probe" in e]
        refits = [{"t": e["t"] - self.t0, "kind": "refit", "label": e.get("name", "refit")}
                  for e in self.sci.notebook if e["kind"] in ("conclusion", "recall")]
        return sorted(self.phases + probes + refits, key=lambda p: p["t"])

    def _relearn_state(self) -> dict | None:
        """The relearn overlay, held for a few seconds after it finishes."""
        if self._relearn_t is None or (self._relearn_end and time.time() - self._relearn_end > 4):
            return None
        ran = [e for e in self.sci.notebook if e["t"] >= self._relearn_t and e["kind"] == "result"]
        return {"phase": "done" if self._relearn_end else "testing" if ran else "changed",
                "elapsed_s": self.relearned_s if self._relearn_end else self.rig.pose.t - self._relearn_rig0,
                "probe": ran[-1].get("probe", "") if ran else "", "step": len(ran), "steps": RELEARN_PROBES}

    async def _next_event(self, timeout: float | None):
        """Wait until the job ends or a command comes in. Never on the wall clock alone, unless publishing: in
        lockstep simulation a car whose loop wakes on wall time joins the world's schedule at a moment set by the
        host's load, which made fleet runs differ from one run to the next."""
        get = asyncio.ensure_future(self.inbox.get())
        try:
            done, _ = await asyncio.wait({self._task, get}, timeout=timeout, return_when=asyncio.FIRST_COMPLETED)
        finally:
            if not get.done():
                get.cancel()
        if get in done:
            await self.handle(get.result())

    async def run(self, dash=None):
        """Consume commands, keep the rig ticking while idle, publish at ~10 Hz. "stop" always wins."""
        last = -math.inf
        while True:
            while not self.inbox.empty():
                await self.handle(self.inbox.get_nowait())
            if self.busy:
                await self._next_event(timeout=DT if dash else None)
            else:
                try:
                    await self.rig.step(0, 0)
                except RigAbort as e:
                    self.status = f"out of bounds: {e}"
            if dash and self.rig.pose and self.rig.pose.t - last >= 0.1:
                last = self.rig.pose.t
                dash.publish(self.state())
                dash.portrait = self.portrait
                if (frame := self.rig.frame()) is not None and self.rig.H is not None:
                    dash.set_frame(topdown(frame, self.rig.H, self.rig.floor))
                    if not hasattr(self.rig, "view"):
                        dash.set_view(frame)


def _title(station: str) -> str:
    return station.upper() if len(station) <= 2 else station.capitalize()


def _nth(k: int) -> str:
    return {1: "1st", 2: "2nd", 3: "3rd"}.get(k, f"{k}th")


def _felt(r: dict) -> str:
    """check_body's answer in a few words: "recalled Body #2 · heavy tote · 1.1 s"."""
    what = f"Body #{r['n']} · {r['name']}"
    return {"same": f"same body, {what}", "recalled": f"recalled {what} · {r['seconds']:.1f} s",
            "learned": f"learned {what} in {r['seconds']:.1f} s"}.get(r["result"], what)

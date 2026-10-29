"""Mission loop: commands in, deliveries out. Watches surprise while driving and relearns when reality leaves the
cloud, then replans from where the car stands."""

import asyncio
import math
import time
from collections import deque
from contextlib import suppress

import numpy as np

from airo import planner
from airo.body import KEY, Body, cloud, save, surprise, wrap
from airo.control import V_MAX, follow
from airo.rig import DT, BaseRig, RigAbort
from airo.server import topdown
from airo.voice import parse_intent

THRESHOLD = 3.0
WINDOW = 0.5
QUIET_S = 2.0
RESCAN_S = 2.0
DRIFT_M = 0.08
RUN_IN = 0.4
NEAR_PARCEL = 0.35
RUN_IN_WIDTH = 0.08
RUN_IN_HEADING = math.radians(30)
STANDOFF = 0.22
PROBE_REACH = 0.6
RELEARN_PROBES = 5
FAULT_Z = 5.0
CAUTION = 0.6
PHASES = {"delivering": ("drive", "drive to the parcel"), "pushing": ("drive", "push to the station"),
          "surprised": ("surprise", "reality left the cloud"), "relearning": ("relearn", "relearn"),
          "learning": ("probe", "learn the body")}
PARAMS = [("g_l", "Gain left", "mm/s/PWM", 1000, 0, 5), ("g_r", "Gain right", "mm/s/PWM", 1000, 0, 5),
          ("dz", "Dead zone", "PWM", 1, 0, 80), ("eta", "Turn efficiency", "", 1, 0, 1),
          ("tau", "Delay", "ms", 1000, 0, 400), ("lag", "Wheel lag", "ms", 1000, 0, 400),
          ("kappa", "Drift", "rad/m", 1, -0.3, 0.3)]
GAUGE_UNITS = {"forward_cm": " cm", "lateral_cm": " cm", "turn_deg": "°"}


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


class _Watched:
    """The rig as control sees it during a delivery: every step also feeds the surprise monitor."""

    def __init__(self, mission: "Mission"):
        self._m = mission

    def __getattr__(self, name):
        return getattr(self._m.rig, name)

    async def step(self, pwm_l: int, pwm_r: int):
        pose = await self._m.rig.step(pwm_l, pwm_r)
        self._m.watch()
        return pose


class Mission:
    """sci follows the scientist contract: best, members, n, name, notebook, learn(), relearn(seg) -> seconds."""

    def __init__(self, rig: BaseRig, sci, inbox: asyncio.Queue | None = None, scan=None, threshold=THRESHOLD,
                 watch_push: bool = False, mode: str = "real", body_path=None):
        self.rig, self.sci = rig, sci
        self.watch_push = watch_push
        self.mode = mode
        self.body_path = body_path
        self.phases: list[dict] = []
        self._relearn_t: float | None = None
        self._relearn_end: float | None = None
        self.inbox = inbox or asyncio.Queue()
        self.scan = scan or (lambda: planner.scan(rig.frame(), rig.H, rig.floor))
        self.threshold = threshold
        self._status, self.command = "idle", ""
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
                 or self._running_in(p))
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

    async def deliver(self, station: str, attempts: int = 5) -> bool:
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
        self.note("conclusion", f"Body #{self.sci.n}, relearned in {self.relearned_s:.1f} s "
                                f"({self.probing_s:.1f} s probing).")

    async def learn(self):
        self.status = "learning"
        await self.sci.learn()
        if self.body_path:
            save(self.sci.best, self.sci.members, self.sci.name, self.body_path)

    async def handle(self, text: str):
        intent = parse_intent(text)
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
        elif intent.kind == "learn" or intent.target in planner.STATIONS:
            await self._cancel()
            work = self.learn() if intent.kind == "learn" else self.deliver(intent.target)
            self._task = asyncio.create_task(self._run(work))
        else:
            self.note("alert", f"Deliver where? Stations: {', '.join(planner.STATIONS)}.")

    async def _run(self, work):
        try:
            await work
            self.status = "idle"
        except RigAbort as e:
            self.status = "aborted"
            self.note("alert", f"Safety stop: {e}.")

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
        b, first = self.body, self.sci.n <= 1
        polygons = [{"label": lb, "points": pts.round(3)} for lb, pts in self.scene.polygons] if self.scene else []
        return {
            "t": p.t if p else 0.0, "status": self.status, "command": self.command,
            "floor": {"w": self.rig.floor[0], "h": self.rig.floor[1]},
            "pose": pose, "cloud": ghost, "path": self.path, "polygons": polygons,
            "body": {"n": self.sci.n, "name": self.sci.name, "relearned_s": self.relearned_s,
                     "probing_s": self.probing_s,
                     "params": [{"name": nm, "value": getattr(b, f) * k, "key": getattr(KEY, f) * k if first else None,
                                 "unit": u,
                                 "lo": lo, "hi": hi} for f, nm, u, k, lo, hi in PARAMS]},
            "surprise": {"values": list(self.values), "threshold": self.threshold},
            "notebook": [_gauge({**n, "t": n["t"] - self.t0})
                         for n in sorted(self.notes + list(self.sci.notebook), key=lambda n: n["t"])],
            "mode": self.mode, "episode": {"phases": self._timeline()}, "relearn": self._relearn_state(),
        }

    def _timeline(self) -> list[dict]:
        """Mission phases plus each probe and refit from the scientist's notebook, in time order."""
        probes = [{"t": e["t"] - self.t0, "kind": "probe", "label": e["probe"]}
                  for e in self.sci.notebook if e["kind"] == "hypothesis" and "probe" in e]
        refits = [{"t": e["t"] - self.t0, "kind": "refit", "label": e.get("name", "refit")}
                  for e in self.sci.notebook if e["kind"] == "conclusion"]
        return sorted(self.phases + probes + refits, key=lambda p: p["t"])

    def _relearn_state(self) -> dict | None:
        """The relearn overlay, held for a few seconds after it finishes."""
        if self._relearn_t is None or (self._relearn_end and time.time() - self._relearn_end > 4):
            return None
        ran = [e for e in self.sci.notebook if e["t"] >= self._relearn_t and e["kind"] == "result"]
        return {"phase": "done" if self._relearn_end else "testing" if ran else "changed",
                "elapsed_s": self.relearned_s if self._relearn_end else self.rig.pose.t - self._relearn_rig0,
                "probe": ran[-1].get("probe", "") if ran else "", "step": len(ran), "steps": RELEARN_PROBES}

    async def run(self, dash=None):
        """Consume commands, keep the rig ticking while idle, publish at ~10 Hz. "stop" always wins."""
        last = -math.inf
        while True:
            while not self.inbox.empty():
                await self.handle(self.inbox.get_nowait())
            if self.busy:
                await asyncio.sleep(DT)
            else:
                try:
                    await self.rig.step(0, 0)
                except RigAbort as e:
                    self.status = f"out of bounds: {e}"
            if dash and self.rig.pose and self.rig.pose.t - last >= 0.1:
                last = self.rig.pose.t
                dash.publish(self.state())
                if (frame := self.rig.frame()) is not None and self.rig.H is not None:
                    dash.set_frame(topdown(frame, self.rig.H, self.rig.floor))
                    if not hasattr(self.rig, "view"):
                        dash.set_view(frame)

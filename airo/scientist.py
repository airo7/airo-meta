"""Muse Wonder as the scientist: it picks probes, states hypotheses and writes conclusions, while every number comes
from tools. A scripted policy takes over whenever the API fails, so learning never stalls."""

import asyncio
import inspect
import json
import math
import time
from dataclasses import asdict
from typing import Callable

import numpy as np

from airo import meta_api
from airo.body import PRIOR, Body, Segment, ensemble, fit, surprise
from airo.memory import Entry, Memory, since_change
from airo.probes import FLOOR, QTY, LIBRARY, SCHEDULE, Recentre, halt, measure, outcome, predict, run, safe, valid, zscore
from airo.rig import BaseRig, RigAbort

# A forward step first: it tests the refit in the regime the delivery drives in, so it confirms or refutes soonest.
RELEARN = ["step_80", "pivot_l_100", "pivot_r_150", "step_back_60", "pivot_l_60"]
CONFIRM_Z = 2.0
# A vague prediction confirms anything, so the tolerance (sd and z floor combined) must also be tight.
CONFIRM_TOL = np.array([3.0, 3.0, 6.0])
MEASURES = {"step": "per-side gain, drift kappa, delay", "ramp": "dead zone", "pivot": "pivot efficiency eta",
            "arc": "mixing law, side gains", "pulse": "delay tau, coast"}
UNITS = ("cm forward", "cm lateral", "deg turn")
LOOK = "In one short line: is anything attached to, sitting on or towed behind the small robot car?"

SYSTEM = """You are the scientist inside AIRO: a $50 skid-steer robot car with no wheel encoders, watched by an \
overhead camera. You learn the car's body by running physical probe experiments on it.
Model: each side moves at g_side * (|pwm| - dz) past a dead zone dz; v = (vL + vR) / 2; turn rate = eta * (vR - vL) \
/ 0.145 m + kappa * v; commands act after a delay tau, then a wheel lag.
Before every run_probe, state one falsifiable hypothesis: what the probe will show and why you chose it. Pick the \
probe that best separates your hypotheses or cuts the largest spread. |z| > 2 means the model was wrong on that \
quantity. Refit after informative probes.
Never invent numbers: quote only numbers returned by tools. Unsafe or unknown probes are rejected, and the probe \
budget is enforced.
Finish with conclude: one or two plain sentences on what the body is, or what changed and its likely physical cause, \
and a short lowercase body_name such as "heavy" or "towing"."""


def _fn(name: str, description: str, **props) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": list(props)}}}


_PROBE = {"type": "string", "description": "probe name from list_probes"}
TOOLS = [
    _fn("get_body", "Current body parameters, their ensemble spread, and the probe budget left."),
    _fn("list_probes", "Probe library: commands, what each kind measures, and whether it is safe from here."),
    _fn("predict", "The model's predicted outcome of a probe (mean, sd) without running it.", probe=_PROBE),
    _fn("run_probe", "Run a probe on the car. The prediction is recorded first; returns measured outcome and z.",
        probe=_PROBE, hypothesis={"type": "string", "description": "one falsifiable sentence"}),
    _fn("refit", "Refit the body to the current data; returns parameters, spread and % change since the start."),
    _fn("conclude", "Finish with what you learned and a short name for this body.",
        text={"type": "string"}, body_name={"type": "string"}),
]
RECALL = _fn("recall", "Adopt a remembered body and test it with one confirming probe. Confirmed ends the relearn; "
                       "otherwise relearn as usual. Skip it to relearn without trying.",
             n={"type": "integer", "description": "body number of a memory candidate"})


def _r(x: float) -> float:
    return float(f"{x:.3g}")


def describe(before: Body, after: Body) -> str:
    """A short name for how a relearned body differs, e.g. "heavier: −15% speed, −25% turn"."""
    flipped = [f"{side} motor reversed" for side, a, b in (("left", before.g_l, after.g_l),
                                                          ("right", before.g_r, after.g_r)) if a * b < 0]
    if flipped:
        return ", ".join(flipped)
    speed = (after.g_l + after.g_r) / (before.g_l + before.g_r) - 1
    turn = (1 + speed) * after.eta / before.eta - 1
    parts = [f"{v:+.0%} {what}".replace("-", "−") for v, what in ((speed, "speed"), (turn, "turn")) if abs(v) >= 0.05]
    if after.tau + after.lag > 1.3 * (before.tau + before.lag):
        parts.append("slower to respond")
    if not parts:
        return "unchanged"
    return f"{'heavier' if speed < -0.05 else 'lighter' if speed > 0.05 else 'changed'}: {', '.join(parts)}"


def _verdict(probe: str, pred: dict, meas: dict) -> str:
    i = max(range(len(QTY)), key=lambda i: abs(pred[QTY[i]][0]))
    (m, sd), u = pred[QTY[i]], "°" if QTY[i] == "turn_deg" else " cm"
    return f"{probe} {m:.0f}{u} ± {math.hypot(sd, FLOOR[i]):.0f} vs {meas[QTY[i]]:.0f}{u}"


def regime(probe: str) -> str:
    return "turn" if LIBRARY[probe].kind in ("pivot", "arc") else "straight"


def regimes(seg: Segment) -> set[str]:
    """How the car was driving in seg, with body._excited's turn threshold."""
    win = seg.cmd[(seg.cmd_t >= seg.t[0]) & (seg.cmd_t <= seg.t[-1])].reshape(-1, 2)
    moving = win[np.abs(win).max(axis=1) > 0]
    turning = np.abs(moving[:, 0] - moving[:, 1]) > 20
    out = ({"turn"} if turning.any() else set()) | ({"straight"} if (~turning).any() else set())
    return out or {"straight"}


def _fmt(d: dict) -> str:
    return ", ".join(f"{v[0]:.1f} ± {v[1]:.1f} {u}" if isinstance(v, tuple) else f"{v:.1f} {u}"
                     for u, v in zip(UNITS, d.values()))


class Scientist:
    """Notebook entries are {"t", "kind", "text", ...}, kind in alert|hypothesis|prediction|result|conclusion|
    recall|fallback; entries written by the scripted policy also carry "fallback": True.

    A relearn ends once a probe run after a refit lands within |z| < 2 on every quantity, or when the budget runs
    out; `verified` says which. Before relearning it tries the remembered body that best explains the window.
    """

    def __init__(self, rig: BaseRig, recentre: Recentre | None = None, body: Body = PRIOR, k: int = 12,
                 on_update: Callable | None = None, memory: Memory | None = None):
        self.rig = rig
        self.recentre = recentre
        self.best = body
        self.members = ensemble([], body, k)
        self.segments: list[Segment] = []
        self.n = 0
        self.name = "prior"
        self.notebook: list[dict] = []
        self.on_update = on_update
        self.k = k
        self._start = body
        self._anchor: Body | None = None
        self._budget = 0
        self._ran: list[str] = []
        self._stale = self._done = self._fallback = self._verify = False
        self._confirmed: dict[str, str] = {}
        self._needed: set[str] = set()
        self.verified = True
        self.memory = memory or Memory()
        self.recalling: Entry | None = None
        self._candidates: list[Entry] = []
        self._cause = ""
        self._t0 = 0.0

    def adopt(self, e: Entry):
        self.best, self.members, self.n, self.name, self.verified = e.body, e.members, e.n, e.name, e.verified

    async def learn(self, budget: int = 10):
        self.segments, self._anchor, self._verify, self.verified = [], None, False, True
        await self._session(f"Learn this car's body from scratch with {budget} probes. A reasonable schedule is "
                            f"{SCHEDULE[:budget]}, but order the probes to cut the largest uncertainty first. Nothing "
                            "is attached to the car yet; name the body after how it drives.", SCHEDULE, budget)

    async def relearn(self, window_seg: Segment, budget: int = len(RELEARN)) -> float:
        """window_seg ends with the surprise; its part since the change seeds the refit, older segments are dropped."""
        t0 = time.monotonic()
        window_seg = since_change(self.members, window_seg)
        s = surprise(self.members, window_seg)
        s = s[np.isfinite(s)]
        t, pose = valid(window_seg)
        pred = outcome(self.members, t[[0, -1]], window_seg.cmd_t, window_seg.cmd, pose[0])
        meas = measure(window_seg)
        seen, _ = await asyncio.gather(self._look(), halt(self.rig))
        what = (f"surprise {np.median(s[-3:]) if len(s) else math.nan:.1f} over the last {t[-1] - t[0]:.1f} s: "
                f"predicted {_fmt(pred)}; measured {_fmt(meas)}")
        await self._note("alert", f"Body changed, {what}. Camera: {seen}", predicted=pred, measured=meas, seen=seen)
        before, self._t0 = self.best, self.rig.pose.t
        self._cause = seen if seen not in ("no camera frame", "camera description unavailable") else what
        self.segments, self._stale, self._anchor, self._verify = [window_seg], True, self.best, True
        self._needed = regimes(window_seg)
        matches = self.memory.match(window_seg, exclude=self.n)
        self._candidates = [e for e, _ in matches]
        if not matches:
            await self.refit()
        known = "; ".join(f"#{e.n} \"{e.name}\" (surprise {m:.1f}, cause: {e.cause})" for e, m in matches)
        need = " and ".join(f"a {r} probe ({'pivot or arc' if r == 'turn' else 'step'})" for r in sorted(self._needed))
        await self._session(f"The body just changed while driving, {what}. Muse Spark sees: \"{seen}\". "
                            + (f"Remembered bodies that explain this window, best first: {known}. Call recall to "
                               "test one with a single probe, or skip it and relearn: refit first. " if matches else
                               "The body is already refitted on that window. ")
                            + f"You have up to {budget} probes and little floor. Refit after each probe; the next "
                            f"probe then tests the refit. The surprise happened while driving "
                            f"{' and '.join(sorted(self._needed))}, so confirm with {need}: |z| < {CONFIRM_Z:g} on "
                            f"every quantity, with a tight prediction.", RELEARN, budget, start=before,
                            tools=TOOLS + [RECALL] if matches else TOOLS)
        self._candidates = []
        return time.monotonic() - t0

    async def _look(self) -> str:
        frame = self.rig.frame() if hasattr(self.rig, "frame") else None
        if frame is None:
            return "no camera frame"
        try:
            return (await asyncio.wait_for(meta_api.describe(frame, LOOK), 8)).strip()
        except Exception:
            return "camera description unavailable"

    async def _session(self, prompt: str, plan: list[str], budget: int, start: Body | None = None,
                       tools: list[dict] = TOOLS):
        self._start, self._budget, self._ran = start or self.best, budget, []
        self._done = self._fallback = False
        self._confirmed = {}
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        try:
            for _ in range(4 * budget + 8):
                msg = await meta_api.wonder(msgs, tools)
                msgs.append(msg.model_dump(exclude_none=True))
                if not msg.tool_calls:
                    msgs.append({"role": "user", "content": "Continue with a tool call; finish with conclude."})
                for call in msg.tool_calls or []:
                    out = await self._call(call.function.name, call.function.arguments)
                    if self._done:
                        return
                    msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(out)})
            why = "no conclusion within the turn limit"
        except RigAbort:
            raise
        except Exception as e:
            why = type(e).__name__
        await self._scripted(plan, why)

    async def _scripted(self, plan: list[str], why: str):
        self._fallback = True
        left = [p for p in plan if p not in self._ran][:self._budget - len(self._ran)]
        first = f"recall #{self._candidates[0].n}, then " if self._candidates else ""
        await self._note("fallback", f"Wonder unavailable ({why}); scripted: {first}probes {', '.join(left) or 'none'}.")
        if self._candidates and (await self.recall(self._candidates[0].n)).get("recalled"):
            return
        for name in left:
            if self._verify and self._needed <= self._confirmed.keys():
                break
            if self._verify and regime(name) in self._confirmed:
                continue
            await self.refit()
            await self.run_probe(name, f"Scripted {name} for {MEASURES[LIBRARY[name].kind]}.")
        moved = [f"{k} {v:+.0f}%" for k, v in (await self.refit())["change_pct"].items() if abs(v) >= 10]
        await self.conclude(f"Scripted refit: {', '.join(moved) or 'no parameter moved 10% or more'}.",
                            describe(self._start, self.best) if self._verify else "bare chassis")

    async def _call(self, name: str, arguments: str | None) -> dict:
        tool = {"get_body": self.get_body, "list_probes": self.list_probes, "predict": self.predict,
                "run_probe": self.run_probe, "refit": self.refit, "conclude": self.conclude,
                "recall": self.recall}.get(name)
        if tool is None:
            return {"error": f"unknown tool {name}"}
        try:
            return await tool(**json.loads(arguments or "{}"))
        except (json.JSONDecodeError, TypeError) as e:
            return {"error": f"bad arguments: {e}"}

    async def get_body(self) -> dict:
        return {"n": self.n, "name": self.name, "body": self._body(), "spread": self._spread(),
                "probes_left": self._budget - len(self._ran)}

    async def list_probes(self) -> dict:
        return {"probes_left": self._budget - len(self._ran), "probes": [
            {"name": p.name, "pwm": [p.pwm_l, p.pwm_r], "seconds": p.seconds, "measures": MEASURES[p.kind],
             "safe_here": safe(self.rig, p, self.best), "done": p.name in self._ran} for p in LIBRARY.values()]}

    async def predict(self, probe: str) -> dict:
        if probe not in LIBRARY:
            return {"error": f"unknown probe {probe}"}
        return predict(self.members, LIBRARY[probe], self._pose())

    async def run_probe(self, probe: str, hypothesis: str) -> dict:
        p = LIBRARY.get(probe)
        if p is None:
            return {"error": f"unknown probe {probe}"}
        if len(self._ran) >= self._budget:
            return {"error": "probe budget used up: refit and conclude"}
        if not safe(self.rig, p, self.best) and self.recentre:
            await self.recentre(self.rig)
        if not safe(self.rig, p, self.best):
            await self._note("alert", f"Rejected {probe}: not enough free floor from here.", probe=probe)
            return {"error": f"{probe} would leave the floor from here; pick one that needs less space"}
        pred = predict(self.members, p, self._pose())
        confirming = self._verify and not self._stale
        await self._note("hypothesis", hypothesis, probe=probe)
        await self._note("prediction", f"{probe}: {_fmt(pred)}", probe=probe, predicted=pred)
        self.segments.append(await run(self.rig, p))
        self._ran.append(probe)
        self._stale = True
        meas = measure(self.segments[-1])
        z = zscore(pred, meas)
        tol = np.hypot([pred[q][1] for q in QTY], FLOOR)
        confirmed = confirming and max(map(abs, z.values())) < CONFIRM_Z and bool((tol <= CONFIRM_TOL).all())
        if confirmed:
            self._confirmed.setdefault(regime(probe), _verdict(probe, pred, meas))
        await self._note("result", f"{probe}: {_fmt(meas)}; z {', '.join(f'{v:+.1f}' for v in z.values())}"
                         + ("; confirms the refit" if confirmed else ""),
                         probe=probe, hypothesis=hypothesis, predicted=pred, measured=meas, z=z, confirmed=confirmed)
        return {"predicted": pred, "measured": meas, "z": z, "confirmed": confirmed,
                "probes_left": self._budget - len(self._ran)}

    async def recall(self, n: int) -> dict:
        """Adopt remembered body n and run one probe in a regime the surprise drove in. The probe is not charged to
        the budget: if it fails, the relearn keeps its data and goes on from the body it had before."""
        e = next((c for c in self._candidates if c.n == n), None)
        if e is None:
            return {"error": f"#{n} is not a candidate; candidates: {[c.n for c in self._candidates] or 'none'}"}
        self._candidates.remove(e)
        probe = next(p for p in RELEARN if regime(p) in self._needed)
        keep = self.best, self.members
        self.best, self.members, self._stale, self.recalling = e.body, e.members, False, e
        try:
            out = await self.run_probe(probe, f"If this is Body #{e.n} ({e.name}) again, {probe} lands in its cloud.")
        finally:
            self.recalling = None
        if probe in self._ran:
            self._ran.remove(probe)
        if not out.get("confirmed"):
            self.best, self.members, self._stale = *keep, True
            await self._emit()
            return {**out, "recalled": None, "next": f"not Body #{n}: refit, then relearn with probes"}
        self.memory.recall(e)
        self.adopt(e)
        self.verified = self._done = True
        secs = self.rig.pose.t - self._t0
        await self._note("recall", f"I know this one: Body #{e.n}, {e.name}, recalled in {secs:.1f} s with one "
                                   f"probe ({_verdict(probe, out['predicted'], out['measured'])}).",
                         n=e.n, name=e.name, predicted=out["predicted"], measured=out["measured"], z=out["z"],
                         recall_s=secs)
        return {"recalled": e.n, "z": out["z"]}

    async def refit(self) -> dict:
        if self._stale and self.segments:
            self.best = await asyncio.to_thread(fit, self.segments, self.best, anchor=self._anchor, signs=True)
            self.members = await asyncio.to_thread(ensemble, self.segments, self.best, self.k, anchor=self._anchor)
            self._stale = False
            await self._emit()
        before = asdict(self._start)
        return {"body": self._body(), "spread": self._spread(),
                "change_pct": {k: _r(100 * (v / before[k] - 1)) for k, v in asdict(self.best).items()
                               if k != "kappa" and before[k]}}

    async def conclude(self, text: str, body_name: str) -> dict:
        if not self._ran and not self._fallback:
            return {"error": "run at least one probe first"}
        if (self._verify and not self._needed <= self._confirmed.keys() and len(self._ran) < self._budget
                and not self._fallback):
            missing = " and ".join(sorted(self._needed - self._confirmed.keys()))
            return {"error": f"not confirmed yet for {missing} driving: refit, then run such a probe whose |z| "
                             f"stays under {CONFIRM_Z:g}, or use up the probe budget"}
        await self.refit()
        text += "".join(f" The {side} motor runs backwards: g_{side[0]} = {g:.4f}."
                        for side, g in (("left", self.best.g_l), ("right", self.best.g_r)) if g < 0)
        if self._verify:
            self.verified = self._needed <= self._confirmed.keys()
            done = "; ".join(self._confirmed.values())
            text += (f" Confirmed: {done}." if self.verified else
                     f" Partial fit, continuing with caution{f' (confirmed only {done})' if done else ''}.")
        e = self.memory.add(body_name, self._cause if self._verify else "learned from scratch", self.best,
                            self.members, self.verified, n=max(self.memory.next_n(), self.n + 1))
        self.n, self.name = e.n, body_name
        self._done = True
        await self._note("conclusion", text, n=self.n, name=body_name, body=asdict(self.best))
        return {"n": self.n}

    def _pose(self) -> tuple[float, float, float]:
        p = self.rig.pose
        return p.x, p.y, p.theta

    def _body(self) -> dict:
        return {k: _r(v) for k, v in asdict(self.best).items()}

    def _spread(self) -> dict:
        return {k: _r(np.std([getattr(m, k) for m in self.members])) for k in asdict(self.best)}

    async def _note(self, kind: str, text: str, **extra):
        self.notebook.append({"t": time.time(), "kind": kind, "text": text, **extra,
                              **({"fallback": True} if self._fallback else {})})
        await self._emit()

    async def _emit(self):
        if self.on_update and inspect.isawaitable(r := self.on_update(self)):
            await r

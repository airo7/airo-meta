"""Muse Wonder as the scientist: it picks probes, states hypotheses and writes conclusions, while every number comes
from tools. A scripted policy takes over whenever the API fails, so learning never stalls."""

import asyncio
import inspect
import json
import math
import re
import time
from dataclasses import asdict
from typing import Callable

import numpy as np

from airo import eye, meta_api
from airo.body import PRIOR, THRESHOLD, Body, Segment, _residual, ensemble, fit, surprise
from airo.memory import Entry, Memory, since_change
from airo.probes import (FLOOR, QTY, LIBRARY, SCHEDULE, Probe, Recentre, executed, halt, measure, outcome, predict, run,
                         safe, valid, zscore)
from airo.rig import BaseRig, RigAbort

# A forward step first: it tests the refit in the regime the delivery drives in, so it confirms or refutes soonest.
RELEARN = ["step_80", "pivot_l_100", "pivot_r_150", "step_back_60", "pivot_l_60"]
CONFIRM_Z = 2.0
# A vague prediction confirms anything, so the tolerance (sd and z floor combined) must also be tight.
CONFIRM_TOL = np.array([3.0, 3.0, 6.0])
MEASURES = {"step": "per-side gain, drift kappa, delay", "ramp": "dead zone", "pivot": "pivot efficiency eta",
            "arc": "mixing law, side gains", "pulse": "delay tau, coast"}
# check_body: one pivot decides same / remembered / new; a new body then gets a forward step if the bay allows it,
# since pivots alone only see eta times gain.
CHECK = "pivot_l_100"
# In order of preference; a bay's structures can leave room only for a smaller turn one way.
CHECKS = [CHECK, "pivot_r_100", "pivot_l_60", "pivot_r_60"]
CHECK_PLAN = ["step_60", "step_back_60", "pivot_r_150", "pivot_l_60"]
# A surprise relearn this soon after check_body left a body unverified completes that entry rather than minting one.
MERGE_S = 60.0
# A probe whose fit residual is BUMPED times the others' median (and over 2 cm) ran into something: one such probe
# inflates the noise estimate and with it the whole posterior.
BUMPED = 4.0
# A remembered body must be this much more likely (log-likelihood, ~2.7x) than the current one to replace it.
CHANGE_LL = 1.0
UNITS = ("cm forward", "cm lateral", "deg turn")
# The eye check moves the tracked point onto the turning centre when they are over EYE_MOVE apart. While it pivots the
# tracked point swings up to twice that distance, so it needs EYE_ROOM of floor; a fit worse than EYE_RMS means the car
# was disturbed.
EYE_MOVE, EYE_ROOM, EYE_RMS = 0.015, 0.15, 0.01
LOOK = ("Finish the sentence 'The camera sees ...' in at most 12 words, speaking as the small robot car: what is "
        "attached to, sitting on or towed behind me? For example 'a water bottle on top of me', or 'nothing on me'.")
# A single straight speed cannot tell the gain from the dead zone (any g * (pwm - dz) through one point fits), so a
# learn concludes only after straight probes at or below SLOW and at or above FAST PWM, and a pivot. Shortest first.
SLOW, FAST = 60, 100
GAPS = {"slow": ["step_60", "step_back_60", "step_40"], "fast": ["step_110", "step_back_110", "ramp_120", "step_150"],
        "pivot": ["pivot_l_100", "pivot_r_100", "pivot_l_60", "pivot_r_60"]}
NEED = {"slow": ("a slow straight drive", "separates the gain from the dead zone at a second speed"),
        "fast": ("a fast straight drive", "separates the gain from the dead zone at a second speed"),
        "pivot": ("a spin on the spot", "measures pivot efficiency eta")}

SYSTEM = """You are the scientist inside AIRO: a $50 skid-steer robot car with no wheel encoders, watched by an \
overhead camera. You learn the car's body by running physical probe experiments on it.
Model: each side moves at g_side * (|pwm| - dz) past a dead zone dz; v = (vL + vR) / 2; turn rate = eta * (vR - vL) \
/ 0.145 m + kappa * v; commands act after a delay tau, then a wheel lag.
Before every run_probe, give a reason: one plain sentence a non-engineer understands, without jargon (no dz, eta, \
kappa, PWM, z or probe names), e.g. "I'll drive slowly to see whether the motors need a push before they move." \
Also state one falsifiable technical hypothesis: what the probe will show and why you chose it. Pick the \
probe that best separates your hypotheses or cuts the largest spread. |z| > 2 means the model was wrong on that \
quantity. Refit after informative probes.
Never invent numbers: quote only numbers returned by tools. Unsafe or unknown probes are rejected, and the probe \
budget is enforced.
Finish with conclude: text is one plain sentence for a non-engineer, without jargon, on what the body is or what \
changed and its likely physical cause, e.g. "This car drives straight and evenly; it barely needs any power to start \
moving."; detail is the technical version with parameters and tool numbers; body_name is short and lowercase, such \
as "heavy" or "towing"."""


def _fn(name: str, description: str, **props) -> dict:
    return {"type": "function", "function": {"name": name, "description": description, "parameters": {
        "type": "object", "properties": props, "required": list(props)}}}


_PROBE = {"type": "string", "description": "probe name from list_probes"}
TOOLS = [
    _fn("get_body", "Current body parameters, their ensemble spread, and the probe budget left."),
    _fn("list_probes", "Probe library: commands, what each kind measures, and whether it is safe from here."),
    _fn("predict", "The model's predicted outcome of a probe (mean, sd) without running it.", probe=_PROBE),
    _fn("run_probe", "Run a probe on the car. The prediction is recorded first; returns measured outcome and z.",
        probe=_PROBE, reason={"type": "string", "description": "why, in one plain jargon-free sentence"},
        hypothesis={"type": "string", "description": "one falsifiable technical sentence"}),
    _fn("refit", "Refit the body to the current data; returns parameters, spread and % change since the start."),
    _fn("conclude", "Finish with what you learned and a short name for this body.",
        text={"type": "string", "description": "one plain jargon-free sentence"},
        detail={"type": "string", "description": "the technical conclusion"}, body_name={"type": "string"}),
]
# In a relearn run_probe refits first and records the engine's prediction, and the prompt carries the body and the
# safe probes, so Wonder needs no refit, predict, get_body or list_probes turns: each Wonder turn costs ~6 s.
RELEARN_TOOLS = [t for t in TOOLS if t["function"]["name"] in ("run_probe", "conclude")]
# Wonder thinking time per relearn, after which the scripted policy finishes it.
WONDER_S = 12.0
RECALL = _fn("recall", "Adopt a remembered body and test it with one confirming probe. Confirmed ends the relearn; "
                       "otherwise relearn as usual. Skip it to relearn without trying.",
             n={"type": "integer", "description": "body number of a memory candidate"})


def _r(x: float) -> float:
    return float(f"{x:.3g}")


def _moved(fit: dict) -> str:
    moved = [f"{k} {v:+.0f}%" for k, v in fit["change_pct"].items() if abs(v) >= 10]
    if abs(fit["dz_change_pwm"]) >= 3:
        moved.append(f"dz {fit['dz_change_pwm']:+.0f} PWM")
    return ", ".join(moved) or "no parameter moved 10% or more"


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


def _cap(s: str) -> str:
    return s[:1].upper() + s[1:]


def _unit(q: str) -> str:
    return "°" if q == "turn_deg" else " cm"


def _amount(q: str, v: float) -> str:
    """e.g. "35 cm forward", "30° left"."""
    way = ("forward", "back") if q == "forward_cm" else ("left", "right")
    return f"{abs(v):.0f}{_unit(q)} {way[v < 0]}"


def _went(q: str, v: float) -> str:
    """e.g. "drove 35 cm", "backed up 20 cm", "turned 30° left"."""
    if q == "forward_cm":
        return f"{'drove' if v >= 0 else 'backed up'} {abs(v):.0f} cm"
    return f"{'turned' if q == 'turn_deg' else 'slid'} {_amount(q, v)}"


def _main(kind: str) -> str:
    return "turn_deg" if kind == "pivot" else "forward_cm"


def _tol(pred: dict, q: str) -> float:
    return math.hypot(pred[q][1], FLOOR[QTY.index(q)])


def _expect(pred: dict, kind: str) -> str:
    q = _main(kind)
    return f"I expect about {_amount(q, pred[q][0])}, give or take {_tol(pred, q):.0f}{_unit(q)}."


def _outcome(pred: dict, meas: dict, z: dict, kind: str) -> str:
    """e.g. "Drove 37 cm, expected 35 ± 2 cm — as predicted."."""
    q = _main(kind)
    m, more = pred[q][0], abs(meas[q]) > abs(pred[q][0])
    s = f"{_cap(_went(q, meas[q]))}, expected {abs(m):.0f} ± {_tol(pred, q):.0f}{_unit(q)} — "
    if abs(z[q]) < CONFIRM_Z:
        s += "as predicted"
    elif q == "forward_cm":
        s += f"much {'further' if more else 'less far'} than expected"
    else:
        s += f"much {'more' if more else 'less'} than expected"
    off = [f"{_went(k, meas[k])} (expected about {_amount(k, pred[k][0])})" for k in QTY
           if k != q and abs(z[k]) >= CONFIRM_Z]
    return s + (f", but {' and '.join(off)}" if off else "") + "."


def _fills(p: Probe) -> set[str]:
    """Which of GAPS a probe covers."""
    if p.kind == "pivot":
        return {"pivot"}
    pwm = max(abs(p.pwm_l), abs(p.pwm_r))
    if p.kind not in ("step", "ramp"):
        return set()
    return {"slow"} if pwm <= SLOW else {"fast"} if pwm >= FAST else set()


def _doing(p: Probe) -> str:
    """e.g. "drive forward slowly", "spin left on the spot"."""
    pwm, left = max(abs(p.pwm_l), abs(p.pwm_r)), p.pwm_r > p.pwm_l
    pace = "slowly" if pwm <= SLOW else "fast" if pwm >= FAST else "at medium speed"
    return {"step": f"{'drive forward' if p.pwm_l > 0 else 'back up'} {pace}",
            "ramp": "speed up gently from a standstill",
            "pivot": f"spin {'left' if left else 'right'} on the spot",
            "arc": f"drive a curve to the {'left' if left else 'right'}",
            "pulse": "give a short push"}[p.kind]


WHY = {"step": "to see how fast I go and whether I drive straight",
       "ramp": "to find how much power my motors need before they move", "pivot": "to see how well I turn",
       "arc": "to see how my two sides work together", "pulse": "to see how quickly I react"}


def _reason(p: Probe) -> str:
    slow = p.kind == "step" and _fills(p) == {"slow"}
    return f"I'll {_doing(p)} {'to see how much power I need to start moving' if slow else WHY[p.kind]}."


def _feel(before: Body, after: Body) -> str:
    """How a relearned body differs, in plain words, e.g. "I feel heavier now: slower and turning less"."""
    flipped = [f"my {side} motor runs backwards now" for side, a, b in (("left", before.g_l, after.g_l),
                                                                        ("right", before.g_r, after.g_r)) if a * b < 0]
    if flipped:
        return _cap(" and ".join(flipped))
    speed = (after.g_l + after.g_r) / (before.g_l + before.g_r) - 1
    turn = (1 + speed) * after.eta / before.eta - 1
    dz, react = after.dz - before.dz, (after.tau + after.lag) / (before.tau + before.lag)
    parts = [w for ok, w in ((abs(speed) >= 0.05, "faster" if speed > 0 else "slower"),
                             (abs(turn) >= 0.05, f"turning {'more' if turn > 0 else 'less'}"),
                             (abs(dz) >= 5, f"needing {'more' if dz > 0 else 'less'} power to start"),
                             (not 1 / 1.3 <= react <= 1.3, f"reacting {'later' if react > 1 else 'sooner'}")) if ok]
    if not parts:
        return "My body feels the same as before"
    lead = "I feel heavier now" if speed < -0.05 else "I feel lighter now" if speed > 0.05 else "My body changed"
    return f"{lead}: {' and '.join(filter(None, (', '.join(parts[:-1]), parts[-1])))}"


def _sees(seen: str) -> str:
    if seen == "no camera frame":
        return "No camera view, so I can't see what changed."
    if seen == "camera description unavailable":
        return "The camera couldn't tell me what changed."
    what = re.sub(r"^((the camera sees|i see|yes|no)[,:]?\s+)+", "", seen.strip().strip("'\"").rstrip("."), flags=re.I)
    return f"The camera sees {what}."


def _where(m) -> str:
    """A body-frame point (forward, left) in metres, in words, e.g. "6.4 cm behind and 1.0 cm left of"."""
    fwd = f"{abs(m[0]) * 100:.1f} cm {'in front of' if m[0] > 0 else 'behind'}" if abs(m[0]) >= 0.005 else ""
    side = f"{abs(m[1]) * 100:.1f} cm {'left' if m[1] > 0 else 'right'} of" if abs(m[1]) >= 0.005 else ""
    return " and ".join(filter(None, (fwd.removesuffix(" of") if side else fwd, side)))


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


def _compact(x):
    """Tool results rounded to 3 significant figures: the context grows with every turn."""
    if isinstance(x, float):
        return _r(x)
    if isinstance(x, dict):
        return {k: _compact(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_compact(v) for v in x]
    return x


def _n(e, current: int) -> int:
    return current if e is None else e.n


def _agree(a: dict, b: dict) -> bool:
    """Two outcomes of the same probe within three noise floors of each other."""
    return all(abs(a[q] - b[q]) < 3 * f for q, f in zip(QTY, FLOOR))


def _brief(out: dict) -> str:
    """A tool result in a few words for the dashboard's model feed."""
    if "error" in out:
        return out["error"]
    if isinstance(out.get("z"), dict):
        return "z " + ", ".join(f"{v:+.1f}" for v in out["z"].values()) + (" ✓" if out.get("confirmed") else "")
    return json.dumps(out, default=str)[:90]


class Scientist:
    """Notebook entries are {"t", "kind", "text", ...}, kind in alert|hypothesis|prediction|result|conclusion|
    recall|fallback|eye; entries written by the scripted policy also carry "fallback": True.

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
        self._gate = self._refused = self._filled = False
        self._start_members = self.members
        self.change = ""
        self._confirmed: dict[str, str] = {}
        self._needed: set[str] = set()
        self.verified = True
        self.memory = memory or Memory()
        self.recalling: Entry | None = None
        self._candidates: list[Entry] = []
        self._cause = ""
        self._looking: asyncio.Task | None = None
        self._unfinished: tuple[float, int, Body] | None = None
        self._merge: int | None = None
        self._signs = True
        self._t0 = 0.0

    def adopt(self, e: Entry):
        self.best, self.members, self.n, self.name, self.verified = e.body, e.members, e.n, e.name, e.verified

    async def learn(self, budget: int = 10):
        self.segments, self._anchor, self._verify, self.verified = [], None, False, True
        self._fallback = False
        await self.find_eye()
        await self._session(f"Learn this car's body from scratch with {budget} probes. A reasonable schedule is "
                            f"{SCHEDULE[:budget]}, but order the probes to cut the largest uncertainty first. Nothing "
                            "is attached to the car yet; name the body after how it drives. You may conclude only "
                            f"after straight probes at two clearly different speeds (one at or below {SLOW} PWM, one "
                            f"at or above {FAST} PWM, forward or back) and at least one pivot: one speed cannot tell "
                            "the gain from the dead zone.", SCHEDULE, budget, gate=True)

    async def find_eye(self):
        """On a rig that tracks a marker, pivot both ways in place and move the tracked point onto the turning centre:
        a marker taped off it makes every pivot look like a slide. The pivots are not probes, so they stay out of the
        fit."""
        tracker = getattr(self.rig, "tracker", None)
        if tracker is None:
            return
        if not all(safe(self.rig, p, self.best, margin=EYE_ROOM) for p in eye.PIVOTS):
            await self._note("eye", "No room to pivot here, so I track my marker as it is.")
            return
        runs = await eye.pivots(self.rig)
        d, rms = eye.fit(runs)
        if (turn := min(eye.turns(runs))) < eye.MIN_TURN or rms > EYE_RMS:
            await self._note("eye", "I couldn't tell where my marker sits on me, so I track it as it is.",
                             detail=f"My pivots could not place my marker (turned {turn:.0f}°, fit rms "
                                    f"{rms * 1000:.0f} mm).")
            return
        at = {"marker_cm": [round(-v * 100, 1) for v in d], "rms_mm": round(rms * 1000, 1)}
        if math.hypot(*d) <= EYE_MOVE:
            await self._note("eye", f"My marker sits on my turning centre, within {math.hypot(*d) * 100:.1f} cm.", **at)
            return
        tracker.offset = tuple(float(v) for v in np.add(tracker.offset, d))
        # the next pose is in the new frame: the probes start from it
        await halt(self.rig, 0.25)
        await self._note("eye", f"My marker sits {_where(-d)} my turning centre; I now track the centre.", moved=True,
                         **at)

    async def relearn(self, window_seg: Segment, budget: int = len(RELEARN)) -> float:
        """window_seg ends with the surprise; its part since the change seeds the refit, older segments are dropped."""
        t0 = time.monotonic()
        matches = self.memory.match(window_seg, self.members, exclude=self.n)
        # Only windows the old body explains well are surely from before the change: a slowly growing surprise,
        # like a tow's, is already the new body below threshold.
        window_seg = since_change(self.members, window_seg, THRESHOLD / 2)
        s = surprise(self.members, window_seg)
        s = s[np.isfinite(s)]
        t, pose = valid(window_seg)
        pred = outcome(self.members, t[[0, -1]], window_seg.cmd_t, window_seg.cmd, pose[0])
        meas = measure(window_seg)
        self._looking = asyncio.create_task(self._look())
        await halt(self.rig)
        what = (f"surprise {np.median(s[-3:]) if len(s) else math.nan:.1f} over the last {t[-1] - t[0]:.1f} s: "
                f"predicted {_fmt(pred)}; measured {_fmt(meas)}")
        z = zscore(pred, meas)
        q = max(QTY, key=lambda q: abs(z[q]))
        await self._note("alert", f"My body feels different: I {_went(q, meas[q])} where I expected "
                                  f"{_amount(q, pred[q][0])}.", detail=f"Body changed, {what}.", predicted=pred,
                         measured=meas, changed=True)
        before, self._t0, self._cause = self.best, self.rig.pose.t, what
        merge, self._unfinished = self._unfinished, None
        if merge and self._t0 - merge[0] < MERGE_S:
            self._merge, before = merge[1], merge[2]
        self.segments, self._stale, self._anchor, self._verify = [window_seg], True, self.best, True
        self._needed = regimes(window_seg)
        self._candidates = [e for e, _ in matches]
        if not matches:
            await self.refit()
        known = "; ".join(f"#{e.n} \"{e.name}\" (surprise {m:.1f}, cause: {e.cause})" for e, m in matches)
        need = " and ".join(f"a {r} probe ({'pivot or arc' if r == 'turn' else 'step'})" for r in sorted(self._needed))
        await self._session(f"The body just changed while driving, {what}. Muse Spark is looking at the car; its "
                            "answer comes with your first tool result. "
                            + (f"Remembered bodies that explain this window, best first: {known}. Call recall to "
                               "test one with a single probe, or skip it and relearn. " if matches else
                               "The body is already refitted on that window. ")
                            + f"{self._summary()} You have up to {budget} probes and little floor. Each run_probe "
                            f"refits first, so it tests the latest fit. The surprise happened while driving "
                            f"{' and '.join(sorted(self._needed))}, so confirm with {need}: |z| < {CONFIRM_Z:g} on "
                            f"every quantity, with a tight prediction. Act now: your first call is "
                            f"{'recall or ' if matches else ''}run_probe.", RELEARN, budget, start=before,
                            tools=RELEARN_TOOLS + [RECALL] if matches else RELEARN_TOOLS, wonder_s=WONDER_S)
        await self._seen()
        self._candidates, self._merge = [], None
        return time.monotonic() - t0

    async def _seen(self) -> str | None:
        """Muse Spark's look at the car, noted once it is in. It runs during Wonder's first turn: waiting for it
        first would add its 5-8 s to every relearn."""
        if self._looking is None:
            return None
        seen, self._looking = await self._looking, None
        if seen not in ("no camera frame", "camera description unavailable"):
            self._cause = seen
        await self._note("alert", _sees(seen), detail=f"Muse Spark: {seen}", seen=seen)
        return seen

    async def _look(self) -> str:
        frame = self.rig.frame() if hasattr(self.rig, "frame") else None
        if frame is None:
            return "no camera frame"
        try:
            return (await asyncio.wait_for(meta_api.describe(frame, LOOK), 8)).strip()
        except Exception:
            return "camera description unavailable"

    async def _session(self, prompt: str, plan: list[str], budget: int, start: Body | None = None,
                       tools: list[dict] = TOOLS, wonder_s: float | None = None, gate: bool = False):
        """wonder_s caps the time spent waiting on Wonder; probes driving in between don't count. gate holds the
        conclusion until the probes cover GAPS."""
        self._start, self._start_members, self._budget, self._ran = start or self.best, self.members, budget, []
        self._done = self._fallback = self._refused = self._filled = False
        self._gate = gate
        self._confirmed = {}
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        why, left = "no conclusion within the turn limit", wonder_s
        try:
            for _ in range(4 * budget + 8):
                if left is not None and left <= 0:
                    why = f"no conclusion within {wonder_s:g} s of Wonder time"
                    break
                t = time.monotonic()
                msg = await (meta_api.wonder(msgs, tools) if left is None else
                             meta_api.wonder(msgs, tools, timeout=left))
                if left is not None:
                    left -= time.monotonic() - t
                msgs.append(msg.model_dump(exclude_none=True))
                if not msg.tool_calls:
                    msgs.append({"role": "user", "content": "Continue with a tool call; finish with conclude."})
                for call in msg.tool_calls or []:
                    k = len(self.notebook)
                    out = await self._call(call.function.name, call.function.arguments)
                    said = next((e["text"] for e in reversed(self.notebook[k:]) if e["kind"] != "eye"), None)
                    meta_api.tool_result(call.id, _brief(out), said or out.get("error"))
                    if not self._done and (seen := await self._seen()) is not None:
                        out = {**out, "camera": seen}
                    if self._done:
                        return
                    msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(_compact(out))})
        except RigAbort:
            raise
        except Exception as e:
            why = type(e).__name__
        await self._scripted(plan, why)

    async def _scripted(self, plan: list[str], why: str):
        self._fallback = True
        await self._seen()
        left = [p for p in plan if p not in self._ran][:self._budget - len(self._ran)]
        first = f"recall #{self._candidates[0].n}, then " if self._candidates else ""
        await self._note("fallback", "Muse Wonder couldn't finish this, so I'm following my own test plan.",
                         detail=f"Wonder unavailable ({why}); scripted: {first}probes {', '.join(left) or 'none'}.")
        if self._candidates and (await self.recall(self._candidates[0].n)).get("recalled"):
            return
        for name in left:
            if self._verify and self._needed <= self._confirmed.keys():
                break
            if self._verify and regime(name) in self._confirmed:
                continue
            await self.refit()
            await self.run_probe(name, f"Scripted {name} for {MEASURES[LIBRARY[name].kind]}.")
        if self._gate:
            await self._fill()
        await self.refit()
        await self.conclude(f"{_feel(self._start, self.best)}." if self._verify else self.sketch(),
                            describe(self._start, self.best) if self._verify else "bare chassis", "Scripted refit.")

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
        return {"probes_left": self._budget - len(self._ran), "probes": {
            p.name: f"{p.pwm_l}/{p.pwm_r} PWM for {p.seconds:g} s, {MEASURES[p.kind]}"
                    f"{'' if safe(self.rig, p, self.best) else ', unsafe here'}{', done' if p.name in self._ran else ''}"
            for p in LIBRARY.values()}}

    def _summary(self) -> str:
        """The body, its spread and the safe probes in one line, so Wonder needs no turn to ask for them."""
        body, spread = self._body(), self._spread()
        ok = [f"{p.name} ({p.kind})" for p in LIBRARY.values() if safe(self.rig, p, self.best)]
        return (f"Body now: {', '.join(f'{k} {v:.3g} ± {spread[k]:.2g}' for k, v in body.items())}. "
                f"Probes safe from here: {', '.join(ok) or 'none, run_probe recentres first'}.")

    async def predict(self, probe: str) -> dict:
        if probe not in LIBRARY:
            return {"error": f"unknown probe {probe}"}
        return predict(self.members, LIBRARY[probe], self._pose())

    async def run_probe(self, probe: str, hypothesis: str = "", reason: str = "") -> dict:
        p = LIBRARY.get(probe)
        if p is None:
            return {"error": f"unknown probe {probe}"}
        if len(self._ran) >= self._budget and not (self._gate and _fills(p) & set(self._missing())):
            return {"error": "probe budget used up: refit and conclude"}
        if not safe(self.rig, p, self.best) and self.recentre:
            await self.recentre(self.rig)
        if not safe(self.rig, p, self.best):
            await self._note("alert", f"Not enough free floor here to {_doing(p)}, so I skipped that test.",
                             detail=f"Rejected {probe}: not enough free floor from here.", probe=probe)
            return {"error": f"{probe} would leave the floor from here; pick one that needs less space"}
        if self._verify and self.recalling is None:
            await self.refit()
        pred = predict(self.members, p, self._pose())
        confirming = self._verify and not self._stale
        await self._note("hypothesis", reason or _reason(p), detail=hypothesis, probe=probe)
        await self._note("prediction", _expect(pred, p.kind), detail=f"{probe}: {_fmt(pred)}", probe=probe,
                         predicted=pred)
        self.segments.append(await run(self.rig, p))
        self._ran.append(probe)
        self._stale = True
        pred, meas = executed(self.members, self.segments[-1]), measure(self.segments[-1])
        z = zscore(pred, meas)
        tol = np.hypot([pred[q][1] for q in QTY], FLOOR)
        confirmed = confirming and max(map(abs, z.values())) < CONFIRM_Z and bool((tol <= CONFIRM_TOL).all())
        if confirmed:
            self._confirmed.setdefault(regime(probe), _verdict(probe, pred, meas))
        await self._note("result", _outcome(pred, meas, z, p.kind) + (" That confirms what I relearned." if confirmed
                                                                      else ""),
                         detail=f"{probe}: {_fmt(meas)}; z {', '.join(f'{v:+.1f}' for v in z.values())}"
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
            out = await self.run_probe(probe, f"If this is Body #{e.n} ({e.name}) again, {probe} lands in its cloud.",
                                       f"Maybe I'm Body #{e.n} ({e.name}) again: one test drive will tell.")
        finally:
            self.recalling = None
        if probe in self._ran:
            self._ran.remove(probe)
        if not out.get("confirmed"):
            self.best, self.members, self._stale = *keep, True
            await self._emit()
            return {**out, "recalled": None, "next": f"not Body #{n}: refit, then relearn with probes"}
        await self._recalled(e, probe, out["predicted"], out["measured"], out["z"])
        self._done = True
        return {"recalled": e.n, "z": out["z"]}

    async def _recalled(self, e: Entry, probe: str, pred: dict, meas: dict, z: dict):
        self.memory.recall(e)
        self.adopt(e)
        self.verified = True
        secs = self.rig.pose.t - self._t0
        await self._note("recall", f"I recognise this body: Body #{e.n} ({e.name}), from one test drive.",
                         detail=f"Recalled in {secs:.1f} s with one probe ({_verdict(probe, pred, meas)}).",
                         n=e.n, name=e.name, predicted=pred, measured=meas, z=z, recall_s=secs)

    async def check_body(self, cause: str) -> dict:
        """After a known load or unload, parked at a cell: an in-place pivot tells whether the body is the same, a
        remembered one, or new; a new one is then learned at the cell. Scripted throughout, so driving never waits
        on Wonder. The car never recentres here, and straight probes run only where safe() allows."""
        before, recentre, self.recentre = self.best, self.recentre, None
        self._t0, self._cause, self._unfinished, self._merge = self.rig.pose.t, cause, None, None
        self.segments, self._stale, self._anchor, self._verify, self._fallback = [], False, before, True, False
        self._ran, self._budget, self._confirmed, self._start = [], 3 + len(CHECK_PLAN), {}, before
        self._needed, self._signs, self._gate = {"straight", "turn"}, False, False
        try:
            await halt(self.rig)
            if (first := self._clear(CHECKS)) is None:
                await self._note("alert", f"{_cap(cause)}: no room to spin here, so I can't check my body. Keeping "
                                          f"Body #{self.n}; driving will show a change.")
                return self._checked("same")
            await self._note("alert", f"{_cap(cause)}: a quick spin to check whether my body changed.",
                             detail=f"Checking with {first}.")
            out = await self.run_probe(first, f"If my body did not change, {first} lands in the current cloud.")
            if "error" in out:
                return self._checked("same")
            runs = [(self.segments[-1], out["measured"])]
            if not (fits := self._weigh(runs)):
                # Unexplained by every body I know: once more, in case something bumped the car.
                self._stale = False
                again = await self.run_probe(first, f"Again: if {first} repeats, the body really changed.")
                runs = [(self.segments[-1], again["measured"])]
                if not (fits := self._weigh(runs)) and not _agree(out["measured"], again["measured"]):
                    await self._note("alert", f"Two test spins disagree and no body I know explains them: something "
                                              f"is moving me. Keeping Body #{self.n}; driving will show a change.",
                                     detail=f"Two {first} runs disagree.")
                    return self._checked("same")
            elif len(fits) > 1 and fits[0][0] - fits[1][0] < CHANGE_LL:
                # Two bodies explain it about as well: another pivot, the other way if there is room, tells them apart.
                self._stale = False
                second = self._clear([c for c in CHECKS if c != first]) or first
                again = await self.run_probe(second, f"{second} tells #{_n(fits[0][1], self.n)} and "
                                                     f"#{_n(fits[1][1], self.n)} apart.")
                if "error" not in again:
                    runs.append((self.segments[-1], again["measured"]))
                    fits = self._weigh(runs)
            body = self._choose(fits, runs[-1][1], self._ran[-1])
            if body == "current":
                await self._note("result", f"Same body as before: Body #{self.n} ({self.name}).",
                                 detail=f"Body #{self.n} still explains {first}.")
                return self._checked("same")
            if body is not None:
                await self._recalled(*body)
                return self._checked("recalled")
            await self._learn_here(cause, before)
            return self._checked("learned")
        finally:
            self.recentre, self._signs = recentre, True

    def _clear(self, probes: list[str]) -> str | None:
        return next((p for p in probes if safe(self.rig, LIBRARY[p], self.best)), None)

    def _checked(self, result: str) -> dict:
        return {"result": result, "n": self.n, "name": self.name, "seconds": self.rig.pose.t - self._t0}

    def _weigh(self, runs: list[tuple[Segment, dict]]) -> list[tuple]:
        """(log-likelihood, body, prediction, z) of the last run for each body that fits every run (|z| under
        CONFIRM_Z, tight), most likely first; body None is the current one. The Gaussian log-likelihood, summed over
        runs, also charges a wide cloud for its spread: a vague body can fit a changed car too."""
        fits = []
        for e in [None] + [e for e in self.memory.entries if e.n != self.n]:
            ll, ok = 0.0, True
            for seg, meas in runs:
                pred = executed(self.members if e is None else e.members, seg)
                z = zscore(pred, meas)
                sd = np.hypot([pred[q][1] for q in QTY], FLOOR)
                ok = ok and max(map(abs, z.values())) < CONFIRM_Z and bool((sd <= CONFIRM_TOL).all())
                ll += -sum(v * v for v in z.values()) / 2 - float(np.log(sd).sum())
            if ok:
                fits.append((ll, e, pred, z))
        return sorted(fits, key=lambda f: -f[0])

    def _choose(self, fits: list[tuple], meas: dict, probe: str) -> str | tuple | None:
        """"current", recall args for a remembered body, or None to learn: the current body is kept unless another
        is CHANGE_LL more likely."""
        if not fits:
            return None
        current = next((f for f in fits if f[1] is None), None)
        if current and current[0] >= fits[0][0] - CHANGE_LL:
            return "current"
        _, e, pred, z = fits[0]
        return e, probe, pred, meas, z

    async def _learn_here(self, cause: str, before: Body):
        for name in CHECK_PLAN:
            if self._needed <= self._confirmed.keys():
                break
            if regime(name) in self._confirmed or not safe(self.rig, LIBRARY[name], self.best):
                continue
            await self.run_probe(name, f"After the refit, {name} lands in the new cloud.")
        self._budget = len(self._ran)
        await self.refit()
        await self.conclude(f"{_cap(cause)}. {_feel(before, self.best)}.", describe(before, self.best), f"{cause}.")
        if not self.verified:
            self._unfinished = (self.rig.pose.t, self.n, before)

    async def refit(self) -> dict:
        if self._stale and self.segments:
            self.best = await asyncio.to_thread(fit, self.segments, self.best, anchor=self._anchor, signs=self._signs)
            if not self._verify and (i := self._bumped()) is not None:
                name = self._ran[i] if len(self._ran) == len(self.segments) else "a probe"
                del self.segments[i]
                await self._note("alert", "One test drive looks like I bumped into something, so I'm leaving it "
                                          "out.", detail=f"{name} does not fit the others: it ran into something, so "
                                                         "it is left out of the fit.")
                self.best = await asyncio.to_thread(fit, self.segments, self.best, anchor=self._anchor,
                                                    signs=self._signs)
            self.members = await asyncio.to_thread(ensemble, self.segments, self.best, self.k, anchor=self._anchor)
            self._stale = False
            await self._emit()
        before, after = asdict(self._start), asdict(self.best)
        # a dead zone near 0 PWM makes any percentage explode, so it moves in PWM
        return {"body": self._body(), "spread": self._spread(),
                "change_pct": {k: _r(100 * (v / before[k] - 1)) for k, v in after.items()
                               if k not in ("kappa", "dz") and before[k]},
                "dz_change_pwm": _r(after["dz"] - before["dz"])}

    def _bumped(self) -> int | None:
        """Index of a probe segment the fit cannot explain while it explains the rest, if there is one."""
        if len(self.segments) < 4:
            return None
        rms = np.array([np.sqrt(np.mean(_residual(self.best, s) ** 2)) for s in self.segments])
        i = int(rms.argmax())
        rest = float(np.median(np.delete(rms, i)))
        return i if rms[i] > max(BUMPED * rest, 0.02) else None

    async def conclude(self, text: str, body_name: str, detail: str = "") -> dict:
        """text is the plain sentence, detail the technical one; a learn short of GAPS is refused once, then the
        engine fills them itself, and one it cannot fill ends uncertain."""
        if not self._ran and not self._fallback:
            return {"error": "run at least one probe first"}
        await self._seen()
        if self._gate and self._missing() and not (self._fallback or self._filled):
            if not self._refused:
                self._refused = True
                return {"error": self._lacking()}
            ran = await self._fill()
            if not self._missing():
                return {"error": f"I ran {', '.join(ran)} myself to fill the gap: refit, then conclude again."}
        if self._gate and self._missing():
            return await self._uncertain()
        if (self._verify and not self._needed <= self._confirmed.keys() and len(self._ran) < self._budget
                and not self._fallback):
            missing = " and ".join(sorted(self._needed - self._confirmed.keys()))
            return {"error": f"not confirmed yet for {missing} driving: refit, then run such a probe whose |z| "
                             f"stays under {CONFIRM_Z:g}, or use up the probe budget"}
        moved = _moved(await self.refit())
        detail = " ".join(filter(None, (detail, f"Change since the start: {moved}.")))
        for side, g in (("left", self.best.g_l), ("right", self.best.g_r)):
            if g < 0:
                text += f" My {side} motor runs backwards."
                detail += f" The {side} motor runs backwards: g_{side[0]} = {g:.4f}."
        if self._verify:
            self.verified = self._needed <= self._confirmed.keys()
            self.change = _feel(self._start, self.best)
            done = "; ".join(self._confirmed.values())
            text += "" if self.verified else " I'm not fully sure of it yet, so I'll drive carefully."
            detail += (f" Confirmed: {done}." if self.verified else
                       f" Partial fit, continuing with caution{f' (confirmed only {done})' if done else ''}.")
        # A relearn that lands on a body already in memory (the current one first) refines that entry: minting an
        # "unchanged" twin would split one body's memory in two.
        same = next((e for e in sorted(self.memory.entries, key=lambda e: e.n != self.n)
                     if describe(e.body, self.best) == "unchanged"), None) if self._verify else None
        if self._merge and (e := self.memory.get(self._merge)):
            self.memory.update(e, body_name, self.best, self.members, self.verified)
        elif same and self.verified:
            e, body_name = same, same.name
            self.memory.update(e, body_name, self.best, self.members, True)
            text += f" That's Body #{e.n} ({e.name}) again."
            detail += f" The same body as #{e.n} ({e.name}), now refined."
        elif same:
            e, body_name = same, same.name
            self.memory.recall(e)
            self.adopt(e)
            text += f" That's Body #{e.n} ({e.name}) again."
            detail += f" The same body as #{e.n} ({e.name}); keeping its confirmed fit."
        else:
            e = self.memory.add(body_name, self._cause if self._verify else "learned from scratch", self.best,
                                self.members, self.verified, n=max(self.memory.next_n(), self.n + 1))
        self.n, self.name = e.n, body_name
        self._done = True
        await self._note("conclusion", text, detail=detail, n=self.n, name=body_name, body=asdict(self.best))
        return {"n": self.n}

    def _missing(self) -> list[str]:
        have = set().union(*(_fills(LIBRARY[p]) for p in self._ran))
        return [k for k in GAPS if k not in have]

    def _lacking(self) -> str:
        need = self._missing()
        why = ("driving at one speed can't tell how much power I need to start moving from how fast I go"
               if {"slow", "fast"} & set(need) else "only a spin shows how well I turn")
        return (f"Not yet: {why}. Still missing: "
                + "; ".join(f"{NEED[k][0]} ({' or '.join(GAPS[k])})" for k in need)
                + ". Run these first; they run even if the probe budget is used up.")

    async def _fill(self) -> list[str]:
        """Run the first probe of each missing gap that fits from here, or after recentring."""
        self._filled, ran = True, []
        for need in self._missing():
            name = self._clear(GAPS[need])
            if name is None and self.recentre:
                await self.recentre(self.rig)
                name = self._clear(GAPS[need])
            if name is None:
                continue
            await self.refit()
            if "error" not in await self.run_probe(name, f"{name} {NEED[need][1]}."):
                ran.append(name)
        return ran

    async def _uncertain(self) -> dict:
        """A fit that explains a single speed drives worse than the body the learn started from, so that one stays
        and nothing is saved."""
        need = self._missing()
        self.best, self.members, self.verified, self._done = self._start, self._start_members, False, True
        await self._note("conclusion", f"I couldn't pin down my body: there was no room for "
                                       f"{' or '.join(NEED[k][0] for k in need)}, so I'm keeping my old body and "
                                       "driving carefully.",
                         detail=f"Not identifiable without {', '.join(need)} probes; ran {', '.join(self._ran)}. The "
                                "fit is discarded, not saved.", n=self.n, name=self.name, body=asdict(self.best),
                         uncertain=True)
        return {"n": self.n, "uncertain": True}

    def sketch(self) -> str:
        """The learned body in plain words."""
        b = self.best
        turn = predict(self.members, LIBRARY["step_110"], (0.0, 0.0, 0.0))["turn_deg"][0]
        way = "drives straight" if abs(turn) < 3 else f"pulls to the {'left' if turn > 0 else 'right'}"
        speed = 50 * (abs(b.g_l) + abs(b.g_r)) * max(110 - b.dz, 0)
        start = ("barely needs any power to start moving" if b.dz < 15 else
                 "needs a little power before it starts moving" if b.dz < 40 else
                 "needs a lot of power before it starts moving")
        return f"This car {way}, covers about {speed:.0f} cm a second at medium power and {start}."

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

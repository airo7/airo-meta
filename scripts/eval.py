"""Benchmark: seeded, headless episodes of learn, then deliver with a body change mid-transit, relearn and finish.

The scientist runs its scripted policy (no API). Each seed learns once; every condition then delivers on a fresh rig
from the learned body. Prints a markdown table and writes data/eval.json.
"""

import argparse
import asyncio
import json
import math
import time
from dataclasses import astuple, replace
from pathlib import Path
from statistics import median

import numpy as np

from airo import meta_api
from airo.body import KEY, Body, cloud_radius_cm, ensemble
from airo.control import recentre
from airo.mission import Mission
from airo.probes import LIBRARY, halt, measure, predict, run, safe, zscore
from airo.rig import RigAbort
from airo.scientist import Scientist
from airo.sim import SimRig, world

FLOOR = (2.0, 1.4)
LEARN_POSE = (0.4, 0.7, 0.0)
# Far from the parcel, so the transit is long enough to watch: surprise is suppressed near the push pose.
DELIVER_POSE = (1.7, 0.75, math.pi)
CHANGE_AFTER_M = 0.3
STATION = "blue"
HELD_OUT = ["step_80", "pivot_l_60", "step_back_60"]
SIM_CHANGES = {
    "none": None,
    "weight": lambda b: replace(b, g_l=b.g_l * 0.65, g_r=b.g_r * 0.65, eta=b.eta * 0.7),
    "tow": lambda b: replace(b, g_l=b.g_l * 0.85, g_r=b.g_r * 0.85, lag=b.lag * 2),
    "slippery": lambda b: replace(b, g_l=b.g_l * 0.9, g_r=b.g_r * 0.9, eta=b.eta * 0.6),
    "flip_motor": lambda b: replace(b, g_l=-b.g_l),
}
MJ_CHANGES = {
    "none": None,
    "weight": lambda r: r.add_weight(),
    "tow": lambda r: r.attach_tow(),
    "slippery": lambda r: r.slippery_patch(),
    "flip_motor": lambda r: r.flip_motor("l"),
}


def make_rig(kind: str, seed: int, pose0):
    kw = dict(floor=FLOOR, objects=world(), pose0=pose0, log_dir=None, seed=seed)
    if kind == "sim":
        rig = SimRig(KEY, **kw)
    else:
        from airo.mjsim import MjRig

        rig = MjRig(bumper=True, **kw)
    meta_api.segment = rig.sam
    return rig


def change(rig, name: str):
    if isinstance(rig, SimRig):
        rig.truth = SIM_CHANGES[name](rig.truth)
    else:
        MJ_CHANGES[name](rig)


def scientist(rig) -> Scientist:
    sci = Scientist(rig, recentre=lambda r: recentre(r, sci.best))
    return sci


async def learn(kind: str, seed: int) -> tuple[dict, tuple[Body, list[Body]]]:
    rig = make_rig(kind, seed, LEARN_POSE)
    sci = scientist(rig)
    t0 = time.perf_counter()
    await sci.learn()
    row = {"seed": seed, "probes": len(sci.segments), "sim_s": rig.t, "wall_s": time.perf_counter() - t0,
           "cloud_cm": cloud_radius_cm(sci.segments, sci.best)}
    if isinstance(rig, SimRig):
        row["err_pct"] = {k: 100 * (v / t - 1) for k, v, t in zip(Body.__dataclass_fields__, astuple(sci.best),
                                                                   astuple(rig.truth)) if k != "kappa"}
        row["kappa_err"] = sci.best.kappa - rig.truth.kappa
    members = ensemble(sci.segments, sci.best, 200)
    z = []
    for name in HELD_OUT:
        p = LIBRARY[name]
        if not safe(rig, p, sci.best):
            await recentre(rig, sci.best)
        pred = predict(members, p, (rig.pose.x, rig.pose.y, rig.pose.theta))
        z += list(zscore(pred, measure(await run(rig, p))).values())
        await halt(rig)
    row["held_out_z"] = z
    await rig.close()
    return row, (sci.best, sci.members)


async def episode(kind: str, seed: int, name: str, learned) -> dict:
    rig = make_rig(kind, seed, DELIVER_POSE)
    sci = scientist(rig)
    sci.best, sci.members = learned
    sci.n, sci.name = 1, "learned"
    m = Mission(rig, sci)
    log, t_change, start = [], None, (rig.pose.x, rig.pose.y)

    async def watch():
        nonlocal t_change
        while True:
            if not log or log[-1][1] != m.status:
                log.append((rig.t, m.status))
            moved = math.dist((rig.pose.x, rig.pose.y), start)
            if name != "none" and t_change is None and m.status == "delivering" and moved > CHANGE_AFTER_M:
                change(rig, name)
                t_change = rig.t
            await asyncio.sleep(0)

    watcher = asyncio.create_task(watch())
    try:
        ok, aborted = await m.deliver(STATION), None
    except RigAbort as e:
        ok, aborted = False, str(e)
    finally:
        watcher.cancel()
    surprises = [t for t, s in log if s == "relearning"]
    before = [t for t in surprises if t_change is None or t < t_change]
    after = [t for t in surprises if t_change is not None and t >= t_change]
    relearn_s = [b - a for (a, s), (b, _) in zip(log, log[1:]) if s == "relearning"]
    alerts = [i for i, e in enumerate(sci.notebook) if e["kind"] == "alert" and e["text"].startswith("Body changed")]
    parcel = next(o.xy for o in rig.objects if o.concept == "small parcel")
    target = next(o.xy for o in rig.objects if o.concept == f"{STATION} paper")
    await rig.close()
    return {"seed": seed, "change": name, "changed_at": t_change, "false_triggers": len(before),
            "detected": bool(after), "latency_s": after[0] - t_change if after else None,
            "relearns": len(alerts), "relearn_s": relearn_s,
            "relearn_probes": sum(e["kind"] == "result" for e in sci.notebook[alerts[0]:]) if alerts else 0,
            "confirmed": sci.verified if alerts else None,
            "delivered": ok, "parcel_err_cm": 100 * math.dist(parcel, target), "aborted": aborted, "sim_s": rig.t}


def _med(xs, fmt="{:.1f}"):
    xs = [x for x in xs if x is not None]
    return fmt.format(median(xs)) if xs else "–"


def _rate(xs):
    xs = [x for x in xs if x is not None]
    return f"{sum(xs)}/{len(xs)}" if xs else "–"


def table(learned: list[dict], episodes: list[dict]) -> str:
    z = np.abs([r["held_out_z"] for r in learned])
    out = ["| learn | probes | sim s | compute s | cloud cm | held-out within 2σ | median |error| |",
           "|---|---|---|---|---|---|---|"]
    err = (", ".join(f"{k} {median(abs(r['err_pct'][k]) for r in learned):.0f}%" for k in learned[0]["err_pct"])
           + f", kappa ±{median(abs(r['kappa_err']) for r in learned):.3f}") if "err_pct" in learned[0] else "–"
    out.append(f"| {len(learned)} seeds | {_med([r['probes'] for r in learned], '{:.0f}')} | "
               f"{_med([r['sim_s'] for r in learned])} | {_med([r['wall_s'] for r in learned])} | "
               f"{_med([r['cloud_cm'] for r in learned])} | {np.mean(z < 2):.0%} | {err} |")
    out += ["", "| change | detected | false triggers | latency s | relearns | confirmed | relearn probes | "
                "relearn s | delivered | parcel error cm | aborted |", "|---|" * 1 + "---|" * 10]
    for name in dict.fromkeys(e["change"] for e in episodes):
        es = [e for e in episodes if e["change"] == name]
        out.append(f"| {name} | {_rate([e['detected'] for e in es]) if name != 'none' else '–'} | "
                   f"{sum(e['false_triggers'] for e in es)} | {_med([e['latency_s'] for e in es])} | "
                   f"{sum(e['relearns'] for e in es)} | {_rate([e['confirmed'] for e in es])} | "
                   f"{_med([e['relearn_probes'] or None for e in es], '{:.0f}')} | "
                   f"{_med([s for e in es for s in e['relearn_s']])} | {_rate([e['delivered'] for e in es])} | "
                   f"{_med([e['parcel_err_cm'] for e in es])} | {sum(bool(e['aborted']) for e in es)} |")
    return "\n".join(out)


async def main(args):
    async def offline(*_, **__):
        raise TimeoutError("eval runs the scripted scientist")

    meta_api.wonder = meta_api.describe = offline
    learned, episodes = [], []
    for seed in range(args.n):
        row, body = await learn(args.rig, seed)
        learned.append(row)
        for name in args.changes:
            episodes.append(await episode(args.rig, seed, name, body))
            e = episodes[-1]
            print(f"seed {seed} {name}: detected {e['detected']} after {e['latency_s']}, delivered {e['delivered']}"
                  f"{', aborted: ' + e['aborted'] if e['aborted'] else ''}", flush=True)
    print(f"\n{args.rig}, {args.n} seeds\n\n{table(learned, episodes)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rig": args.rig, "n": args.n, "learn": learned, "episodes": episodes},
                                         indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--rig", choices=("sim", "mujoco"), default="sim")
    p.add_argument("--changes", nargs="+", choices=list(SIM_CHANGES), default=list(SIM_CHANGES))
    p.add_argument("--out", default="data/eval.json")
    asyncio.run(main(p.parse_args()))

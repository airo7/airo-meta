"""Benchmark: seeded, headless episodes of learn, then deliver with a body change mid-transit, relearn and finish.

The scientist runs its scripted policy unless --wonder. Each seed learns once; every condition then delivers on a
fresh rig from the learned body. The recall condition instead patrols a loop through weight on, off and on again: it
should relearn the first change, then recall the bodies it has been. The carry condition (--rig mujoco --changes
carry) runs three robots on the factory, each fetching heavy and light totes, and grades every check_body against
the real load. Prints markdown tables, writes data/eval.json.
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
from airo.control import follow, recentre
from airo.memory import Memory
from airo.mission import Mission, Surprised, _Watched
from airo.planner import PARCEL, STATIONS
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
SIM_CHANGES |= {"loaded": None, "tow_loaded": SIM_CHANGES["tow"]}
MJ_CHANGES = {
    "none": None,
    "weight": lambda r: r.add_weight(),
    "tow": lambda r: r.attach_tow(),
    "slippery": lambda r: r.slippery_patch(),
    "flip_motor": lambda r: r.flip_motor("l"),
    "loaded": None,
    "tow_loaded": lambda r: r.attach_tow(),
}
# These start with the weight on and a body learned with it, as a car carrying a heavy tote.
LOADED = ("loaded", "tow_loaded")
PATROL = [(1.5, 0.55), (1.5, 0.9), (0.7, 0.9), (0.7, 0.55)]
LEGS = 4
RECALL_STAGES = [("weight on", "relearn", 2), ("weight off", "recall", 1), ("weight on", "recall", 2)]
CARRY = [[("heavy", "assembly"), ("light", "qa"), ("heavy", "packing"), ("light", "dock"), ("heavy", "qa"),
          ("light", "assembly")],
         [("heavy", "packing"), ("heavy", "dock"), ("light", "qa"), ("heavy", "assembly"), ("light", "packing"),
          ("light", "dock")],
         [("light", "dock"), ("heavy", "assembly"), ("light", "packing"), ("heavy", "qa"), ("heavy", "dock"),
          ("light", "qa")]]
# The carry world runs in real time, sped up: stepped as fast as possible, idle cars outpace a probing car and its
# commands are held longer than planned.
CARRY_SPEED = 4.0


def make_rig(kind: str, seed: int, pose0):
    """kind "factory" is one car of the carry fleet, with its tote tray, at the first robot's start."""
    kw = dict(floor=FLOOR, objects=world(), pose0=pose0, log_dir=None, seed=seed)
    if kind == "factory":
        from airo.mjsim import fleet
        from airo.site import FACTORY

        rig = fleet(FACTORY, 1, seed=seed, log_dir=None)[0]
    elif kind == "sim":
        rig = SimRig(KEY, **kw)
    else:
        from airo.mjsim import MjRig

        rig = MjRig(bumper=True, **kw)
    meta_api.segment = rig.sam
    return rig


def load(rig):
    if isinstance(rig, SimRig):
        rig.truth = SIM_CHANGES["weight"](rig.truth)
    else:
        rig.add_weight()


def change(rig, name: str):
    if isinstance(rig, SimRig):
        rig.truth = SIM_CHANGES[name](rig.truth)
    else:
        MJ_CHANGES[name](rig)


def scientist(rig, memory: Memory | None = None) -> Scientist:
    sci = Scientist(rig, recentre=lambda r: recentre(r, sci.best), memory=memory)
    return sci


async def learn(kind: str, seed: int, loaded: bool = False) -> tuple[dict, tuple[Body, list[Body]]]:
    rig = make_rig(kind, seed, LEARN_POSE)
    if loaded:
        load(rig)
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
    if name in LOADED:
        load(rig)
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
            if SIM_CHANGES[name] and t_change is None and m.status == "delivering" and moved > CHANGE_AFTER_M:
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
    parcel = next(o.xy for o in rig.objects if o.concept == PARCEL)
    target = next(o.xy for o in rig.objects if o.concept == STATIONS[STATION])
    await rig.close()
    return {"seed": seed, "change": name, "changed_at": t_change, "false_triggers": len(before),
            "detected": bool(after), "latency_s": after[0] - t_change if after else None,
            "relearns": len(alerts), "relearn_s": relearn_s,
            "relearn_probes": sum(e["kind"] == "result" for e in sci.notebook[alerts[0]:]) if alerts else 0,
            "confirmed": sci.verified if alerts else None,
            "delivered": ok, "parcel_err_cm": 100 * math.dist(parcel, target), "aborted": aborted, "sim_s": rig.t}


async def recall_episode(kind: str, seed: int, learned) -> list[dict]:
    rig = make_rig(kind, seed, (1.0, 0.7, 0.0))
    if not isinstance(rig, SimRig) and not hasattr(rig, "remove_weight"):
        print("recall: skipped, this MjRig has no remove_weight() yet")
        await rig.close()
        return []
    mem = Memory()
    mem.add("learned", "learned from scratch", *learned, True)
    sci = scientist(rig, mem)
    sci.adopt(mem.get(1))
    m = Mission(rig, sci)
    if isinstance(rig, SimRig):
        bare = rig.truth
        changes = {"weight on": lambda: setattr(rig, "truth", SIM_CHANGES["weight"](bare)),
                   "weight off": lambda: setattr(rig, "truth", bare)}
    else:
        changes = {"weight on": rig.add_weight, "weight off": rig.remove_weight}
    rows = []
    for step, (stage, expect, n) in enumerate(RECALL_STAGES):
        mark, start, t_change, latency = len(sci.notebook), (rig.pose.x, rig.pose.y), None, None
        m.relearned_s = None

        async def change():
            nonlocal t_change
            while math.dist((rig.pose.x, rig.pose.y), start) < CHANGE_AFTER_M:
                await asyncio.sleep(0)
            changes[stage]()
            t_change = rig.t

        task = asyncio.create_task(change())
        try:
            for _ in range(LEGS):
                await patrol(m)
        except Surprised as e:
            if t_change is not None:
                latency = rig.t - t_change
                await m.relearn(str(e))
        finally:
            task.cancel()
        new = sci.notebook[mark:]
        recalled = next((e["n"] for e in new if e["kind"] == "recall"), None)
        rows.append({"seed": seed, "step": step, "stage": stage, "detected": latency is not None,
                     "latency_s": latency, "recalled": recalled, "n": sci.n, "time_s": m.relearned_s,
                     "right": latency is not None and sci.n == n and (recalled is not None) == (expect == "recall"),
                     "probes": sum(e["kind"] == "result" for e in new), "verified": sci.verified})
        print(f"seed {seed} recall, {stage}: detected {latency is not None}, Body #{sci.n}"
              f"{' recalled' if recalled else ''} in {m.relearned_s}", flush=True)
        if latency is None:
            break
    await rig.close()
    return rows


async def patrol(m: Mission):
    """Three corners on from the nearest. follow stops once near its last point, so a full lap back to the start
    would end at once."""
    p = m.rig.pose
    i = min(range(len(PATROL)), key=lambda i: math.dist(PATROL[i], (p.x, p.y)))
    await follow(_Watched(m), m.body, [(p.x, p.y), *(PATROL[(i + k) % len(PATROL)] for k in (1, 2, 3))],
                 v_max=m.v_max)


async def carry_episode(seed: int, learned, busy: str = "normal") -> tuple[list[dict], dict]:
    """Three robots share one library seeded with the bare body. Each check_body is graded against the real load:
    the body it ends on must have been learned under the same load, a light tote and none counting as alike."""
    from airo.fleet import Fleet, crew
    from airo.mjsim import fleet
    from airo.site import FACTORY

    rigs = fleet(FACTORY, len(CARRY), seed=seed, realtime=True)
    rigs[0].world.speed = CARRY_SPEED
    meta_api.segment = rigs[0].sam
    pool = Memory()
    pool.add("bare chassis", "learned from scratch", *learned, True)
    load = {1: "bare"}
    rows = []

    def make(rig, robot):
        mem = pool.view(robot)
        sci = scientist(rig, mem)
        sci.adopt(mem.inherit())
        check = sci.check_body

        async def graded(cause: str) -> dict:
            truth = "bare" if cause == "unloaded" else cause.split()[2]
            before, mark = sci.n, len(sci.notebook)
            r = await check(cause)
            load.setdefault(r["n"], truth)
            alike = {load[r["n"]], truth} <= {"bare", "light"} or load[r["n"]] == truth
            shoved = any(_shoved(e) for e in sci.notebook[mark:] if e["kind"] == "result" and "measured" in e)
            rows.append({"seed": seed, "robot": robot, "cause": cause, "truth": truth, "from": before, **r,
                         "body": load[r["n"]], "right": alike, "disturbed": shoved})
            print(f"seed {seed} {robot} {cause}: {r['result']} #{before} -> #{r['n']} ({load[r['n']]}) in "
                  f"{r['seconds']:.1f} s{'' if alike else ', WRONG'}{', disturbed' if shoved else ''}", flush=True)
            return r

        sci.check_body = graded
        return sci

    f = Fleet(FACTORY, crew(FACTORY, rigs, make))
    if f.life is not None:
        f.life.set_level(busy)

    trail = {r.id: [] for r in f.robots}

    async def track():
        """Status changes with poses, for the report of a car that leaves the floor."""
        while True:
            for r in f.robots:
                p, t, status = r.rig.pose, trail[r.id], r.mission.status.split(":")[0]
                if p and (not t or t[-1][1] != status):
                    t.append((round(r.rig.world.t, 1), status, round(p.x, 2), round(p.y, 2)))
                    del t[:-6]
            await asyncio.sleep(0.05)

    async def deliveries(r, jobs):
        for kind, station in jobs:
            r.mission.inbox.put_nowait(f"bring a {kind} tote to {station}")
            while not r.mission.busy:
                await asyncio.sleep(0.01)
            while r.mission.busy:
                await asyncio.sleep(0.01)

    runs = [asyncio.create_task(r.mission.run()) for r in f.robots] + [asyncio.create_task(track())]
    try:
        await asyncio.gather(*(deliveries(r, jobs) for r, jobs in zip(f.robots, CARRY)))
    finally:
        for t in runs:
            t.cancel()
        for rig in rigs:
            await rig.close()
        rigs[0].world.close()
    done = {r.id: sum(n["text"].startswith("Delivered a") for n in r.mission.notes) for r in f.robots}
    relearns = sum(e["kind"] == "alert" and e["text"].startswith("Body changed") for r in f.robots
                   for e in r.sci.notebook)
    stops = {r.id: (n["text"], trail[r.id]) for r in f.robots for n in r.mission.notes
             if n["text"].startswith("Safety stop")}
    short = {r.id: [n["text"] for n in r.mission.notes if n["kind"] == "alert"][-2:] for r in f.robots
             if done[r.id] < len(CARRY[0]) and r.id not in stops}
    print(f"seed {seed} delivered {done}, {relearns} surprise relearns{f', stopped: {stops}' if stops else ''}"
          f"{f', short: {short}' if short else ''}", flush=True)
    return rows, {"seed": seed, "delivered": done, "stops": stops, "short": short, "relearns": relearns}


def _shoved(result: dict) -> bool:
    """A probe something pushed: a pivot turns in place (< 2 cm), and no probe slides 5 cm sideways."""
    m = result["measured"]
    return abs(m["lateral_cm"]) > 5 or (LIBRARY[result["probe"]].kind == "pivot" and abs(m["forward_cm"]) > 5)


def carry_table(rows: list[dict], runs: list[dict]) -> str:
    """Checks with a pushed probe are the cell's physics, not the scientist's, so they are counted apart; a car flung
    off the floor makes no more checks, so flings show in the deliveries."""
    clean = [r for r in rows if not r["disturbed"]]
    out = ["| load | checks | right body | same | recalled | learned | median s | max s |", "|---|---|---|---|---|---|---|---|"]
    for truth in ("heavy", "light", "bare"):
        rs = [r for r in clean if r["truth"] == truth]
        n = {k: sum(r["result"] == k for r in rs) for k in ("same", "recalled", "learned")}
        out.append(f"| {'unloaded' if truth == 'bare' else truth + ' tote'} | {len(rs)} | "
                   f"{_rate([r['right'] for r in rs])} | {n['same']} | {n['recalled']} | {n['learned']} | "
                   f"{_med([r['seconds'] for r in rs])} | {max((r['seconds'] for r in rs), default=0):.1f} |")
    jobs = len(runs) * sum(map(len, CARRY))
    hit = [r for r in rows if r["disturbed"]]
    out.append(f"\nChecks with a pushed probe, apart: {len(hit)}, {_rate([r['right'] for r in hit])} right. Delivered "
               f"{sum(sum(r['delivered'].values()) for r in runs)}/{jobs}; robots stopped after leaving the floor: "
               f"{sum(len(r['stops']) for r in runs)}; surprise relearns, all false here: "
               f"{sum(r.get('relearns', 0) for r in runs)}.")
    return "\n".join(out)


def recall_table(rows: list[dict]) -> str:
    out = ["| step | expect | detected | right body | latency s | time s | probes | confirmed |",
           "|---|---|---|---|---|---|---|---|"]
    for step, (stage, expect, n) in enumerate(RECALL_STAGES):
        rs = [r for r in rows if r["step"] == step]
        out.append(f"| {stage} | {expect} #{n} | {_rate([r['detected'] for r in rs])} | "
                   f"{_rate([r['right'] for r in rs])} | {_med([r['latency_s'] for r in rs])} | "
                   f"{_med([r['time_s'] for r in rs])} | {_med([r['probes'] for r in rs], '{:.0f}')} | "
                   f"{_rate([r['verified'] for r in rs])} |")
    return "\n".join(out)


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
    out += [] if not episodes else ["", "| change | detected | false triggers | latency s | relearns | confirmed | relearn probes | "
                "relearn s | delivered | parcel error cm | aborted |", "|---|" * 1 + "---|" * 10]
    for name in dict.fromkeys(e["change"] for e in episodes):
        es = [e for e in episodes if e["change"] == name]
        out.append(f"| {name} | {_rate([e['detected'] for e in es]) if SIM_CHANGES[name] else '–'} | "
                   f"{sum(e['false_triggers'] for e in es)} | {_med([e['latency_s'] for e in es])} | "
                   f"{sum(e['relearns'] for e in es)} | {_rate([e['confirmed'] for e in es])} | "
                   f"{_med([e['relearn_probes'] or None for e in es], '{:.0f}')} | "
                   f"{_med([s for e in es for s in e['relearn_s']])} | {_rate([e['delivered'] for e in es])} | "
                   f"{_med([e['parcel_err_cm'] for e in es])} | {sum(bool(e['aborted']) for e in es)} |")
    return "\n".join(out)


async def main(args):
    async def offline(*_, **__):
        raise TimeoutError("eval runs the scripted scientist")

    if not args.wonder:
        meta_api.wonder = meta_api.describe = offline
    learned, episodes, recalls, carries, carry_runs = [], [], [], [], []
    for seed in range(args.n):
        row, body = await learn(args.rig, seed)
        learned.append(row)
        if "recall" in args.changes:
            recalls += await recall_episode(args.rig, seed, body)
        if "carry" in args.changes and args.rig == "mujoco":
            rows, summary = await carry_episode(seed, (await learn("factory", seed))[1], args.busy)
            carries, carry_runs = carries + rows, carry_runs + [summary]
        loaded = (await learn(args.rig, seed, loaded=True))[1] if set(LOADED) & set(args.changes) else None
        for name in (c for c in args.changes if c not in ("recall", "carry")):
            episodes.append(await episode(args.rig, seed, name, loaded if name in LOADED else body))
            e = episodes[-1]
            print(f"seed {seed} {name}: detected {e['detected']} after {e['latency_s']}, delivered {e['delivered']}"
                  f"{', aborted: ' + e['aborted'] if e['aborted'] else ''}", flush=True)
    print(f"\n{args.rig}, {args.n} seeds\n\n{table(learned, episodes)}")
    if recalls:
        print(f"\n{recall_table(recalls)}")
    if carries:
        print(f"\n{carry_table(carries, carry_runs)}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"rig": args.rig, "n": args.n, "learn": learned, "episodes": episodes,
                                         "recall": recalls, "carry": carries,
                                         "carry_runs": carry_runs},
                                         indent=1, default=float))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--rig", choices=("sim", "mujoco"), default="sim")
    p.add_argument("--changes", nargs="+", choices=[*SIM_CHANGES, "recall", "carry"],
                   default=[*(c for c in SIM_CHANGES if c not in LOADED), "recall"])
    p.add_argument("--busy", choices=("calm", "normal", "busy"), default="normal",
                   help="how often warehouse events happen in the carry condition")
    p.add_argument("--wonder", action="store_true", help="live Muse Wonder and Spark; needs MODEL_API_KEY")
    p.add_argument("--out", default="data/eval.json")
    asyncio.run(main(p.parse_args()))

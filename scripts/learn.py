"""D3 gate: the car learns its body from probes, saves data/body.json and prints the fit next to the answer key."""

import argparse
import asyncio
import math
import sys
from dataclasses import astuple

import numpy as np

from airo.body import BODY_PATH, KEY, SIM_BODY_PATH, Body, cloud, ensemble, posterior, save
from airo.control import recentre
from airo.rig import RigAbort, add_args, open_rig
from airo.scientist import Scientist

UNITS = {"g_l": ("mm/s/PWM", 1000), "g_r": ("mm/s/PWM", 1000), "dz": ("PWM", 1), "eta": ("", 1),
         "tau": ("ms", 1000), "lag": ("ms", 1000), "kappa": ("rad/m", 1)}


def radius_cm(segments, best: Body) -> float:
    """Spread of where the posterior puts the car after 1.5 s of a gentle left arc from rest; 200 draws keep the
    Monte Carlo noise well under the changes between probes."""
    members = ensemble(segments, best, k=200)
    t = np.linspace(0, 1.5, 16)
    end = cloud(members, t, [0], [(100, 140)], (0, 0, 0), wheels0=(0, 0))[:, -1, :2]
    return 100 * math.sqrt(((end - end.mean(axis=0)) ** 2).sum(axis=1).mean())


def gates(b: Body, radii: list[float]) -> list[tuple[str, bool]]:
    return [("gains within 20% of 3.16 mm/s/PWM", all(abs(g / KEY.g_l - 1) <= 0.2 for g in (b.g_l, b.g_r))),
            ("dead zone within ±8 PWM", abs(b.dz - KEY.dz) <= 8),
            ("delay 150-250 ms", 0.15 <= b.tau <= 0.25),
            ("drift to the right (kappa < 0)", b.kappa < 0),
            ("cloud radius shrinks, at most one bump over 10%",
             len(radii) > 1 and radii[-1] < radii[0] and sum(b > 1.1 * a for a, b in zip(radii, radii[1:])) <= 1)]


async def main(args):
    rig = await open_rig(args, log_dir="data/runs")
    radii: list[float] = []
    seen, members = 0, None

    def on_update(sci):
        nonlocal seen, members
        for e in sci.notebook[seen:]:
            if e["kind"] == "result":
                print(f"{e['probe']:>14}  z {', '.join(f'{v:+5.1f}' for v in e['z'].values())}")
            elif e["kind"] in ("fallback", "alert", "conclusion"):
                print(f"[{e['kind']}] {e['text']}")
        seen = len(sci.notebook)
        if sci.members is not members and len(sci.segments):
            members = sci.members
            radii.append(radius_cm(sci.segments, sci.best))
            print(f"{'refit':>14}  {len(sci.segments)} probes, cloud {radii[-1]:.1f} cm")

    sci = Scientist(rig, recentre=lambda r: recentre(r, sci.best), k=args.k, on_update=on_update)
    members = sci.members
    radii.append(radius_cm([], sci.best))
    print(f"{'prior':>14}  cloud {radii[-1]:.1f} cm")
    try:
        await sci.learn(args.budget)
    finally:
        await rig.close()

    b = sci.best
    sd = np.sqrt(np.diag(posterior(sci.segments, b)))
    print(f"\n{'':>6} {'learned':>9} {'± 2σ':>8} {'key':>9} {'Δ%':>6}")
    for (name, (unit, k)), v, s, key in zip(UNITS.items(), astuple(b), sd, astuple(KEY)):
        print(f"{name:>6} {v * k:9.3g} {2 * s * k:8.2g} {key * k:9.3g} {100 * (v / key - 1):+6.0f}  {unit}")
    print()
    results = gates(b, radii)
    for what, ok in results:
        print(f"{'PASS' if ok else 'FAIL'}  {what}")
    out = args.out or (SIM_BODY_PATH if args.sim else BODY_PATH)
    save(b, sci.members, sci.name, out)
    print(f"\nsaved {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    p.add_argument("--budget", type=int, default=10)
    p.add_argument("-k", type=int, default=12, help="ensemble size")
    p.add_argument("--out", help="default data/body.json, or data/body_sim.json with --sim")
    args = p.parse_args()
    if args.dry:
        p.error("learning drives the car; use goto.py --dry to check the camera")
    try:
        asyncio.run(main(args))
    except RigAbort as e:
        sys.exit(f"airo: {e}")

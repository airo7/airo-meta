"""Pivots the car on the spot both ways and fits where its turning centre sits relative to the tracked point.

A marker taped off the turning centre swings on a circle while the car pivots, so every pivot looks like a slide.
Prints the --marker-offset to use; run it again with that offset and the residual should be under a centimetre.
"""

import argparse
import asyncio
import sys

import numpy as np

from airo.eye import MIN_TURN, fit, pivots, turns
from airo.rig import RigAbort, add_args, open_rig


async def main(args):
    rig = await open_rig(args)
    try:
        runs = await pivots(rig)
    finally:
        await rig.close()
    deg = turns(runs)
    if min(deg) < MIN_TURN:
        sys.exit(f"pivots turned only {deg[0]:.0f}° and {deg[1]:.0f}°: too little to fit a circle")
    d, rms = fit(runs)
    off = np.array(args.marker_offset) + d
    print(f"pivots {deg[0]:.0f}° and {deg[1]:.0f}°, circle fit rms {rms * 1000:.1f} mm")
    print(f"turning centre is {d[0] * 100:+.1f} cm forward, {d[1] * 100:+.1f} cm left of the tracked point")
    print(f"use --marker-offset {off[0]:.3f} {off[1]:.3f}" if np.hypot(*d) > 0.01 else "centred: no offset needed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    try:
        asyncio.run(main(p.parse_args()))
    except RigAbort as e:
        sys.exit(f"airo: {e}")

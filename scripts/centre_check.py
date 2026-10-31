"""Pivots the car on the spot both ways and fits where its turning centre sits relative to the tracked point.

A marker taped off the turning centre swings on a circle while the car pivots, so every pivot looks like a slide.
Prints the --marker-offset to use; run it again with that offset and the residual should be under a centimetre.
"""

import argparse
import asyncio
import math
import sys

import numpy as np

from airo.rig import RigAbort, add_args, open_rig

PWM, SECONDS, SETTLE = 100, 1.2, 0.6


def fit(runs: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Per-run fixed centre c_k and one body-frame offset d: tracked (x, y) = c_k - R(theta) d, linear least squares.
    Returns d and the rms residual in metres."""
    rows, rhs = [], []
    for k, run in enumerate(runs):
        for x, y, th in run:
            cs, sn = math.cos(th), math.sin(th)
            ex, ey = np.zeros(2 * len(runs)), np.zeros(2 * len(runs))
            ex[2 * k], ey[2 * k + 1] = 1, 1
            rows += [[*ex, -cs, sn], [*ey, -sn, -cs]]
            rhs += [x, y]
    a, b = np.array(rows), np.array(rhs)
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    return sol[-2:], float(np.sqrt(np.mean((a @ sol - b) ** 2)))


async def pivot(rig, pwm_l: int, pwm_r: int) -> np.ndarray:
    poses = []
    for i in range(round((SECONDS + SETTLE) * 20)):
        moving = i < SECONDS * 20
        p = await rig.step(pwm_l if moving else 0, pwm_r if moving else 0)
        if p:
            poses.append((p.x, p.y, p.theta))
    return np.array(poses)


async def main(args):
    rig = await open_rig(args)
    try:
        runs = [await pivot(rig, -PWM, PWM), await pivot(rig, PWM, -PWM)]
    finally:
        await rig.close()
    turns = [math.degrees(abs(np.unwrap(r[:, 2])[-1] - r[0, 2])) for r in runs]
    if min(turns) < 45:
        sys.exit(f"pivots turned only {turns[0]:.0f}° and {turns[1]:.0f}°: too little to fit a circle")
    d, rms = fit(runs)
    off = np.array(args.marker_offset) + d
    print(f"pivots {turns[0]:.0f}° and {turns[1]:.0f}°, circle fit rms {rms * 1000:.1f} mm")
    print(f"turning centre is {d[0] * 100:+.1f} cm forward, {d[1] * 100:+.1f} cm left of the tracked point")
    print(f"use --marker-offset {off[0]:.3f} {off[1]:.3f}" if np.hypot(*d) > 0.01 else "centred: no offset needed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    try:
        asyncio.run(main(p.parse_args()))
    except RigAbort as e:
        sys.exit(f"airo: {e}")

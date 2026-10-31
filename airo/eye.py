"""Where the car's turning centre sits relative to the tracked point: two pivots in place, opposite ways, and a fit.

A marker taped off the turning centre swings on a circle while the car pivots, so every pivot looks like a slide.
"""

import math

import numpy as np

from airo.probes import Probe, run, valid

PIVOTS = [Probe("eye_l", "pivot", -100, 100, 1.2), Probe("eye_r", "pivot", 100, -100, 1.2)]
MIN_TURN = 45.0


def fit(runs: list[np.ndarray]) -> tuple[np.ndarray, float]:
    """Per-run fixed centre c_k and one body-frame offset d: tracked (x, y) = c_k - R(theta) d, linear least squares.
    Returns d (forward, left) and the rms residual in metres."""
    rows, rhs = [], []
    for k, r in enumerate(runs):
        for x, y, th in r:
            cs, sn = math.cos(th), math.sin(th)
            ex, ey = np.zeros(2 * len(runs)), np.zeros(2 * len(runs))
            ex[2 * k], ey[2 * k + 1] = 1, 1
            rows += [[*ex, -cs, sn], [*ey, -sn, -cs]]
            rhs += [x, y]
    a, b = np.array(rows), np.array(rhs)
    sol = np.linalg.lstsq(a, b, rcond=None)[0]
    return sol[-2:], float(np.sqrt(np.mean((a @ sol - b) ** 2)))


def turns(runs: list[np.ndarray]) -> list[float]:
    return [math.degrees(abs(r[-1, 2] - r[0, 2])) for r in runs]


async def pivots(rig) -> list[np.ndarray]:
    """(x, y, unwrapped theta) rows of each pivot, settle included."""
    return [valid(await run(rig, p))[1] for p in PIVOTS]

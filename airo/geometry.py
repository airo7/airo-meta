"""Floor outlines: a car's footprint, rectangles, and whether two convex outlines overlap."""

import math

import numpy as np

CAR = (0.13, 0.08)


def rect(x: float, y: float, theta: float, hx: float, hy: float) -> np.ndarray:
    """Corners of a hx by hy half-size rectangle centred at (x, y), turned theta."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[x + a * hx * c - b * hy * s, y + a * hx * s + b * hy * c]
                     for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))])


def car(pose, pad: float = 0.0) -> np.ndarray:
    x, y, th = pose
    return rect(x, y, th, CAR[0] + pad, CAR[1] + pad)


def overlap(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether two convex outlines overlap: no edge normal of either separates them."""
    for poly in (a, b):
        for e in np.diff(np.vstack([poly, poly[:1]]), axis=0):
            n = (-e[1], e[0])
            if (a @ n).max() < (b @ n).min() or (b @ n).max() < (a @ n).min():
                return False
    return True

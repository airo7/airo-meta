"""Something new on the real floor: the overhead frame warped to a top-down floor image and compared with the empty
floor as first seen. The car is masked out; floor it uncovers joins the reference as it drives away."""

import math

import cv2
import numpy as np

from airo.planner import Scene

RES = 0.005
CAR_R = 0.2
DIFF = 28
MIN_AREA = 0.0025
# more than this much of the floor changed at once is the light or the phone's exposure, not an object
GLOBAL = 0.25
# floor cells around a changed patch to compare it with (about 6 cm)
RING = 25


class FloorWatch:
    def __init__(self, floor: tuple[float, float], res: float = RES):
        self.floor, self.res = floor, res
        self.shape = (math.ceil(floor[1] / res), math.ceil(floor[0] / res))
        self.ref: np.ndarray | None = None
        self.known = np.zeros(self.shape, bool)
        self.grid = np.zeros(self.shape, bool)
        self.dist = np.full(self.shape, np.inf)
        self.polygons: list[np.ndarray] = []
        ii, jj = np.mgrid[:self.shape[0], :self.shape[1]]
        self._xy = ((jj + 0.5) * res, (ii + 0.5) * res)

    def _topdown(self, frame: np.ndarray, H: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        small = cv2.resize(gray, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), 1.5)
        to_cells = np.array([[1 / self.res, 0, -0.5], [0, 1 / self.res, -0.5], [0, 0, 1]])
        M = to_cells @ H @ np.diag([2.0, 2.0, 1.0])
        return cv2.warpPerspective(small, M, self.shape[::-1], flags=cv2.INTER_LINEAR).astype(np.float32)

    def update(self, frame: np.ndarray, H: np.ndarray, pose) -> bool:
        """False when the frame was skipped (light change)."""
        cur = self._topdown(frame, H)
        car = np.hypot(self._xy[0] - pose.x, self._xy[1] - pose.y) < CAR_R
        if self.ref is None:
            self.ref = cur.copy()
        uncovered = ~car & ~self.known
        self.ref[uncovered] = cur[uncovered]
        self.known |= uncovered
        seen = self.known & ~car
        if not seen.any():
            return False
        gain = np.median(self.ref[seen]) / max(np.median(cur[seen]), 1.0)
        changed = (np.abs(cur * gain - self.ref) > DIFF) & seen
        if changed.mean() > GLOBAL:
            return False
        changed = cv2.morphologyEx(changed.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
        n, labels, stats, _ = cv2.connectedComponentsWithStats(changed)
        keep = [k for k in range(1, n) if stats[k, cv2.CC_STAT_AREA] * self.res ** 2 >= MIN_AREA
                and not self._gone(labels == k, cur * gain, seen & ~changed.astype(bool))]
        self.grid = np.isin(labels, keep)
        self.dist = cv2.distanceTransform((~self.grid).astype(np.uint8), cv2.DIST_L2, 3) * self.res \
            if keep else np.full(self.shape, np.inf)
        self.polygons = []
        for k in keep:
            contours, _ = cv2.findContours((labels == k).astype(np.uint8), cv2.RETR_EXTERNAL,
                                           cv2.CHAIN_APPROX_SIMPLE)
            c = cv2.convexHull(max(contours, key=cv2.contourArea))[:, 0]
            self.polygons.append((c + 0.5) * self.res)
        return True

    def _gone(self, blob: np.ndarray, cur: np.ndarray, floor: np.ndarray) -> bool:
        """Something in the reference that has since been taken away: the patch now looks like the floor around it but
        did not before. The reference heals there, so the empty spot is not kept as an obstacle."""
        ring = cv2.dilate(blob.astype(np.uint8), np.ones((RING, RING), np.uint8)).astype(bool) & ~blob & floor
        if ring.sum() < RING:
            return False
        now = abs(float(np.median(cur[blob]) - np.median(cur[ring])))
        then = abs(float(np.median(self.ref[blob]) - np.median(self.ref[ring])))
        if then > DIFF / 2 and now < then / 3:
            self.ref[blob] = cur[blob]
            return True
        return False

    def clearance(self, xy) -> float:
        i, j = int(xy[1] / self.res), int(xy[0] / self.res)
        if not (0 <= i < self.shape[0] and 0 <= j < self.shape[1]):
            return math.inf
        return float(self.dist[i, j])

    def ahead(self, path, reach: float, radius: float, ignore=None) -> tuple[float, float] | None:
        """The first point along path, within reach of its start, closer than radius to something on the floor.
        Nearness to the outline ignore (a box being pushed) does not count."""
        hull = None if ignore is None else np.asarray(ignore, np.float32)
        left = reach
        for a, b in zip(path, path[1:]):
            a, b = np.asarray(a, float), np.asarray(b, float)
            d = float(np.hypot(*(b - a)))
            for s in np.arange(0, min(d, left) + 1e-9, self.res * 4):
                p = a + (b - a) * (s / d if d else 0)
                c = self.clearance(p)
                pt = float(p[0]), float(p[1])
                if c < radius and (hull is None or -cv2.pointPolygonTest(hull, pt, True) > c + 4 * self.res):
                    return pt
            left -= d
            if left <= 0:
                break
        return None

    def scene(self, res: float = 0.02) -> Scene:
        k = round(res / self.res)
        rows, cols = self.shape[0] // k, self.shape[1] // k
        grid = self.grid[:rows * k, :cols * k].reshape(rows, k, cols, k).any(axis=(1, 3))
        return Scene(grid, res, {}, None, [("obstacle", p) for p in self.polygons])

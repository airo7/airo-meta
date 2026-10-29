"""Floor planner: SAM masks to a 2 cm occupancy grid in the floor frame, A* with clearance, push poses.

Grid cell (i, j) is centred at x = (j + 0.5) res, y = (i + 0.5) res, so rows run along +y.
"""

import asyncio
import heapq
import math
from dataclasses import dataclass

import cv2
import numpy as np

from airo import meta_api
from airo.camera import to_floor

STATIONS = {"red": "red rectangle", "blue": "blue rectangle"}
OBSTACLE, PARCEL = "box", "yellow box"
_STEPS = [(di, dj) for di in (-1, 0, 1) for dj in (-1, 0, 1) if di or dj]


@dataclass
class Scene:
    grid: np.ndarray
    res: float
    stations: dict[str, tuple[float, float]]
    parcel: tuple[float, float] | None
    polygons: list[tuple[str, np.ndarray]]


def warp_mask(mask: np.ndarray, H: np.ndarray, shape: tuple[int, int], res: float) -> np.ndarray:
    to_cells = np.array([[1 / res, 0, -0.5], [0, 1 / res, -0.5], [0, 0, 1]])
    return cv2.warpPerspective(mask.astype(np.uint8), to_cells @ H, shape[::-1], flags=cv2.INTER_NEAREST) > 0


def _outline(mask: np.ndarray, H: np.ndarray) -> np.ndarray | None:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return to_floor(H, cv2.approxPolyDP(max(contours, key=cv2.contourArea), 2, True).reshape(-1, 2))


def _centroid(poly: np.ndarray) -> tuple[float, float]:
    m = cv2.moments(poly.astype(np.float32))
    c = (m["m10"] / m["m00"], m["m01"] / m["m00"]) if m["m00"] else poly.mean(axis=0)
    return float(c[0]), float(c[1])


def _box_floor(s: meta_api.Segment, H: np.ndarray) -> tuple[float, float]:
    x0, y0, x1, y1 = s.box
    return tuple(to_floor(H, np.array([[(x0 + x1) / 2, (y0 + y1) / 2]], float))[0])


def _best(segs: list[meta_api.Segment]) -> meta_api.Segment | None:
    return max(segs, key=lambda s: s.score or 0.0, default=None)


async def scan(frame: np.ndarray, H: np.ndarray, floor_wh: tuple[float, float], obstacles: str = OBSTACLE,
               stations: dict[str, str] = STATIONS, parcel: str | None = PARCEL,
               near: tuple[float, float] | None = None, res: float = 0.02) -> Scene:
    """Labels: "obstacle" for each box, "parcel", and the station names. Obstacles and the parcel are occupied.
    near picks the parcel closest to that floor point, for a fleet whose totes look alike; the others are obstacles.
    parcel None looks for no parcel, for robots that carry rather than push."""
    concepts = (obstacles, *([parcel] if parcel else []), *stations.values())
    found = await asyncio.gather(*(meta_api.segment(frame, c) for c in concepts))
    obs, par, sta = found[0], found[1] if parcel else [], found[2 if parcel else 1:]
    p = _best(par) if near is None else min(par, key=lambda s: math.dist(_box_floor(s, H), near), default=None)
    obs += [s for s in par if s is not p] if near is not None else []
    # SAM's "box" also returns the parcel.
    obs = [s for s in obs if p is None or (s.mask & p.mask).sum() < 0.5 * min(s.mask.sum(), p.mask.sum())]
    labelled = [("obstacle", s) for s in obs] + [("parcel", s) for s in [p] if s]
    labelled += [(name, s) for name, segs in zip(stations, sta) if (s := _best(segs))]

    grid = np.zeros((math.ceil(floor_wh[1] / res), math.ceil(floor_wh[0] / res)), bool)
    polygons = []
    for label, s in labelled:
        if label in ("obstacle", "parcel"):
            grid |= warp_mask(s.mask, H, grid.shape, res)
        if (poly := _outline(s.mask, H)) is not None:
            polygons.append((label, poly))
    centres = {label: _centroid(p) for label, p in polygons if label != "obstacle"}
    return Scene(grid, res, {k: centres[k] for k in stations if k in centres}, centres.get("parcel"), polygons)


def occupy(scene: Scene, poly: np.ndarray, label: str):
    """Mark a floor polygon occupied, and outline it for the dashboard."""
    grid = scene.grid.astype(np.uint8)
    cv2.fillPoly(grid, [np.round(np.asarray(poly) / scene.res - 0.5).astype(np.int32)], 1)
    scene.grid = grid > 0
    scene.polygons.append((label, np.asarray(poly)))


def plan(scene: Scene, start, goal, clearance: float = 0.16, ignore=(),
         smooth: bool = True) -> list[tuple[float, float]] | None:
    """Waypoints in metres from start to goal, or None. `ignore` holds polygon labels or floor polygons."""
    res = scene.res
    occ = scene.grid.astype(np.uint8)
    for item in ignore:
        for poly in [p for lbl, p in scene.polygons if lbl == item] if isinstance(item, str) else [item]:
            cells = np.round(np.asarray(poly) / res - 0.5).astype(np.int32)
            cv2.fillPoly(occ, [cells], 0)
            # Outlines come from the image mask, the grid from the warp; they can disagree by a cell at the edge.
            cv2.polylines(occ, [cells], True, 0, 3)
    dist = cv2.distanceTransform(1 - occ, cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * res

    rows, cols = occ.shape
    s, g = [(min(max(int(p[1] / res), 0), rows - 1), min(max(int(p[0] / res), 0), cols - 1)) for p in (start, goal)]
    free = dist >= clearance
    ii, jj = np.mgrid[:rows, :cols]
    for c in (s, g):
        # The car may already sit inside the margin: near an endpoint, allow cells no closer to an obstacle than it.
        free |= ((ii - c[0]) ** 2 + (jj - c[1]) ** 2 <= (clearance / res) ** 2) & (dist >= dist[c])

    cells = _astar(free, s, g)
    if cells is None:
        return None
    if smooth:
        cells = _shortcut(free, cells)
    mid = [((j + 0.5) * res, (i + 0.5) * res) for i, j in cells[1:-1]]
    return [(float(start[0]), float(start[1])), *mid, (float(goal[0]), float(goal[1]))]


def _astar(free: np.ndarray, s: tuple[int, int], g: tuple[int, int]) -> list[tuple[int, int]] | None:
    rows, cols = free.shape
    ok = free.tolist()
    h = lambda c: max(abs(c[0] - g[0]), abs(c[1] - g[1])) + (math.sqrt(2) - 1) * min(abs(c[0] - g[0]), abs(c[1] - g[1]))
    cost, came, done = {s: 0.0}, {s: None}, set()
    heap = [(h(s), s)]
    while heap:
        _, c = heapq.heappop(heap)
        if c == g:
            path = []
            while c is not None:
                path.append(c)
                c = came[c]
            return path[::-1]
        if c in done:
            continue
        done.add(c)
        for di, dj in _STEPS:
            n = (c[0] + di, c[1] + dj)
            if not (0 <= n[0] < rows and 0 <= n[1] < cols and ok[n[0]][n[1]]):
                continue
            if di and dj and not (ok[c[0] + di][c[1]] and ok[c[0]][c[1] + dj]):
                continue
            nc = cost[c] + (math.sqrt(2) if di and dj else 1.0)
            if nc < cost.get(n, math.inf):
                cost[n], came[n] = nc, c
                heapq.heappush(heap, (nc + h(n), n))
    return None


def _visible(free: np.ndarray, a: tuple[int, int], b: tuple[int, int]) -> bool:
    t = np.linspace(0, 1, 2 * max(abs(b[0] - a[0]), abs(b[1] - a[1])) + 1)
    i, j = np.rint(np.outer(1 - t, a) + np.outer(t, b)).astype(int).T
    return bool(free[i, j].all())


def _shortcut(free: np.ndarray, cells: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out, k = [cells[0]], 0
    while k < len(cells) - 1:
        m = k + 1
        while m + 1 < len(cells) and _visible(free, cells[k], cells[m + 1]):
            m += 1
        out.append(cells[m])
        k = m
    return out


def push_pose(parcel, station, standoff: float = 0.22) -> tuple[float, float, float]:
    """Pose behind the parcel facing the station, `standoff` metres from the parcel centre."""
    th = math.atan2(station[1] - parcel[1], station[0] - parcel[0])
    return parcel[0] - standoff * math.cos(th), parcel[1] - standoff * math.sin(th), th

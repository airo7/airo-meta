"""Where people may walk in the hall: an occupancy grid of everything they must keep out of (guarded cells, the racks,
the conveyor, roll cages and goods, and where the robots park: bays and charging docks), planned on with
the floor planner's A*. Cars standing anywhere, the forklift and things dropped in the aisles are added per plan."""

import math

import cv2
import numpy as np

from airo import hall, planner
from airo.geometry import CAR, rect

RES = 0.03
ORIGIN = (-1.0, -1.0)
SIZE = (5.3, 4.0)
CLEARANCE = 0.05
CAR_PAD = 0.05
PARKING = (0.16, 0.11)
DOCK_HALF_WIDTH = 0.2


class WalkMap:
    def __init__(self, site, cells: dict):
        rows, cols = round(SIZE[1] / RES), round(SIZE[0] / RES)
        self.scene = planner.Scene(np.zeros((rows, cols), bool), RES, {}, None, [])
        block = lambda poly: planner.occupy(self.scene, np.asarray(poly) - ORIGIN, "static")
        for c in cells.values():
            pad = 0.02
            block([c.at(c.front - pad, -c.side - pad), c.at(c.front - pad, c.side + pad),
                   c.at(c.rear + pad, c.side + pad), c.at(c.rear + pad, -c.side - pad)])
        front = hall.rack_front(site) - 0.03
        block([(-1.0, front), (SIZE[0] + ORIGIN[0], front), (SIZE[0] + ORIGIN[0], 3.0), (-1.0, 3.0)])
        pick = cells["pick"]
        end = pick.at(0.27)
        block([(-1.0, end[1] - 0.1), (end[0], end[1] - 0.1), (end[0], end[1] + 0.1), (-1.0, end[1] + 0.1)])
        for x, y, th in hall.CAGES:
            block(rect(x, y, th, 0.12, 0.1))
        for x, y, th in hall.GOODS:
            block(rect(x, y, th, 0.12, 0.17))
        # bays and docks, not queue spots: those are in the driving lanes, and a car waiting there is in the way
        # like any other
        for b in (site.pick, *site.drops.values()):
            block(rect(*b.pose, *PARKING))
        for x, y, th in site.starts:
            # the dock and its charger behind it, wide enough to close the gaps between neighbouring docks
            block(rect(x - 0.1 * math.cos(th), y - 0.1 * math.sin(th), th, PARKING[0] + 0.1, DOCK_HALF_WIDTH))
        self.static = self.scene.grid.copy()

    def plan(self, start, goal, cars=(), things=()) -> list[tuple[float, float]] | None:
        """A walking path from start to goal around the static map, cars (x, y, theta) and things (outlines)."""
        self.scene.grid = self.static.copy()
        self.scene.polygons = []
        for x, y, th in cars:
            planner.occupy(self.scene, rect(x, y, th, CAR[0] + CAR_PAD, CAR[1] + CAR_PAD) - ORIGIN, "car")
        for poly in things:
            planner.occupy(self.scene, np.asarray(poly) - ORIGIN, "thing")
        # from or to a spot inside something (a goal by a dropped carton, a worker caught in a parking box): plan
        # between the nearest open spots and walk the last bit straight
        a, b = self._open(start), self._open(goal)
        path = planner.plan(self.scene, np.subtract(a, ORIGIN), np.subtract(b, ORIGIN), clearance=CLEARANCE)
        if path is None:
            return None
        path = [(x + ORIGIN[0], y + ORIGIN[1]) for x, y in path]
        return [tuple(start)] * (a != tuple(start)) + path + [tuple(goal)] * (b != tuple(goal))

    def _open(self, xy) -> tuple[float, float]:
        """xy if a person fits there, else the nearest spot where one does."""
        clear = cv2.distanceTransform((~self.scene.grid).astype(np.uint8), cv2.DIST_L2, 5) * RES > CLEARANCE + RES
        i, j = int((xy[1] - ORIGIN[1]) / RES), int((xy[0] - ORIGIN[0]) / RES)
        if 0 <= i < clear.shape[0] and 0 <= j < clear.shape[1] and clear[i, j]:
            return tuple(xy)
        ii, jj = np.nonzero(clear)
        k = np.argmin((ii - i) ** 2 + (jj - j) ** 2)
        return (jj[k] + 0.5) * RES + ORIGIN[0], (ii[k] + 0.5) * RES + ORIGIN[1]

    def free(self, xy) -> bool:
        i, j = (int((xy[1] - ORIGIN[1]) / RES), int((xy[0] - ORIGIN[0]) / RES))
        return 0 <= i < self.static.shape[0] and 0 <= j < self.static.shape[1] and not self.static[i, j]

"""Detour clearance from the car's own width and learned spread, with plain narration."""

import cv2
import numpy as np

from airo.body import Body, lateral_sd
from airo import planner
from airo.mission import HALF_W, choose_route, nearest_free
from airo.planner import Scene

RES = 0.02
BEST = Body(g_l=0.003, g_r=0.003, dz=25, eta=1.0, tau=0.2, lag=0.1, kappa=0.0)
# two blocks leave a 0.22 m gap at x in [0.9, 1.1], with room to go around above and below
WALL = [(0.9, 0.3, 1.1, 0.64), (0.9, 0.86, 1.1, 1.2)]
START, GOAL = (0.3, 0.75), (1.7, 0.75)


def scene(*rects):
    grid = np.zeros((75, 100), bool)
    for x0, y0, x1, y1 in rects:
        grid[round(y0 / RES):round(y1 / RES), round(x0 / RES):round(x1 / RES)] = True
    return Scene(grid, RES, {}, None, [])


def clearance_at(sc):
    dist = cv2.distanceTransform((~sc.grid).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * RES

    def at(p):
        i = min(max(int(p[1] / RES), 0), dist.shape[0] - 1)
        j = min(max(int(p[0] / RES), 0), dist.shape[1] - 1)
        return float(dist[i, j])

    return at


def spread(d):
    return [Body(g_l=0.003 + s * d, g_r=0.003 - s * d, dz=25, eta=1.0, tau=0.2, lag=0.1, kappa=0.0)
            for s in (-1, 1)]


def test_lateral_sd_grows_with_ensemble_spread():
    assert lateral_sd([BEST], BEST) == 0.0
    assert lateral_sd(spread(0.001), BEST) > lateral_sd(spread(0.0001), BEST) > 0


def test_through_the_gap_when_it_fits():
    sc = scene(*WALL)
    route, note = choose_route(sc, START, GOAL, 0.10, 0.03, None, clearance_at(sc))
    assert route is not None
    assert all(0.3 <= x[1] <= 1.2 for x in route)
    assert note is not None and "it fits." in note and "20 cm" in note


def test_around_with_the_long_way_note_when_it_does_not():
    sc = scene(*WALL)
    route, note = choose_route(sc, START, GOAL, 0.20, 0.03, None, clearance_at(sc))
    assert route is not None
    assert any(p[1] > 1.2 or p[1] < 0.3 for p in route)
    assert note is not None and "going the long way." in note
    assert "16 cm" in note and "40 cm" in note and f"{2 * HALF_W * 100:.0f}" in note


def test_waiting_note_when_nothing_fits():
    sc = scene((0.9, 0.0, 1.1, 0.64), (0.9, 0.86, 1.1, 1.5))
    route, note = choose_route(sc, START, GOAL, 0.20, 0.03, None, clearance_at(sc))
    assert route is None and note is not None and "waiting." in note


def test_a_box_on_the_destination_parks_beside_it():
    # the real floor's box on Nov 2: corner 2 at (0.65, 0.15) was 6 cm from it
    sc = scene((0.55, 0.20, 0.67, 0.34))
    at = clearance_at(sc)
    inside = lambda x, y: 0.15 <= x <= 0.65 and 0.15 <= y <= 1.07  # noqa: E731
    rows, cols = sc.grid.shape
    x, y = (np.arange(cols) + 0.5) * RES, (np.arange(rows) + 0.5) * RES
    allowed = ((y >= 0.15) & (y <= 1.07))[:, None] & ((x >= 0.15) & (x <= 0.65))[None, :]
    goal, need = (0.65, 0.15), 0.15
    assert planner.plan(sc, (0.38, 0.65), goal, clearance=need, allowed=allowed) is None
    spot = nearest_free(goal, need, at, inside)
    assert spot is not None and at(spot) >= need and inside(*spot)
    assert np.hypot(spot[0] - goal[0], spot[1] - goal[1]) < 0.3
    assert planner.plan(sc, (0.38, 0.65), spot, clearance=need, allowed=allowed) is not None
    assert nearest_free(goal, need, at, inside, reach=0.05) is None


def test_a_gap_under_need_is_not_called_a_fit():
    sc = scene(*WALL)
    route, note = choose_route(sc, START, GOAL, 0.10, 0.03, None, lambda p: 0.08)
    assert route is not None and "tight gap: 16 cm" in note and "it fits" not in note

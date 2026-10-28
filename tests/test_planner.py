import asyncio
import math

import cv2
import numpy as np
import pytest

from airo import meta_api
from airo.meta_api import Segment
from airo.planner import Scene, plan, push_pose, scan, warp_mask

RES = 0.02


def scene(*rects, parcel=None):
    """2 x 1.5 m floor with axis-aligned occupied rectangles (x0, y0, x1, y1) in metres."""
    grid = np.zeros((75, 100), bool)
    polygons = []
    for label, (x0, y0, x1, y1) in [("obstacle", r) for r in rects] + ([("parcel", parcel)] if parcel else []):
        grid[round(y0 / RES):round(y1 / RES), round(x0 / RES):round(x1 / RES)] = True
        polygons.append((label, np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])))
    return Scene(grid, RES, {}, None, polygons)


def clearance(sc, path):
    i, j = np.nonzero(sc.grid)
    occ = np.stack([(j + 0.5) * RES, (i + 0.5) * RES], 1)
    pts = np.concatenate([np.linspace(a, b, 50) for a, b in zip(path[:-1], path[1:])])
    return np.linalg.norm(pts[:, None] - occ[None], axis=2).min()


WALL = (0.98, 0.0, 1.02, 1.0)
BOX = (0.4, 0.9, 0.6, 1.1)


def test_plans_around_wall_and_box():
    sc = scene(WALL, BOX)
    path = plan(sc, (0.3, 0.3), (1.7, 0.3))
    assert path[0] == (0.3, 0.3) and path[-1] == (1.7, 0.3)
    assert max(y for _, y in path) > 1.0
    assert clearance(sc, path) >= 0.16 - RES


def test_unreachable_goal():
    assert plan(scene((0.98, 0.0, 1.02, 1.5)), (0.3, 0.3), (1.7, 0.3)) is None


def test_smoothing_reduces_waypoints():
    sc = scene(WALL, BOX)
    raw = plan(sc, (0.3, 0.3), (1.7, 0.3), smooth=False)
    smooth = plan(sc, (0.3, 0.3), (1.7, 0.3))
    assert len(smooth) < len(raw) / 5
    assert clearance(sc, smooth) >= 0.16 - RES


def test_start_inside_margin_and_ignored_parcel():
    sc = scene(BOX, parcel=(0.9, 0.0, 1.1, 1.5))
    assert plan(sc, (0.5, 0.8), (1.7, 0.3)) is None
    assert plan(sc, (0.5, 0.8), (1.7, 0.3), ignore=("parcel",)) is not None


def test_warp_mask_with_perspective_homography():
    floor_to_img = np.array([[400.0, 30, 100], [-20, -380, 650], [5e-5, -1e-4, 1]])
    rect = np.array([[0.5, 0.4], [1.2, 0.4], [1.2, 0.9], [0.5, 0.9]])
    img_pts = cv2.perspectiveTransform(rect[None], floor_to_img)[0]
    mask = np.zeros((720, 1280), np.uint8)
    cv2.fillPoly(mask, [np.round(img_pts).astype(np.int32)], 1)

    grid = warp_mask(mask > 0, np.linalg.inv(floor_to_img), (75, 100), RES)
    i, j = np.nonzero(grid)
    assert abs((j.min() + 0.5) * RES - 0.5) <= RES and abs((j.max() + 0.5) * RES - 1.2) <= RES
    assert abs((i.min() + 0.5) * RES - 0.4) <= RES and abs((i.max() + 0.5) * RES - 0.9) <= RES
    assert abs(grid.sum() - 35 * 25) < 0.1 * 35 * 25


def test_scan_with_fake_sam(monkeypatch):
    # 400 px per metre, image y down, floor y up.
    H = np.array([[1 / 400, 0, 0], [0, -1 / 400, 1.5], [0, 0, 1]])
    frame = np.zeros((600, 800, 3), np.uint8)

    def seg(x0, y0, x1, y1, score=0.9):
        m = np.zeros(frame.shape[:2], bool)
        m[600 - round(y1 * 400):600 - round(y0 * 400), round(x0 * 400):round(x1 * 400)] = True
        return Segment((0, 0, 0, 0), m, score)

    fake = {
        "cardboard box": [seg(0.8, 0.2, 1.0, 0.6), seg(1.4, 0.8, 1.6, 1.0)],
        "small parcel": [seg(0.3, 0.3, 0.4, 0.4, 0.4), seg(0.5, 1.0, 0.6, 1.1, 0.8)],
        "red paper": [seg(1.6, 0.1, 1.9, 0.5)],
        "blue paper": [],
    }

    async def fake_segment(image, concept):
        await asyncio.sleep(0)
        return fake[concept]

    monkeypatch.setattr(meta_api, "segment", fake_segment)
    sc = asyncio.run(scan(frame, H, (2.0, 1.5)))

    assert sc.grid.shape == (75, 100)
    assert sc.grid[20, 45] and sc.grid[45, 75] and sc.grid[52, 27]
    assert not sc.grid[15, 87] and not sc.grid[17, 17]
    assert sc.parcel == pytest.approx((0.55, 1.05), abs=0.01)
    assert sc.stations.keys() == {"red"} and sc.stations["red"] == pytest.approx((1.75, 0.3), abs=0.01)
    assert sorted(lbl for lbl, _ in sc.polygons) == ["obstacle", "obstacle", "parcel", "red"]


def test_push_pose():
    x, y, th = push_pose((1.0, 1.0), (2.0, 2.0))
    assert th == pytest.approx(math.pi / 4)
    assert math.hypot(x - 1.0, y - 1.0) == pytest.approx(0.22) and x < 1.0 and y < 1.0

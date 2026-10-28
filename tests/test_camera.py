import math

import cv2
import numpy as np
import pytest

from airo.camera import CAR_ID, CAR_MARKER_M, CORNER_MARKER_M, DICT, PoseTracker, floor_layout

W, H = 1920, 1080
FLOOR_TO_IMG = np.array([[700.0, 20, 160], [10, -700, 1000], [1e-5, 2e-5, 1]])
SIDE = 200


def stamp(canvas, marker_id, centre, size, theta=math.pi / 2):
    """Draw a marker whose top edge faces theta, mapping marker pixels to floor to image."""
    c, s = math.cos(theta - math.pi / 2), math.sin(theta - math.pi / 2)
    to_floor = np.array([[c * size / SIDE, s * size / SIDE, 0], [s * size / SIDE, -c * size / SIDE, 0], [0, 0, 1]])
    to_floor[:2, 2] = np.asarray(centre) - to_floor[:2, :2] @ [SIDE / 2, SIDE / 2]
    M = FLOOR_TO_IMG @ to_floor
    img = cv2.aruco.generateImageMarker(DICT, marker_id, SIDE)
    warped = cv2.warpPerspective(img, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=255)
    mask = cv2.warpPerspective(np.full_like(img, 255), M, (W, H)) > 0
    canvas[mask] = warped[mask]


@pytest.mark.parametrize("marker_h", [0.0, 0.1])
def test_pose_from_synthetic_overhead(marker_h):
    layout = floor_layout(2.0, 1.2)
    canvas = np.full((H, W), 255, np.uint8)
    for i, centre in layout.items():
        stamp(canvas, i, centre, CORNER_MARKER_M)

    tracker = PoseTracker(layout, marker_h=marker_h, cam_h=1.8)
    tracker.update(canvas)
    nadir = cv2.perspectiveTransform(np.array([[[W / 2, H / 2]]]), tracker.H).ravel()

    truth = np.array([1.0, 0.6])
    theta = math.radians(30)
    # An elevated marker appears pushed away from the nadir by 1 / k.
    seen = nadir + (truth - nadir) / tracker.k
    stamp(canvas, CAR_ID, seen, CAR_MARKER_M / tracker.k, theta)

    pose = tracker.update(canvas)
    assert tracker.residual_m < 0.002
    assert math.hypot(pose.x - truth[0], pose.y - truth[1]) < 0.005
    assert abs(pose.theta - theta) < math.radians(1)

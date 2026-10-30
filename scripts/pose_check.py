"""Day-1 gate: car pose at >=15 Hz with <=10 mm floor residual and jitter (keep the car still)."""

import argparse
import math
import os
import sys
import time
from collections import deque

import cv2
import numpy as np

from airo.camera import PoseTracker, floor_layout, open_camera, to_floor


def draw(frame, tracker: PoseTracker, pose):
    if tracker.H is None:
        return
    inv = np.linalg.inv(tracker.H)
    corners = np.array([v for v in tracker.layout.values()])
    cv2.polylines(frame, [to_floor(inv, corners).astype(np.int32)], True, (0, 255, 0), 2)
    if pose:
        tip = (pose.x + 0.15 * math.cos(pose.theta), pose.y + 0.15 * math.sin(pose.theta))
        a, b = to_floor(inv, np.array([[pose.x, pose.y], tip])).astype(int)
        cv2.arrowedLine(frame, tuple(a), tuple(b), (0, 0, 255), 4, tipLength=0.3)


def report(tracker: PoseTracker, poses) -> tuple[float, float, float]:
    xy = np.array([(p.x, p.y) for p in poses])
    jitter = float(np.linalg.norm(xy.std(axis=0))) * 1000
    heading = math.degrees(float(np.std(np.unwrap([p.theta for p in poses]))))
    p = poses[-1]
    print(f"{tracker.hz:5.1f} Hz  residual {tracker.residual_m * 1000:5.1f} mm  static jitter {jitter:5.1f} mm  "
          f"heading std {heading:4.2f} deg  pose ({p.x:.3f}, {p.y:.3f}, {math.degrees(p.theta):.1f})")
    return tracker.hz, tracker.residual_m * 1000, jitter


def main(args):
    src = args.video or args.camera
    cap = cv2.VideoCapture(src) if args.video else open_camera(src)
    tracker = PoseTracker(floor_layout(*args.floor), marker_h=args.marker_h, cam_h=args.cam_h)
    poses = deque(maxlen=60)
    start = last = time.monotonic()

    while cap.isOpened() and time.monotonic() - start < args.seconds:
        ok, frame = cap.read()
        if not ok:
            break
        pose = tracker.update(frame)
        if pose:
            poses.append(pose)
        if args.show:
            draw(frame, tracker, pose)
            cv2.imshow("airo pose", cv2.resize(frame, None, fx=0.5, fy=0.5))
            if cv2.waitKey(1) == 27:
                break
        if time.monotonic() - last >= 1 and len(poses) > 1:
            last = time.monotonic()
            report(tracker, poses)

    if len(poses) < 2:
        sys.exit("no pose: check that markers 0-4 are visible")
    hz, res, jitter = report(tracker, poses)
    ok = hz >= 15 and res <= 10 and jitter <= 10
    print("PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--camera", default=os.getenv("AIRO_CAMERA_INDEX", "0"))
    p.add_argument("--video")
    p.add_argument("--floor", nargs=2, type=float, required=True, metavar=("W", "H"),
                   help="corner marker centre distances in metres")
    p.add_argument("--marker-h", type=float, default=0.0, help="car marker height above floor (m)")
    p.add_argument("--cam-h", type=float, default=1.8, help="camera height above floor (m)")
    p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--show", action="store_true")
    main(p.parse_args())

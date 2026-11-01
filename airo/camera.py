"""Overhead camera to car pose in a metric floor frame, via ArUco markers.

Floor frame: x along corner markers 1 to 2, y along 1 to 4, all markers taped with their top edge toward +y.
"""

import math
import time
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np

DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
CAR_ID = 0
CAR_MARKER_M = 0.095
CORNER_MARKER_M = 0.12
# markers are found on a copy at most this wide: a 1080p phone stream then tracks at 720p cost and the dashboard
# still shows the full frame
DETECT_W = 1280
# ArUco corner order TL, TR, BR, BL as offsets in the floor frame (top edge toward +y).
_UNIT = np.array([[-0.5, 0.5], [0.5, 0.5], [0.5, -0.5], [-0.5, -0.5]])


@dataclass(frozen=True)
class Pose:
    x: float
    y: float
    theta: float
    t: float


def floor_layout(w_m: float, h_m: float) -> dict[int, tuple[float, float]]:
    """Corner marker centres in metres, measured centre to centre."""
    return {1: (0.0, 0.0), 2: (w_m, 0.0), 3: (w_m, h_m), 4: (0.0, h_m)}


def open_camera(source: int | str, width: int = 1920, height: int = 1080) -> cv2.VideoCapture:
    """A camera index, or a stream URL such as a phone running IP Webcam: http://<phone>:8080/video."""
    if isinstance(source, str) and not source.isdigit():
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap
    cap = cv2.VideoCapture(int(source), cv2.CAP_AVFOUNDATION)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


_params = cv2.aruco.DetectorParameters()
_params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
_detector = cv2.aruco.ArucoDetector(DICT, _params)


def detect(gray: np.ndarray) -> dict[int, np.ndarray]:
    corners, ids, _ = _detector.detectMarkers(gray)
    return {} if ids is None else {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def fit_floor(markers: dict[int, np.ndarray], layout: dict, size_m: float = CORNER_MARKER_M):
    """Image-to-floor homography from all 16 corner points; returns (H, rms residual in metres)."""
    ids = [i for i in layout if i in markers]
    if len(ids) < 4:
        return None, math.inf
    img = np.concatenate([markers[i] for i in ids]).astype(np.float64)
    flr = np.concatenate([np.array(layout[i]) + _UNIT * size_m for i in ids])
    H, _ = cv2.findHomography(img, flr)
    err = to_floor(H, img) - flr
    return H, float(np.sqrt((err ** 2).sum(axis=1).mean()))


def to_floor(H: np.ndarray, pts: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(pts.reshape(-1, 1, 2).astype(np.float64), H).reshape(-1, 2)


class PoseTracker:
    """Car pose from marker 0. The floor homography is a running median, so the car may occlude a corner.

    The car marker sits above the floor, so its floor projection is scaled away from the camera nadir by
    cam_h / (cam_h - marker_h); this assumes the phone looks roughly straight down. The pose is the car's turning
    centre, offset (forward, left) metres from the marker centre.
    """

    def __init__(self, layout: dict, marker_h: float = 0.0, cam_h: float = 1.8, window: int = 15,
                 offset: tuple[float, float] = (0.0, 0.0)):
        self.layout = layout
        self.k = 1 - marker_h / cam_h
        self.offset = offset
        self._hs: deque[np.ndarray] = deque(maxlen=window)
        self._ts: deque[float] = deque(maxlen=30)
        self.H: np.ndarray | None = None
        self.residual_m = math.inf

    def update(self, frame: np.ndarray, t: float | None = None) -> Pose | None:
        t = time.monotonic() if t is None else t
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        s = min(1.0, DETECT_W / gray.shape[1])
        small = gray if s == 1 else cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        markers = {i: c / s for i, c in detect(small).items()}

        H, res = fit_floor(markers, self.layout)
        if H is not None:
            self._hs.append(H / H[2, 2])
            self.H = np.median(self._hs, axis=0)
            self.residual_m = res
        if self.H is None or CAR_ID not in markers:
            return None

        c = to_floor(self.H, markers[CAR_ID])
        nadir = to_floor(self.H, np.array([[gray.shape[1] / 2, gray.shape[0] / 2]]))[0]
        c = nadir + (c - nadir) * self.k
        fwd = (c[0] + c[1]) / 2 - (c[2] + c[3]) / 2
        th = math.atan2(fwd[1], fwd[0])
        (a, b), cs, sn = self.offset, math.cos(th), math.sin(th)
        x, y = c.mean(axis=0) + (a * cs - b * sn, a * sn + b * cs)
        self._ts.append(t)
        return Pose(float(x), float(y), th, t)

    @property
    def hz(self) -> float:
        return (len(self._ts) - 1) / (self._ts[-1] - self._ts[0]) if len(self._ts) > 1 else 0.0

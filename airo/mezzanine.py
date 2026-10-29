"""The observation mezzanine in the hall's front-left corner, at 1:4: a steel deck over the walkway's start, a
stair down to the floor, a glass railing on its two open sides, and a small control room with screens against the
wall. From the railing at eye height the whole operation is in view: the pick cell, every station, the docks and
the lanes between them. Static, merged per material like the rest of the hall."""

import math

import cv2
import numpy as np

from airo.meshes import Batch
from airo.scene import texture

X0, X1 = -0.98, -0.42
Y0, Y1 = -1.2, 0.5
Z = 0.85
RAIL_H = 0.27
# where people stand at the railing, and where the "deck" camera's eye is: a person's eye height behind the rail
EYE = (-0.52, -0.97, Z + 0.42)
LOOK = (1.35, 1.02, 0.0)
SPOTS = ((-0.5, -0.12), (-0.5, -0.3), (-0.62, -0.05))
STAIR_X = 0.9


def build(spec):
    texture(spec, "screen", screen(), emission=0.9, specular=0.1)
    b = Batch("deck")
    t = 0.02
    b.box("deck_grating", ((X1 - X0) / 2, (Y1 - Y0) / 2, t), ((X0 + X1) / 2, (Y0 + Y1) / 2, Z - t))
    for x, y, hx, hy in (((X0 + X1) / 2, Y1, (X1 - X0) / 2, 0.004), (X1, (Y0 + Y1) / 2, 0.004, (Y1 - Y0) / 2)):
        b.box("hazard_y", (hx, hy, 0.012), (x, y, Z + 0.012))
        b.box("mezz_steel", (hx + 0.004, hy + 0.004, 0.03), (x, y, Z - 0.05))
    for y in np.linspace(Y0 + 0.02, Y1 - 0.02, 4):
        b.box("mezz_steel", ((X1 - X0) / 2, 0.012, 0.025), ((X0 + X1) / 2, y, Z - 0.065))
    for x, y in [(X1 - 0.02, y) for y in (Y0 + 0.02, -0.35, Y1 - 0.02)] + [(X0 + 0.03, Y1 - 0.02)]:
        b.box("mezz_steel", (0.016, 0.016, (Z - 0.04) / 2), (x, y, (Z - 0.04) / 2))
        b.box("mezz_steel", (0.03, 0.03, 0.004), (x, y, 0.004))
        b.box("hazard_y", (0.0165, 0.0165, 0.03), (x, y, 0.03))
    _rail(b, (X1, Y0 + 0.28), (X1, Y1))
    _rail(b, (X1, Y1), (X0, Y1))
    _stair(b)
    _control_room(b)
    b.emit(spec)


def _rail(b: Batch, a, z):
    """Glass panels between posts, a steel handrail on top."""
    a, z = np.array(a, float), np.array(z, float)
    length = float(np.linalg.norm(z - a))
    u = (z - a) / length
    yaw = math.atan2(u[1], u[0])
    k = max(1, round(length / 0.32))
    for i in range(k + 1):
        p = a + u * length * i / k
        b.box("rail_top", (0.0035, 0.0035, RAIL_H / 2), (*p, Z + RAIL_H / 2))
    for i in range(k):
        m = a + u * length * (i + 0.5) / k
        half = length / k / 2 - 0.008
        b.box("glass", (half, 0.0012, RAIL_H * 0.42), (*m, Z + 0.025 + RAIL_H * 0.42), yaw=yaw)
    b.box("rail_top", (length / 2 + 0.004, 0.0035, 0.0035), (*(a + z) / 2, Z + RAIL_H), yaw=yaw)


def _stair(b: Batch, width: float = 0.11):
    """A straight stair from the deck's front corner down to the floor, beside the walkway, with handrails."""
    y = Y0 + 0.14
    n = 19
    rise, run = Z / n, (STAIR_X - X1) / n
    for k in range(n):
        x = X1 + run * (k + 0.5)
        b.box("deck_grating", (run / 2 + 0.002, width, 0.006), (x, y, Z - rise * (k + 1) + 0.006))
        b.box("hazard_y", (0.003, width, 0.0065), (x + run / 2 - 0.003, y, Z - rise * (k + 1) + 0.006))
    slope = math.degrees(math.atan2(Z, STAIR_X - X1))
    mid = ((X1 + STAIR_X) / 2, Z / 2)
    length = math.hypot(STAIR_X - X1, Z) / 2
    for s in (-1, 1):
        b.box("mezz_steel", (length, 0.006, 0.02), (mid[0], y + s * (width + 0.006), mid[1] - 0.01),
              euler=(0, slope, 0))
        b.box("rail_top", (length, 0.006, 0.006), (mid[0], y + s * (width + 0.006), mid[1] + RAIL_H),
              euler=(0, slope, 0))
        for f in (0.15, 0.5, 0.85):
            x = X1 + f * (STAIR_X - X1)
            zz = Z * (1 - f)
            b.box("mezz_steel", (0.005, 0.005, RAIL_H / 2), (x, y + s * (width + 0.006), zz + RAIL_H / 2))
    b.box("mezz_steel", (0.016, 0.016, Z / 2), (X1 + 0.35 * (STAIR_X - X1), y, Z * 0.65 / 2))


def _control_room(b: Batch):
    """A glazed box against the wall: steel frame, roof, a desk facing the floor, screens above it."""
    x0, x1, y0, y1, h = X0 + 0.005, X0 + 0.27, Y1 - 0.48, Y1 - 0.02, 0.55
    top = Z + h
    for x, y in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
        b.box("mezz_steel", (0.008, 0.008, h / 2), (x, y, Z + h / 2))
    b.box("mezz_steel", ((x1 - x0) / 2 + 0.01, (y1 - y0) / 2 + 0.01, 0.012), ((x0 + x1) / 2, (y0 + y1) / 2, top))
    b.box("mezz_steel", ((x1 - x0) / 2, 0.006, 0.04), ((x0 + x1) / 2, y0, Z + 0.04))
    for y in (y0, y1):
        b.box("glass", ((x1 - x0) / 2 - 0.008, 0.0012, h / 2 - 0.06), ((x0 + x1) / 2, y, Z + h / 2 + 0.03))
    b.box("glass", (0.0012, (y1 - y0) / 2 - 0.008, h / 2 - 0.06), (x1, (y0 + y1) / 2, Z + h / 2 + 0.03))
    b.box("desk", (0.05, (y1 - y0) / 2 - 0.04, 0.006), (x1 - 0.07, (y0 + y1) / 2, Z + 0.19))
    b.box("mezz_steel", (0.04, (y1 - y0) / 2 - 0.06, 0.09), (x1 - 0.08, (y0 + y1) / 2, Z + 0.09))
    for k in range(3):
        y = y0 + 0.1 + k * 0.17
        b.box("arm_dark", (0.004, 0.065, 0.04), (x1 - 0.1, y, Z + 0.26), euler=(0, -12, 0))
        b.box("screen", (0.0042, 0.06, 0.035), (x1 - 0.096, y, Z + 0.26), euler=(0, -12, 0))
        b.box("arm_dark", (0.004, 0.004, 0.03), (x1 - 0.1, y, Z + 0.21))


def screen(px=(256, 160)) -> np.ndarray:
    """A fleet dashboard: the floor plan with lanes and dots, and a strip of status bars."""
    w, h = px
    img = np.full((h, w, 3), (40, 26, 18), np.uint8)
    cv2.rectangle(img, (12, 12), (170, 130), (90, 70, 50), 2)
    for (x0, y0, x1, y1) in ((20, 20, 50, 60), (70, 18, 110, 40), (120, 18, 160, 40), (130, 70, 162, 120)):
        cv2.rectangle(img, (x0, y0), (x1, y1), (120, 110, 90), -1)
    pts = np.array([[40, 90], [80, 70], [120, 60], [140, 95]], np.int32)
    cv2.polylines(img, [pts], False, (80, 200, 255), 2, cv2.LINE_AA)
    for (x, y), c in zip(((60, 105), (100, 64), (138, 92)), ((64, 179, 255), (255, 196, 69), (168, 111, 255))):
        cv2.circle(img, (x, y), 5, c, -1, cv2.LINE_AA)
    for k, c in enumerate(((90, 220, 120), (90, 220, 120), (40, 170, 250), (90, 220, 120))):
        cv2.rectangle(img, (185, 20 + 26 * k), (185 + 50 - 9 * k, 34 + 26 * k), c, -1)
    cv2.putText(img, "AIRO FLEET", (16, 152), cv2.FONT_HERSHEY_DUPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return img

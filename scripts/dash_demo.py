"""Dashboard with a fake 40 s episode: deliver, weight added, surprise, pivot probes, Body #2 -> #3, resume."""

import argparse
import asyncio
import math
import time
from collections import deque

import cv2
import numpy as np

from airo.camera import CAR_MARKER_M, CORNER_MARKER_M, DICT
from airo.server import Dashboard

W, H, PPM = 2.0, 1.4, 500
SPEED, CYCLE, K = 0.25, 40.0, 18
T_HIT, T_STOP, T_BODY, T_GO = 14.0, 17.0, 24.8, 26.0

PARAMS = [("Gain left", "mm/s/PWM", 0, 5), ("Gain right", "mm/s/PWM", 0, 5), ("Dead zone", "PWM", 0, 80),
          ("Pivot efficiency", "", 0, 1), ("Delay", "ms", 0, 400), ("Drift", "rad/m", 0, 0.15)]
BODIES = {
    2: ("Bare chassis", [3.21, 3.05, 41, 0.60, 196, 0.043], [3.16, 3.10, 38, 0.62, 190, 0.05]),
    3: ("Carrying a 400 g pack", [2.47, 2.41, 46, 0.55, 238, 0.047], [2.52, 2.45, 44, 0.57, 230, 0.05]),
}
NOTES = [
    (0.3, "hypothesis", "Wheels need a minimum PWM before they turn at all."),
    (0.5, "result", "Ramp probe: wheels start turning at 41 PWM.", dict(predicted=35, sd=6, measured=41, unit=" PWM")),
    (1.0, "conclusion", "Body #2 fitted from 9 probes. Cloud radius 3 cm at 1.5 s ahead."),
    (16.0, "alert", "Reality left the cloud: surprise 4.9 for three windows. Delivery paused."),
    (17.6, "hypothesis", "Something heavy was added. Both gains should drop and the delay should grow."),
    (18.4, "prediction", "Pivot at +100 PWM for 1 s. Body #2 says 96°\u00a0±\u00a07°; if I am heavier, under 80°."),
    (20.6, "result", "Measured 71°, z = −3.6 against Body #2.", dict(predicted=96, sd=7, measured=71, unit="°")),
    (21.0, "conclusion", "Heavier. Refitting gains and delay from the pivot."),
    (21.3, "hypothesis", "The refit body explains the turn, so it should predict the way back."),
    (21.6, "prediction", "Pivot back at −100 PWM. The refit says 70°\u00a0±\u00a04°."),
    (23.4, "result", "Measured 69°, inside the refit cloud.", dict(predicted=70, sd=4, measured=69, unit="°")),
    (24.8, "conclusion", "Heavier body confirmed: gains −22\u00a0%, delay +42 ms. Saved as Body #3."),
]
PHASES = [(-40, "probe", "Learning Body #2"), (0, "drive", "Delivering"), (15.5, "surprise", "Surprise"),
          (17, "probe", "Pivot +100"), (20.6, "refit", "Refit"), (21.4, "probe", "Pivot −100"),
          (23.4, "refit", "Refit"), (24.8, "relearn", "Body #3"), (26, "drive", "Delivering")]
CLOCK = 300
VIEW_WH, MARGIN = (1600, 900), 1.0


def make_loop(cx=1.0, cy=0.7, a=0.68, b=0.42, n=2000):
    phi = np.linspace(0, 2 * np.pi, n)
    pts = np.c_[cx + a * np.tanh(2 * np.cos(phi)) / np.tanh(2), cy + b * np.tanh(2 * np.sin(phi)) / np.tanh(2)]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))]
    return s, pts


LOOP_S, LOOP = make_loop()
PERIM = LOOP_S[-1]
GOALS = [("blue", PERIM / 4, (1.0, 1.22)), ("orange", 3 * PERIM / 4, (1.0, 0.16))]


def on_loop(s):
    s = np.asarray(s) % PERIM
    p = np.c_[np.interp(s, LOOP_S, LOOP[:, 0]), np.interp(s, LOOP_S, LOOP[:, 1])]
    d = np.c_[np.interp(s + 0.01, LOOP_S, LOOP[:, 0], period=PERIM), np.interp(s + 0.01, LOOP_S, LOOP[:, 1], period=PERIM)]
    t = d - np.c_[np.interp(s - 0.01, LOOP_S, LOOP[:, 0], period=PERIM), np.interp(s - 0.01, LOOP_S, LOOP[:, 1], period=PERIM)]
    t /= np.linalg.norm(t, axis=1, keepdims=True)
    return p, t, np.c_[t[:, 1], -t[:, 0]]


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def px(p):
    return np.array([p[0] * PPM, (H - p[1]) * PPM])


def rect(centre, size, deg=0.0):
    c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    half = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.array(size) / 2
    return half @ np.array([[c, s], [-s, c]]) + centre


def stamp(img, marker_id, corners_floor):
    """Warp an ArUco marker onto the floor image; corners are TL, TR, BR, BL in floor metres."""
    m = cv2.aruco.generateImageMarker(DICT, marker_id, 120)
    src = np.float32([[0, 0], [120, 0], [120, 120], [0, 120]])
    M = cv2.getPerspectiveTransform(src, np.float32([px(p) for p in corners_floor]))
    size = img.shape[1::-1]
    warped = cv2.warpPerspective(cv2.cvtColor(m, cv2.COLOR_GRAY2BGR), M, size)
    mask = cv2.warpPerspective(np.full_like(m, 255), M, size) > 0
    img[mask] = warped[mask]


OBJECTS = [  # label, centre, size, angle, BGR
    ("blue station", (1.0, 1.24), (0.297, 0.21), 0, (190, 110, 40)),
    ("orange station", (1.0, 0.14), (0.297, 0.21), 0, (40, 130, 235)),
    ("box", (1.02, 0.70), (0.30, 0.22), 18, (80, 125, 170)),
    ("box", (0.58, 0.66), (0.16, 0.16), -10, (90, 135, 180)),
    ("box", (1.80, 1.18), (0.22, 0.18), 6, (75, 120, 165)),
    ("parcel", (1.42, 0.86), (0.12, 0.12), 30, (110, 160, 205)),
]


def background(rng):
    noise = cv2.GaussianBlur(rng.normal(0, 1, (int(H * PPM), int(W * PPM))).astype(np.float32), (0, 0), 3)
    img = (np.array([168, 176, 182], np.float32) + noise[..., None] * 6).clip(0, 255).astype(np.uint8)
    for label, c, size, deg, bgr in OBJECTS:
        cv2.fillPoly(img, [np.int32([px(p) for p in rect(c, size, deg)])], bgr, cv2.LINE_AA)
        if label == "box":
            cv2.fillPoly(img, [np.int32([px(p) for p in rect(c, (size[0], 0.035), deg)])], (115, 165, 205), cv2.LINE_AA)
    for i, (x, y) in enumerate([(0, 0), (W, 0), (W, H), (0, H)], start=1):
        cv2.fillPoly(img, [np.int32([px(p) for p in rect((x, y), (0.17, 0.17))])], (235, 238, 240))
        stamp(img, i, rect((x, y), (CORNER_MARKER_M, CORNER_MARKER_M))[[3, 2, 1, 0]])
    return img


def draw_car(img, x, y, theta):
    deg = math.degrees(theta)
    for dx, dy in [(0.08, 0.1), (0.08, -0.1), (-0.08, 0.1), (-0.08, -0.1)]:
        c = (x + dx * math.cos(theta) - dy * math.sin(theta), y + dx * math.sin(theta) + dy * math.cos(theta))
        cv2.fillPoly(img, [np.int32([px(p) for p in rect(c, (0.065, 0.03), deg)])], (25, 25, 28), cv2.LINE_AA)
    cv2.fillPoly(img, [np.int32([px(p) for p in rect((x, y), (0.26, 0.17), deg)])], (55, 55, 60), cv2.LINE_AA)
    stamp(img, 0, rect((x, y), (CAR_MARKER_M, CAR_MARKER_M), deg - 90)[[3, 2, 1, 0]])


def world_bg():
    """Checker floor around the arena, in the same pixel scale as the top-down frame."""
    m = int(MARGIN * PPM)
    hh, ww = int(H * PPM) + 2 * m, int(W * PPM) + 2 * m
    yy, xx = np.mgrid[:hh, :ww] // (PPM // 4)
    img = np.where(((xx + yy) % 2 == 0)[..., None], (74, 50, 32), (92, 64, 42)).astype(np.uint8)
    yy, xx = np.mgrid[:VIEW_WH[1], :VIEW_WH[0]] / np.array(VIEW_WH[::-1])[:, None, None]
    shade = (0.35 + 0.65 * np.clip(yy * 1.6, 0, 1)) * (1 - 0.45 * ((xx - 0.5) ** 2 + (yy - 0.55) ** 2) * 2.2)
    return img, shade[..., None].astype(np.float32)


def chase_view(world, shade, target, az, dist=1.55, height=1.0, f=1250):
    """Pinhole camera orbiting the target; the floor plane maps to the view by a homography."""
    c = np.array([target[0] + dist * math.cos(az), target[1] + dist * math.sin(az), height])
    fwd = np.array([*target, 0.0]) - c
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    R = np.array([right, np.cross(fwd, right), fwd])
    K = np.array([[f, 0, VIEW_WH[0] / 2], [0, f, VIEW_WH[1] / 2], [0, 0, 1]])
    to_world_px = np.array([[PPM, 0, MARGIN * PPM], [0, -PPM, (H + MARGIN) * PPM], [0, 0, 1]])
    Hv = K @ np.c_[R[:, 0], R[:, 1], -R @ c] @ np.linalg.inv(to_world_px)
    view = cv2.warpPerspective(world, Hv, VIEW_WH, flags=cv2.INTER_LINEAR, borderValue=(40, 26, 16))
    return (view * shade).astype(np.uint8)


def draw_ghosts(img, poses):
    layer = img.copy()
    for x, y, th in poses:
        pts = np.int32([px(p) for p in rect((x, y), (0.26, 0.17), math.degrees(th))])
        cv2.fillPoly(layer, [pts], (255, 200, 150), cv2.LINE_AA)
        cv2.polylines(img, [pts], True, (255, 215, 175), 1, cv2.LINE_AA)
    cv2.addWeighted(layer, 0.22, img, 0.78, 0, img)


class Episode:
    def __init__(self, seed=3):
        self.rng = np.random.default_rng(seed)
        self.bg = background(self.rng)
        self.world, self.shade = world_bg()
        self.cam = None
        self.dv, self.dk = self.rng.normal(size=(2, K))
        self.polys = [{"label": lb, "points": (rect(c, np.add(sz, 0.03), deg) + self.rng.normal(0, 0.006, (4, 2))).round(3)}
                      for lb, c, sz, deg, _ in OBJECTS]
        self.surprise = deque((self.surprise_value(-1) for _ in range(100)), maxlen=160)
        self.next_window = 0.0
        self.goal = self.command = ""

    def pose(self, T):
        moved = min(T, T_HIT) + 0.7 * min(max(T - T_HIT, 0), T_STOP - T_HIT) + 0.8 * max(T - T_GO, 0)
        s = SPEED * moved + 0.35 * PERIM
        off = 0.13 * smooth((T - T_HIT) / (T_STOP - T_HIT)) * math.exp(-max(T - T_GO, 0) / 1.5)
        p, t, n = on_loop([s])
        xy = p[0] + n[0] * off
        pivot = math.radians(71) * (smooth(T - 19) - smooth(T - 22))
        heading = math.atan2(t[0, 1], t[0, 0])
        return s, off, xy, heading, heading + pivot

    def cloud(self, T, s, off, xy, heading):
        if T_STOP <= T < T_GO:
            mean, sd = (96, 7) if T < 20.6 else (-70, 4)
            ang = heading + math.radians((0 if T < 20.6 else 71) + mean) + np.radians(sd * self.dv)
            r = np.array([0.2, 0.36])
            return np.stack([xy[0] + np.outer(np.cos(ang), r), xy[1] + np.outer(np.sin(ang), r)], -1)
        sv, sk = (0.06, 0.16) if T < T_BODY else (0.04, 0.1)
        d = np.linspace(0, SPEED * 2, 16)
        out = []
        for dv, dk in zip(self.dv * sv, self.dk * sk):
            p, _, n = on_loop(s + d * (1 + dv))
            out.append(p + n * (off + dk * d[:, None] ** 2 * 2))
        return np.stack(out)

    def path(self, T, s):
        name, gs, station = min(GOALS, key=lambda g: (g[1] - s) % PERIM)
        self.goal = f"deliver the parcel to {name}"
        if T_STOP <= T < T_GO:
            return []
        p, _, _ = on_loop(np.linspace(s, s + (gs - s) % PERIM, 40))
        return np.vstack([p, station])

    def surprise_value(self, T):
        base = abs(self.rng.normal(0.9, 0.35))
        if T_HIT <= T < 18:
            base += 4.3 * smooth((T - T_HIT) / 2) * (1 if T < T_STOP else math.exp(-(T - T_STOP) * 1.5))
        return base

    def state(self, T):
        s, off, xy, heading, theta = self.pose(T)
        while self.next_window <= T:
            self.surprise.append(self.surprise_value(self.next_window))
            self.next_window += 0.5
        n = 3 if T >= T_BODY else 2
        name, values, keys = BODIES[n]
        status = ("surprised" if T >= 16 else "delivering") if T < T_STOP else "relearning" if T < T_GO else "delivering"
        path = self.path(T, s)
        cloud = self.cloud(T, s, off, xy, heading)
        frame = self.bg.copy()
        draw_car(frame, *xy, theta)
        return frame, self.view(frame, cloud, xy, T), {
            "t": CLOCK + T, "mode": "sim", "status": status, "command": self.command or self.goal,
            "floor": {"w": W, "h": H},
            "pose": {"x": xy[0], "y": xy[1], "theta": theta},
            "cloud": cloud,
            "path": path,
            "polygons": self.polys,
            "body": {"n": n, "name": name, "relearned_s": round(T_BODY - T_STOP, 1) if n == 3 else None,
                     "params": [{"name": nm, "value": v, "key": k, "unit": u, "lo": lo, "hi": hi}
                                for (nm, u, lo, hi), v, k in zip(PARAMS, values, keys)]},
            "surprise": {"values": list(self.surprise), "threshold": 3.0},
            "notebook": [{"t": CLOCK + t, "kind": k, "text": x, **(d[0] if d else {})} for t, k, x, *d in NOTES if t <= T],
            "episode": {"phases": [{"t": CLOCK + t, "kind": k, "label": lb} for t, k, lb in PHASES if t <= T]},
            "relearn": self.relearn(T),
        }

    def relearn(self, T):
        if 16 <= T < T_STOP:
            return {"phase": "changed", "elapsed_s": 0.0, "probe": "", "step": 0, "steps": 2}
        if T_STOP <= T < T_BODY:
            step = 1 if T < 20.6 else 2
            return {"phase": "testing", "elapsed_s": T - T_STOP, "probe": ["Pivot +100 PWM", "Pivot −100 PWM"][step - 1],
                    "step": step, "steps": 2}
        if T_BODY <= T < T_BODY + 4.5:
            return {"phase": "done", "elapsed_s": T_BODY - T_STOP, "probe": "", "step": 2, "steps": 2}
        return None

    def view(self, frame, cloud, xy, T):
        ends = cloud[:, -1] if T_STOP <= T < T_GO else cloud[:, len(cloud[0]) // 2]
        prev = cloud[:, -2] if T_STOP <= T < T_GO else cloud[:, len(cloud[0]) // 2 - 1]
        if T_STOP <= T < T_GO:
            prev = np.broadcast_to(xy, ends.shape)
        heads = np.arctan2(*(ends - prev).T[::-1])
        ghost = frame.copy()
        draw_ghosts(ghost, [(*xy, th) for th in heads] if T_STOP <= T < T_GO else np.c_[ends, heads])
        world = self.world.copy()
        m = int(MARGIN * PPM)
        world[m:m + ghost.shape[0], m:m + ghost.shape[1]] = ghost
        self.cam = np.array(xy) if self.cam is None else self.cam + 0.25 * (np.array(xy) - self.cam)
        return chase_view(world, self.shade, self.cam, -1.95 + 0.2 * math.sin(T / 7))


async def run(dash: Dashboard, start: float):
    t0 = time.monotonic() - start
    ep, last = Episode(), 0.0
    while True:
        T = (time.monotonic() - t0) % CYCLE
        if T < last:
            ep = Episode()
        last = T
        while not dash.commands.empty():
            ep.command = dash.commands.get_nowait()
        frame, view, state = ep.state(T)
        dash.set_frame(frame)
        dash.set_view(view)
        dash.publish(state)
        await asyncio.sleep(0.05)


async def main(args):
    dash = Dashboard()
    print(f"dashboard on http://{args.host}:{args.port}")
    await asyncio.gather(dash.serve(args.host, args.port), run(dash, args.start))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the episode")
    asyncio.run(main(ap.parse_args()))

"""Dashboard demo without MuJoCo: an 80 s factory episode against the v3 state.

Three robots deliver between Assembly, QA, Packing and the Dock. AIRO 1 gets a weight, relearns Body #2, loses it and
recalls #1, then gets it again and recalls #2 in about a second. AIRO 2 finds an oil spill and marks it for the fleet.
AIRO 3 recalls the body AIRO 1 shared. The Meta model feed is scripted, with thumbnails at /models/<id>.jpg, and the 3D
view is a small projective render that answers /camera like the twin does.
"""

import argparse
import asyncio
import json
import math
import time
from collections import deque
from itertools import pairwise

import cv2
import numpy as np
from fastapi.responses import Response

from airo.server import Dashboard

W, H, PPM, MARGIN = 6.0, 4.0, 160, 2.0
CYCLE, LANE, WINDOW, MEMBERS = 80.0, 0.12, 0.5, 10
VIEW_WH, FOCAL = (1600, 900), 1250
LIGHT = np.array([-0.35, -0.45, 0.82]) / np.linalg.norm([-0.35, -0.45, 0.82])

STATIONS = {"assembly": (0.45, 3.1), "qa": (5.55, 3.1), "packing": (5.55, 0.9), "dock": (0.45, 0.9)}
A, Q, P, D = STATIONS.values()
AISLES = [[(0.45, 3.1), (5.55, 3.1)], [(0.45, 0.9), (5.55, 0.9)], [(0.9, 0.9), (0.9, 3.1)], [(5.1, 0.9), (5.1, 3.1)],
          [(3.0, 0.9), (3.0, 3.1)]]
RACKS = [(2.0, 2.0), (4.0, 2.0)]
BENCHES = [((0.45, 3.68), (0.8, 0.4), 0.62), ((5.55, 3.68), (0.8, 0.4), 0.62), ((5.55, 0.32), (0.8, 0.4), 0.62),
           ((0.45, 0.3), (0.7, 0.45), 0.14)]
_a = np.linspace(0, 2 * np.pi, 28, endpoint=False)
_r = 1 + 0.12 * np.sin(3 * _a) + 0.07 * np.cos(5 * _a + 1)
SPILL = np.c_[3.0 + 0.42 * _r * np.cos(_a), 1.5 + 0.3 * _r * np.sin(_a)].round(3)

PARAMS = [("Gain left", "mm/s/PWM", 0, 5), ("Gain right", "mm/s/PWM", 0, 5), ("Dead zone", "PWM", 0, 80),
          ("Turn efficiency", "", 0, 1), ("Delay", "ms", 0, 400), ("Wheel lag", "ms", 0, 400),
          ("Drift", "rad/m", -0.3, 0.3)]
KEY = [3.16, 3.10, 38, 0.62, 190, 60, 0.05]
HEAVY = [2.47, 2.41, 46, 0.46, 238, 84, 0.047]
KIND_MODEL = {"segment": "sam-3.1", "describe": "muse-spark-1.3", "wonder": "muse-wonder-1.0",
              "transcribe": "muse-voice-transcribe-1.0", "image": "muse-image-1.0"}
PORTRAIT = "A studio self-portrait of a small four-wheeled robot car, {}, soft light, three-quarter view"
WORKING = ("surprised", "relearning", "rerouting")

# Each robot: legs ("drive", t0, t1, waypoints) or ("hold", t0, t1); probe pivots (t0, t1, deg) during holds;
# surprise bumps (rise, peak, settle, height); story (t, status, now, command); active body (t, n);
# relearn (t0, t1, phase, step, probe, start); memory portraits index FEED.
ROBOTS = {
    "airo-1": {
        "name": "AIRO 1", "color": "#64d2ff", "bare": [3.21, 3.05, 41, 0.60, 196, 58, 0.043],
        "legs": [("drive", 0, 10, [A, (2.2, 3.1)]), ("hold", 10, 21),
              ("drive", 21, 37, [(2.2, 3.1), (5.1, 3.1), (5.1, 0.9), P]), ("hold", 37, 40),
              ("drive", 40, 56, [P, D]), ("hold", 56, 60),
              ("drive", 60, 70, [D, (0.9, 0.9), (0.9, 3.1), A]), ("hold", 70, 80)],
        "pivots": [(11.4, 13.4, 71), (15.0, 16.6, -69), (18.2, 19.6, 33), (19.8, 20.8, -33)],
        "bumps": [(7.4, 9.4, 10.0, 4.9), (40.9, 42.6, 43.4, 3.6), (60.7, 62.2, 62.9, 3.8)],
        "weight": [(7.0, 39.0), (58.0, 99.0)], "cargo": [(-1, 37, "parcel"), (58, 70, "crate")],
        "story": [(0, "delivering", "Taking the parcel from Assembly to Packing.", "deliver to packing"),
               (7.6, "delivering", "Turns are coming up short. Watching the next window.", None),
               (9.2, "surprised", "That wasn't the body I know. Stopping to find out why.", None),
               (11.0, "relearning", "Testing a hypothesis: I'm heavier. Pivoting left at 100.", None),
               (15.0, "relearning", "The turn came up 25° short. Pivoting back to check the refit.", None),
               (18.2, "relearning", "One more turn to confirm the new body.", None),
               (20.5, "delivering", "I'm heavier now: Body #2. Back on the way to Packing.", None),
               (37.0, "unloading", "Parcel delivered at Packing. The weight pack comes off too.", None),
               (40.0, "delivering", "Heading to the Dock for the next crate.", "go to dock"),
               (42.4, "surprised", "Lighter than expected. Checking the bodies I remember.", None),
               (43.4, "delivering", "Recalled Body #1 in 0.9 s, no probes needed. On to the Dock.", None),
               (56.0, "loading", "Loading a heavy crate at the Dock.", None),
               (60.0, "delivering", "Taking the crate to Assembly.", "deliver to assembly"),
               (62.0, "surprised", "Heavy again. This feels like Body #2.", None),
               (62.9, "delivering", "Recalled Body #2 in 1.1 s. Carrying on to Assembly.", None),
               (70.0, "idle", "Crate delivered at Assembly. Ready for the next job.", None)],
        "bodies": [(-99, 1), (20.5, 2), (43.4, 1), (62.9, 2)],
        "relearn": [(9.2, 11.0, "changed", 0, "", 9.2), (11.0, 15.0, "testing", 1, "Pivot left 100", 9.2),
                 (15.0, 18.2, "testing", 2, "Pivot right 100", 9.2), (18.2, 20.5, "testing", 3, "Pivot left 60", 9.2),
                 (20.5, 24.5, "done", 3, "", 9.2), (42.4, 43.4, "changed", 0, "", 42.4),
                 (62.0, 62.9, "changed", 0, "", 62.0)],
        "notes": [(-40, "conclusion", "Body #1 fitted from 9 probes. Cloud radius 3 cm at 1.5 s ahead.", {}),
               (9.2, "alert", "Reality left the cloud: surprise 4.9 for three windows. Delivery paused.", {}),
               (9.8, "hypothesis", "A weight was added. Both gains should drop and turns should come up short.", {}),
               (11.0, "prediction", "Pivot left at 100 PWM for 1 s. Body #1 says 96° ± 7°; heavier, under 80°.",
                {"predicted": 96, "sd": 7, "unit": "°"}),
               (13.6, "result", "Measured 71°, z = −3.6 against Body #1.", {"predicted": 96, "sd": 7, "measured": 71, "unit": "°"}),
               (14.0, "conclusion", "Heavier. Refitting gains, delay and turn efficiency.", {}),
               (15.0, "prediction", "Pivot back at −100 PWM. The refit says 70° ± 4°.", {"predicted": 70, "sd": 4, "unit": "°"}),
               (16.8, "result", "Measured 69°, inside the refit cloud.", {"predicted": 70, "sd": 4, "measured": 69, "unit": "°"}),
               (18.2, "prediction", "Pivot left at 60 PWM. The refit says 31° ± 4°.", {"predicted": 31, "sd": 4, "unit": "°"}),
               (20.3, "result", "Measured 33°, inside the cloud.", {"predicted": 31, "sd": 4, "measured": 33, "unit": "°"}),
               (20.5, "conclusion", "Heavier body confirmed: gains −23 %, delay +42 ms. Saved as Body #2 and shared with the fleet.", {}),
               (42.4, "alert", "Surprise 3.6: lighter than Body #2.", {}),
               (43.4, "recall", "Body #1 explains the last window: predicted 42 cm, measured 41 cm.",
                {"n": 1, "name": "bare", "predicted": 42, "sd": 3, "measured": 41, "unit": " cm", "z": -0.3, "recall_s": 0.9}),
               (62.0, "alert", "Surprise 3.8: heavier than Body #1.", {}),
               (62.9, "recall", "Body #2 explains the last window: predicted 27 cm, measured 26 cm.",
                {"n": 2, "name": "heavier", "predicted": 27, "sd": 2, "measured": 26, "unit": " cm", "z": -0.4, "recall_s": 1.1})],
        "memory": [{"t": -99, "n": 1, "name": "bare", "cause": "First learn, 9 probes", "portrait": 0, "recalls": [43.4]},
                {"t": 20.5, "n": 2, "name": "heavier", "cause": "400 g weight pack", "portrait": 13, "recalls": [62.9]}],
    },
    "airo-2": {
        "name": "AIRO 2", "color": "#bf5af2", "bare": [3.08, 3.17, 38, 0.63, 184, 66, -0.021],
        "legs": [("hold", 0, 4), ("drive", 4, 12, [Q, (3.0, 3.1)]), ("drive", 12, 20, [(3.0, 3.1), (3.0, 1.78)]),
              ("hold", 20, 29), ("drive", 29, 33, [(3.0, 1.78), (3.0, 3.1)]),
              ("drive", 33, 47, [(3.0, 3.1), (0.9, 3.1), (0.9, 0.9), D]), ("hold", 47, 52),
              ("drive", 52, 72, [D, (5.1, 0.9), (5.1, 3.1), Q]), ("hold", 72, 80)],
        "plans": [(4, 27, [(3.0, 3.1), (3.0, 0.9), D])],
        "pivots": [(21.0, 22.2, 8), (22.4, 23.4, -8)],
        "bumps": [(18.6, 19.3, 20.0, 5.6)],
        "weight": [], "cargo": [(2.6, 47, "crate")],
        "story": [(0, "idle", "Waiting at QA for the next job.", ""),
               (2.6, "delivering", "Taking the crate from QA to the Dock, through the centre aisle.", "deliver to dock"),
               (19.2, "surprised", "The wheels are slipping. Is it my body or the floor?", None),
               (20.6, "relearning", "Testing whether the floor changed rather than my body.", None),
               (27.0, "rerouting", "This patch is an oil spill. Marked it for the fleet; rerouting.", None),
               (29.0, "delivering", "Going around the oil spill to the Dock.", None),
               (47.0, "unloading", "Crate delivered at the Dock.", None),
               (52.0, "delivering", "Returning to QA along the south aisle.", "go to qa"),
               (72.0, "idle", "Back at QA. Ready for the next job.", None)],
        "bodies": [(-99, 1)],
        "relearn": [(19.2, 20.6, "changed", 0, "", 19.2), (20.6, 23.4, "testing", 1, "Step back 80", 19.2),
                 (23.4, 27.0, "testing", 2, "Step 80 on the patch", 19.2)],
        "notes": [(-40, "conclusion", "Body #1 fitted from 8 probes. Cloud radius 3 cm at 1.5 s ahead.", {}),
               (19.4, "alert", "Reality left the cloud: surprise 5.6, sliding sideways.", {}),
               (20.6, "hypothesis", "If my body changed, a step on clean floor misses too. If the floor changed, only this patch does.", {}),
               (21.2, "prediction", "Step 80 PWM back on clean floor. Body #1 says 21 cm ± 2.",
                {"predicted": 21, "sd": 2, "unit": " cm"}),
               (23.2, "result", "Measured 21 cm. My body is unchanged.", {"predicted": 21, "sd": 2, "measured": 21, "unit": " cm"}),
               (23.6, "prediction", "Step 80 PWM on the patch. Body #1 says 21 cm ± 2.", {"predicted": 21, "sd": 2, "unit": " cm"}),
               (25.8, "result", "Measured 13 cm and drifting: z = −4.1.", {"predicted": 21, "sd": 2, "measured": 13, "unit": " cm"}),
               (27.0, "conclusion", "A place change, not a body change: an oil spill about 0.8 m across. Marked for the fleet.", {})],
        "memory": [{"t": -99, "n": 1, "name": "bare", "cause": "First learn, 8 probes", "portrait": 1, "recalls": []},
                {"t": 20.5, "n": 2, "name": "heavier", "cause": "400 g weight pack", "portrait": 13, "recalls": [], "src": "airo-1"}],
    },
    "airo-3": {
        "name": "AIRO 3", "color": "#ff6482", "bare": [3.30, 3.12, 44, 0.57, 205, 61, 0.012],
        "legs": [("hold", 0, 5), ("drive", 5, 22, [P, D]), ("hold", 22, 28),
              ("drive", 28, 37, [D, (0.9, 0.9), (0.9, 3.1), A]), ("hold", 37, 44), ("drive", 44, 60, [A, Q]),
              ("hold", 60, 64), ("drive", 64, 74, [Q, (5.1, 3.1), (5.1, 0.9), P]), ("hold", 74, 80)],
        "pivots": [], "bumps": [(45.6, 47.2, 47.8, 3.9), (60.6, 61.2, 61.5, 2.6)],
        "weight": [(42.0, 61.0)], "cargo": [(3, 22, "parcel"), (42, 60, "crate")],
        "story": [(0, "loading", "Loading parcels at Packing.", ""),
               (5.0, "delivering", "Taking parcels to the Dock along the south aisle.", "deliver to dock"),
               (22.0, "unloading", "Parcels delivered at the Dock.", None),
               (27.2, "idle", "AIRO 2 found an oil spill in the centre aisle. My routes avoid it now.", None),
               (28.0, "delivering", "Heading to Assembly for a heavy crate.", "go to assembly"),
               (37.0, "loading", "Loading a heavy crate at Assembly.", None),
               (44.0, "delivering", "Taking the crate to QA.", "deliver to qa"),
               (46.8, "surprised", "Heavier than Body #1. AIRO 1 shared a body like this.", None),
               (47.8, "delivering", "Recalled AIRO 1's heavier body in 1.0 s and verified it here.", None),
               (60.0, "unloading", "Crate delivered at QA.", None),
               (64.0, "delivering", "Returning to Packing.", "go to packing"),
               (74.0, "idle", "Back at Packing. Ready for the next job.", None)],
        "bodies": [(-99, 1), (47.8, 2), (61.5, 1)],
        "relearn": [(46.8, 47.8, "changed", 0, "", 46.8)],
        "notes": [(-40, "conclusion", "Body #1 fitted from 9 probes. Cloud radius 2 cm at 1.5 s ahead.", {}),
               (27.0, "conclusion", "Fleet update: AIRO 2 marked an oil spill in the centre aisle. Planning around it.", {}),
               (46.9, "alert", "Surprise 3.9: heavier than Body #1.", {}),
               (47.8, "recall", "AIRO 1's Body #2 explains the window: predicted 24 cm, measured 25 cm. Verified on this robot.",
                {"n": 2, "name": "heavier", "predicted": 24, "sd": 2, "measured": 25, "unit": " cm", "z": 0.4, "recall_s": 1.0}),
               (61.5, "recall", "Crate off. Body #1 explains the window again.",
                {"n": 1, "name": "bare", "predicted": 30, "sd": 2, "measured": 30, "unit": " cm", "z": 0.1, "recall_s": 0.8})],
        "memory": [{"t": -99, "n": 1, "name": "bare", "cause": "First learn, 9 probes", "portrait": 2, "recalls": [61.5]},
                {"t": 20.5, "n": 2, "name": "heavier", "cause": "400 g weight pack", "portrait": 13, "recalls": [47.8], "src": "airo-1",
                     "verified": 47.8}],
    },
}

# t, robot, kind, input, output, ms, ok, image, tools [(name, args, result)]
FEED = [
    (-70, "airo-1", "image", PORTRAIT.format("bare chassis"), "Body #1 · bare", 6400, True, ("portrait", "airo-1", False), None),
    (-68, "airo-2", "image", PORTRAIT.format("bare chassis"), "Body #1 · bare", 6900, True, ("portrait", "airo-2", False), None),
    (-66, "airo-3", "image", PORTRAIT.format("bare chassis"), "Body #1 · bare", 6100, True, ("portrait", "airo-3", False), None),
    (-48, "airo-1", "wonder", "Nine probes in. Is the fit good enough to drive?",
     "Yes. Every held-out probe landed inside 2σ. Naming it Body #1, bare.", 1800, True, None,
     [("refit", {}, "cloud 3 cm at 1.5 s"), ("conclude", {"name": "bare"}, "Body #1")]),
    (1.0, "airo-3", "segment", "parcel", "3 parcels, mean score 0.91", 820, True, ("frame", "airo-3", 4.0, "cargo"), None),
    (2.6, "airo-2", "transcribe", "microphone", "AIRO 2, take the crate from QA to the dock.", 640, True, None, None),
    (3.4, "airo-2", "segment", "crate", "1 crate, score 0.93", 760, True, ("frame", "airo-2", 3.4, "cargo"), None),
    (9.6, "airo-1", "describe", "What changed on the car?", "A grey weight pack is strapped to the top deck of the car.",
     1100, True, ("frame", "airo-1", 9.6, None), None),
    (10.4, "airo-1", "wonder", "Surprise 4.9 over three windows while turning. Spark: a grey weight pack on the top deck.",
     "I think I'm heavier. Both gains should drop and pivots should come up short. A pivot will tell.", 2400, True, None,
     [("get_body", {}, "Body #1 · bare"), ("predict", {"probe": "pivot_l_100"}, "96° ± 7"),
      ("run_probe", {"probe": "pivot_l_100"}, "71°, z −3.6")]),
    (14.2, "airo-1", "wonder", "The pivot came up 25° short.", "Refitting. If the refit is right it predicts the way back too.",
     1900, True, None, [("refit", {}, "gains −23 %, delay +42 ms"), ("run_probe", {"probe": "pivot_r_100"}, "69°, z −0.2")]),
    (20.3, "airo-1", "wonder", "Two probes inside the refit cloud.",
     "Confirmed. Naming it Body #2, heavier, and sharing it with the fleet.", 1600, True, None,
     [("run_probe", {"probe": "pivot_l_60"}, "33°, z +0.5"), ("conclude", {"name": "heavier"}, "Body #2")]),
    (21.8, "airo-2", "describe", "What is under the car?", "A dark, glossy puddle spreads across the centre aisle.",
     1200, True, ("frame", "airo-2", 21.8, None), None),
    (22.6, "airo-2", "wonder", "Surprise 5.6, sliding sideways. Spark: a glossy puddle under the car.",
     "Body or floor? One step on clean floor and one on the patch will tell.", 2100, True, None,
     [("run_probe", {"probe": "step_back_80"}, "21 cm, z +0.1"), ("run_probe", {"probe": "step_80"}, "13 cm, z −4.1")]),
    (23.5, "airo-1", "image", PORTRAIT.format("carrying a grey weight pack on its top deck"), "Body #2 · heavier",
     6800, True, ("portrait", "airo-1", True), None),
    (26.2, "airo-2", "segment", "oil spill", "1 region, 0.31 m², score 0.89", 910, True, ("frame", "airo-2", 26.2, "spill"), None),
    (27.0, "airo-2", "wonder", "Clean floor fits Body #1; the patch does not.",
     "A place change, not a body change. Marking the spill as a zone for the whole fleet.", 1500, True, None,
     [("mark_zone", {"label": "oil spill", "kind": "slippery"}, "zone 1")]),
    (31.5, "airo-3", "describe", "Is the Dock clear?", "Timed out after 8 s", 8000, False, None, None),
    (36.4, "airo-1", "segment", "packing station", "1 station, score 0.96", 700, True, ("frame", "airo-1", 36.4, "station"), None),
    (39.6, "airo-1", "transcribe", "microphone", "AIRO 1, head to the dock next.", 590, True, None, None),
    (42.8, "airo-1", "wonder", "Surprise 3.6: lighter than Body #2.", "Checking memory first: Body #1 predicts the last window best.",
     700, True, None, [("recall", {"bodies": [1, 2]}, "Body #1, z −0.3")]),
    (46.0, "airo-3", "segment", "crate", "1 crate, score 0.92", 780, True, ("frame", "airo-3", 46.0, "cargo"), None),
    (47.2, "airo-3", "wonder", "Surprise 3.9: heavier than Body #1.",
     "AIRO 1's Body #2 fits this window. Recalling it instead of relearning.", 800, True, None,
     [("recall", {"bodies": [1, 2]}, "Body #2 from airo-1, z +0.4")]),
    (57.0, "airo-1", "segment", "crate", "1 crate, score 0.94", 690, True, ("frame", "airo-1", 58.5, "cargo"), None),
    (62.4, "airo-1", "wonder", "Surprise 3.8: heavier than Body #1.", "Body #2 again. Recalling it; no probes needed.",
     650, True, None, [("recall", {"bodies": [1, 2]}, "Body #2, z −0.4")]),
    (66.0, "airo-2", "describe", "What is ahead?", "A clear aisle, and two pallets stacked by the Packing bench.",
     1000, True, ("frame", "airo-2", 66.0, None), None),
    (70.4, "airo-1", "transcribe", "microphone", "Nice work. AIRO 3, back to packing when you're done.", 710, True, None, None),
]
ZONE_T = 27.0


def smooth(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def latest(rows, T):
    return next((r for r in reversed(rows) if r[0] <= T), None)


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def route(points, lane=LANE):
    """Waypoints to a smooth drive line in the right-hand lane, parametrised by arc length."""
    p = np.asarray(points, float)
    seg = [np.linspace(a, b, max(2, int(np.linalg.norm(b - a) / 0.02)), endpoint=False) for a, b in pairwise(p)]
    d = np.vstack([*seg, p[-1:]])
    k = 12
    pad = np.pad(d, ((k, k), (0, 0)), mode="edge")
    d = np.stack([np.convolve(pad[:, i], np.ones(2 * k + 1) / (2 * k + 1), mode="valid") for i in range(2)], 1)
    t = np.gradient(d, axis=0)
    t /= np.linalg.norm(t, axis=1, keepdims=True) + 1e-9
    d = d + np.c_[t[:, 1], -t[:, 0]] * lane
    return np.r_[0, np.cumsum(np.linalg.norm(np.diff(d, axis=0), axis=1))], d


def arc(leg, T):
    """Distance along a drive leg at time T, easing in and out of each stop."""
    s, t0, t1 = leg[3][0], leg[1], leg[2]
    u = min(max((T - t0) / (t1 - t0), 0.0), 1.0)
    return s[-1] * (u - math.sin(2 * math.pi * u) / (2 * math.pi))


def point(leg, at):
    s, d = leg[3]
    return np.array([np.interp(at, s, d[:, 0]), np.interp(at, s, d[:, 1])])


def heading(leg, at):
    s = leg[3][0]
    a, b = point(leg, max(at - 0.03, 0)), point(leg, min(at + 0.03, s[-1]))
    return math.atan2(b[1] - a[1], b[0] - a[0])


class Track:
    def __init__(self, spec):
        self.legs = [(kind, t0, t1, route(pts[0]) if pts else None) for kind, t0, t1, *pts in spec["legs"]]
        self.pivots = spec["pivots"]

    def at(self, T):
        """x, y, heading and whether the robot is driving, at episode time T."""
        T = min(max(T, 0.0), CYCLE - 1e-6)
        i = next(i for i, leg in enumerate(self.legs) if leg[1] <= T < leg[2])
        leg, n = self.legs[i], len(self.legs)
        if leg[0] == "drive":
            at = arc(leg, T)
            return (*point(leg, at), heading(leg, at), True)
        prev = next(self.legs[(i - k) % n] for k in range(1, n) if self.legs[(i - k) % n][0] == "drive")
        nxt = next(self.legs[(i + k) % n] for k in range(1, n) if self.legs[(i + k) % n][0] == "drive")
        end = prev[3][0][-1]
        x, y = point(prev, end)
        h0, h1 = heading(prev, end), heading(nxt, 0)
        th = h0 + wrap(h1 - h0) * smooth((T - (leg[2] - 1.6)) / 1.6)
        th += sum(math.radians(deg) * smooth((T - a) / (b - a)) for a, b, deg in self.pivots if leg[1] <= a < leg[2])
        return x, y, th, False


def hex_bgr(h):
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


def look(eye, target, size=VIEW_WH, f=FOCAL):
    eye = np.asarray(eye, float)
    fwd = np.asarray(target, float) - eye
    fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    right /= np.linalg.norm(right)
    R = np.array([right, np.cross(fwd, right), fwd])
    K = np.array([[f, 0, size[0] / 2], [0, f, size[1] / 2], [0, 0, 1]])
    return K, R, eye


def project(cam, pts):
    K, R, c = cam
    q = (np.asarray(pts, float) - c) @ R.T
    return q[:, :2] / np.maximum(q[:, 2:], 1e-6) * K[[0, 1], [0, 1]] + K[:2, 2], q[:, 2]


TEX = np.array([[PPM, 0, MARGIN * PPM], [0, -PPM, (H + MARGIN) * PPM], [0, 0, 1]])
FACES = [((4, 5, 6, 7), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)), ((1, 2, 6, 5), (1, 0, 0)), ((2, 3, 7, 6), (0, 1, 0)),
         ((3, 0, 4, 7), (-1, 0, 0))]


def draw_boxes(img, cam, boxes):
    """Painter's algorithm over the visible faces of oriented boxes ((x, y, z0), (l, w, h), yaw, bgr)."""
    faces = []
    for (x, y, z0), (l, w, h), yaw, bgr in boxes:
        c, s = math.cos(yaw), math.sin(yaw)
        xy = np.array([[-l, -w], [l, -w], [l, w], [-l, w]]) / 2 @ np.array([[c, s], [-s, c]]) + (x, y)
        pts = np.r_[np.c_[xy, np.full(4, z0)], np.c_[xy, np.full(4, z0 + h)]]
        uv, z = project(cam, pts)
        if (z < 0.05).any():
            continue
        for idx, (nx, ny, nz) in FACES:
            n = np.array([nx * c - ny * s, nx * s + ny * c, nz])
            centre = pts[list(idx)].mean(0)
            if n @ (cam[2] - centre) > 0:
                shade = 0.48 + 0.52 * max(0.0, n @ LIGHT)
                faces.append((np.linalg.norm(cam[2] - centre), uv[list(idx)], [float(v) * shade for v in bgr]))
    for _, pts, col in sorted(faces, key=lambda f: -f[0]):
        cv2.fillConvexPoly(img, np.int32(pts * 16), col, cv2.LINE_AA, 4)


def car(x, y, th, color, weight=False, cargo=None):
    dark, tyre, brass = (34, 34, 38), (24, 24, 26), (70, 150, 190)
    parts = [((0, 0, 0.032), (0.25, 0.15, 0.006), dark), ((-0.03, 0, 0.038), (0.09, 0.06, 0.028), (70, 66, 60)),
             ((0, 0, 0.085), (0.23, 0.14, 0.006), hex_bgr(color)), ((-0.01, 0, 0.091), (0.075, 0.055, 0.01), (150, 90, 20)),
             ((0.112, 0, 0.091), (0.022, 0.06, 0.028), (190, 190, 195))]
    parts += [((dx, dy * 0.0725, 0), (0.065, 0.028, 0.065), tyre) for dx in (0.08, -0.08) for dy in (1, -1)]
    parts += [((dx, dy, 0.038), (0.008, 0.008, 0.047), brass) for dx in (0.1, -0.1) for dy in (0.055, -0.055)]
    if weight:
        parts.append(((-0.015, 0, 0.091), (0.13, 0.1, 0.05), (98, 100, 104)))
    if cargo:
        size = 0.16 if cargo == "crate" else 0.12
        parts.append(((0.135 + size / 2, 0, 0), (size, size, size * 0.8), (88, 138, 186) if cargo == "crate" else (110, 165, 210)))
    c, s = math.cos(th), math.sin(th)
    return [((x + px * c - py * s, y + px * s + py * c, z0), size, th, bgr) for (px, py, z0), size, bgr in parts]


def statics(rng):
    boxes = []
    for cx, cy in RACKS:
        for dx in (-0.53, 0.53):
            for dy in (-0.23, 0.23):
                boxes.append(((cx + dx, cy + dy, 0), (0.04, 0.04, 0.62), 0, (120, 92, 52)))
        for z in (0.06, 0.34, 0.6):
            boxes.append(((cx, cy, z), (1.1, 0.5, 0.025), 0, (150, 152, 156)))
        for z in (0.085, 0.365):
            x = cx - 0.5
            while x < cx + 0.3:
                w = rng.uniform(0.16, 0.26)
                boxes.append(((x + w / 2, cy + rng.uniform(-0.05, 0.05), z),
                              (w, rng.uniform(0.26, 0.38), rng.uniform(0.12, 0.22)), rng.uniform(-0.1, 0.1),
                              (int(rng.integers(80, 100)), int(rng.integers(128, 150)), int(rng.integers(176, 200)))))
                x += w + rng.uniform(0.02, 0.08)
    for (cx, cy), (l, w), h in BENCHES:
        boxes.append(((cx, cy, 0), (l, w, h), 0, (96, 98, 104) if h > 0.3 else (90, 140, 175)))
        if h > 0.3:
            boxes.append(((cx, cy, h), (l + 0.04, w + 0.04, 0.03), 0, (175, 178, 182)))
    return boxes


def tex_px(p):
    return int((p[0] + MARGIN) * PPM), int((H + MARGIN - p[1]) * PPM)


def floor_texture(rng):
    hh, ww = int((H + 2 * MARGIN) * PPM), int((W + 2 * MARGIN) * PPM)
    coarse = cv2.resize(cv2.GaussianBlur(rng.normal(0, 1, (hh // 8, ww // 8)).astype(np.float32), (0, 0), 2), (ww, hh))
    img = np.full((hh, ww, 3), (52, 54, 58), np.float32)
    m = int(MARGIN * PPM)
    img[m:m + int(H * PPM), m:m + int(W * PPM)] = (116, 118, 121)
    img += (coarse * 6 + rng.normal(0, 2.5, (hh, ww)))[..., None]
    img = img.clip(0, 255).astype(np.uint8)
    for x in range(1, int(W)):
        cv2.line(img, tex_px((x, 0)), tex_px((x, H)), (116, 119, 122), 1, cv2.LINE_AA)
    for y in range(1, int(H)):
        cv2.line(img, tex_px((0, y)), tex_px((W, y)), (116, 119, 122), 1, cv2.LINE_AA)
    for width, col in ((0.74, (40, 196, 236)), (0.68, (104, 107, 110))):
        for a, b in AISLES:
            cv2.line(img, tex_px(a), tex_px(b), col, int(width * PPM), cv2.LINE_AA)
    for name, (x, y) in STATIONS.items():
        tl, br = tex_px((x - 0.27, y + 0.27)), tex_px((x + 0.27, y - 0.27))
        cv2.rectangle(img, tl, br, (150, 154, 158), -1)
        cv2.rectangle(img, tl, br, (236, 238, 240), 5, cv2.LINE_AA)
        label = name.upper() if name == "qa" else name.capitalize()
        cv2.putText(img, label, (tl[0] + 6, br[1] + 34), cv2.FONT_HERSHEY_DUPLEX, 0.85, (232, 234, 236), 2, cv2.LINE_AA)
    spill, sheen = np.zeros(img.shape[:2], np.float32), np.zeros(img.shape[:2], np.float32)
    cv2.fillPoly(spill, [np.int32([tex_px(p) for p in SPILL])], 1.0, cv2.LINE_AA)
    cv2.ellipse(sheen, tex_px((2.9, 1.58)), (30, 7), -12, 0, 360, 1.0, -1, cv2.LINE_AA)
    spill, sheen = cv2.GaussianBlur(spill, (0, 0), 2.5)[..., None], cv2.GaussianBlur(sheen, (0, 0), 7)[..., None]
    img = (img * (1 - 0.82 * spill) + np.array([34, 30, 28]) * 0.82 * spill + 70 * sheen * spill).clip(0, 255).astype(np.uint8)
    shadow = np.zeros(img.shape[:2], np.float32)
    for cx, cy in RACKS:
        cv2.rectangle(shadow, tex_px((cx - 0.6, cy + 0.3)), tex_px((cx + 0.6, cy - 0.3)), 1.0, -1)
    for (cx, cy), (l, w), _ in BENCHES:
        cv2.rectangle(shadow, tex_px((cx - l / 2, cy + w / 2)), tex_px((cx + l / 2, cy - w / 2)), 1.0, -1)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 14)
    return (img * (1 - 0.4 * shadow[..., None])).astype(np.uint8)


def vignette():
    yy, xx = np.mgrid[:VIEW_WH[1], :VIEW_WH[0]] / np.array(VIEW_WH[::-1])[:, None, None]
    shade = (0.55 + 0.45 * np.clip(yy * 1.5 + 0.1, 0, 1)) * (1 - 0.5 * ((xx - 0.5) ** 2 + (yy - 0.55) ** 2) * 1.6)
    return shade[..., None].astype(np.float32)


PART_INFO = {"wheel": "65 mm rubber tyre on a TT gear motor", "motor": "TT gear motor, 1:48, driven by PWM",
             "battery": "2 × 18650 cells, 7.4 V", "board": "Arduino UNO and the SmartCar shield",
             "sensor": "HC-SR04 on an SG90 servo", "marker": "ArUco tag the overhead camera tracks"}
PRESETS = {"chase": (2.3, 1.45), "iso": (2.3, 2.0), "top": (0.3, 3.0), "floor": (5.6, 5.0)}


class View:
    """Fake orbit camera answering /camera; it frames the selected robot from the outside of the floor."""

    def __init__(self):
        self.az = self.target = None
        self.preset("chase")

    def preset(self, name):
        self.name, self.daz = name, 0.0
        self.dist, self.height = PRESETS.get(name, (0.42, 0.24))

    def handle(self, req):
        part = None
        if "preset" in req:
            self.preset(req["preset"])
            part = self.part(req["preset"])
        elif "pick" in req:
            u, v = req["pick"]
            r, ang = math.hypot((u - 0.5) * 16 / 9, v - 0.5), math.atan2(v - 0.5, u - 0.5)
            if r < 0.2:
                part = self.part(list(PART_INFO)[int((ang + math.pi) / (2 * math.pi) * len(PART_INFO)) % len(PART_INFO)])
                if req.get("focus"):
                    self.name, self.dist, self.height = None, 0.42, 0.24
        else:
            self.name = None
            if "orbit" in req:
                self.daz += math.radians(req["orbit"][0])
                self.height = min(max(self.height - math.radians(req["orbit"][1]) * self.dist, 0.1), 6.0)
            if "zoom" in req:
                self.dist = min(max(self.dist * req["zoom"], 0.2), 8.0)
                self.height = min(max(self.height * req["zoom"], 0.1), 6.0)
        return {"camera": {"preset": self.name, "follow": self.name != "floor"}, "part": part}

    @staticmethod
    def part(name):
        return {"name": name.capitalize(), "detail": PART_INFO[name]} if name in PART_INFO else None

    def camera(self, xy, T):
        overview = self.name == "floor"
        target = np.array([W / 2, H / 2]) if overview else np.asarray(xy, float)
        self.target = target if self.target is None else self.target + 0.12 * (target - self.target)
        tx, ty = self.target
        out = -2.0 if overview else math.atan2((ty - H / 2) / H, (tx - W / 2) / W) + 0.45
        self.az = out if self.az is None else self.az + 0.05 * wrap(out - self.az)
        az = self.az + 0.08 * math.sin(T / 9) + self.daz
        # Aim a little in front of the robot so it sits above the now line at the bottom of the screen.
        lead = 0.0 if overview else 0.12 * self.dist
        aim = (tx + lead * math.cos(az), ty + lead * math.sin(az), 0.06)
        return look((tx + self.dist * math.cos(az), ty + self.dist * math.sin(az), self.height), aim)


def params(values, key=False):
    return [{"name": nm, "value": v, "key": k if key else None, "unit": u, "lo": lo, "hi": hi}
            for (nm, u, lo, hi), v, k in zip(PARAMS, values, KEY)]


class Episode:
    def __init__(self, cycle=0, seed=7):
        rng = np.random.default_rng(seed)
        self.cycle = cycle
        self.tex, self.statics, self.shade = floor_texture(rng), statics(rng), vignette()
        self.tracks = {rid: Track(spec) for rid, spec in ROBOTS.items()}
        self.noise = {rid: np.random.default_rng(i) for i, rid in enumerate(ROBOTS)}
        self.dv, self.dk = rng.normal(size=(2, MEMBERS))
        self.surprise = {rid: deque((self._surprise(rid, -1) for _ in range(120)), maxlen=160) for rid in ROBOTS}
        self.next_window = 0.0
        self.seen: dict[int, float] = {}
        self.only: str | None = None
        self.override: dict[str, tuple[float, str]] = {}

    def _surprise(self, rid, t):
        v = abs(self.noise[rid].normal(0.85, 0.3))
        for a, b, c, peak in ROBOTS[rid]["bumps"]:
            if a <= t:
                v += peak * smooth((t - a) / (b - a)) * (1 if t < c else math.exp(-(t - c) * 2.2))
        return v

    def weighted(self, rid, T):
        return any(a <= T < b for a, b in ROBOTS[rid]["weight"])

    def cargo(self, rid, T):
        return next((c for a, b, c in ROBOTS[rid]["cargo"] if a <= T < b), None)

    def status(self, rid, T):
        return latest(ROBOTS[rid]["story"], T)[1]

    def cloud(self, rid, T, x, y, th, moving):
        tr = self.tracks[rid]
        tau = np.linspace(0, 1.5, 12)
        if moving:
            fut = np.array([tr.at(T + t)[:2] for t in tau])
            sv, sk = (0.25, 0.5) if self.status(rid, T) == "surprised" else (0.07, 0.12)
            out = []
            for dv, dk in zip(self.dv, self.dk):
                q = np.clip(tau * (1 + sv * dv), 0, tau[-1])
                p = np.c_[np.interp(q, tau, fut[:, 0]), np.interp(q, tau, fut[:, 1])]
                n = np.gradient(p, axis=0)
                n = np.c_[-n[:, 1], n[:, 0]] / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
                out.append(p + n * (dk * sk * q[:, None] ** 2 * 0.4))
            return np.round(out, 3)
        probe = next(((a, deg) for a, b, deg in ROBOTS[rid]["pivots"] if a - 1.2 <= T < b), None)
        if not probe:
            return []
        ang = th + math.radians(probe[1]) * (1 - smooth((T - probe[0]) / 1.5)) + np.radians(5 * self.dv)
        r = np.linspace(0.04, 0.34, 6)
        return np.round(np.stack([x + np.outer(np.cos(ang), r), y + np.outer(np.sin(ang), r)], -1), 3)

    def path(self, rid, T, x, y):
        plan = next((pts for a, b, pts in ROBOTS[rid].get("plans", []) if a <= T < b), None)
        if plan is not None:
            return np.round([[x, y], *plan], 3)
        pts = []
        for leg in self.tracks[rid].legs:
            kind, t0, t1, line = leg
            if t1 <= T:
                continue
            if kind == "hold":
                if self.status(rid, t0 + 0.01) not in WORKING:
                    break
                continue
            s, d = line
            pts += d[s >= arc(leg, T)][::6].tolist() + [d[-1].tolist()]
        return np.round([[x, y], *pts], 3) if pts else []

    def body(self, rid, T):
        spec = ROBOTS[rid]
        n = latest(spec["bodies"], T)[1]
        relearned = 11.3 if rid == "airo-1" and n == 2 and T < 43.4 else None
        return {"n": n, "name": next(m["name"] for m in spec["memory"] if m["n"] == n), "relearned_s": relearned,
                "probing_s": 8.8 if relearned else None, "portrait": None,
                "params": params(spec["bare"] if n == 1 else HEAVY, key=n == 1)}

    def memory(self, rid, T, n):
        out = []
        for m in ROBOTS[rid]["memory"]:
            if m["t"] > T:
                continue
            src = m.get("src")
            out.append({"n": m["n"], "name": m["name"], "cause": m["cause"],
                        "params": params(ROBOTS[rid]["bare"] if m["n"] == 1 else HEAVY),
                        "portrait": f"/models/{self.model_id(m['portrait'])}.jpg" if FEED[m["portrait"]][0] <= T else None,
                        "learned_t": max(m["t"], -60), "recalls": sum(r <= T for r in m["recalls"]),
                        "active": m["n"] == n, "from": src, "verified": src is None or m.get("verified", 99) <= T})
        return out

    def model_id(self, i):
        return self.cycle * 100 + i + 1

    def models(self, T):
        now, out = time.time(), []
        for i, (t, rid, kind, inp, outp, ms, ok, image, tools) in enumerate(FEED):
            if t > T:
                break
            mid = self.model_id(i)
            self.seen.setdefault(mid, now - (T - t))
            out.append({"id": mid, "t": self.seen[mid], "robot": rid, "model": KIND_MODEL[kind], "kind": kind,
                        "input": inp, "output": outp, "ms": ms, "ok": ok,
                        "image": f"/models/{mid}.jpg" if image else None,
                        "tools": [{"id": k, "name": nm, "args": json.dumps(a), "result": r}
                                  for k, (nm, a, r) in enumerate(tools)] if tools else None})
        return out[-40:]

    def robot(self, rid, T):
        spec = ROBOTS[rid]
        x, y, th, moving = self.tracks[rid].at(T)
        _, status, now, _ = latest(spec["story"], T)
        command = next(c for t, _, _, c in reversed(spec["story"]) if t <= T and c is not None)
        if rid in self.override and 0 <= T - self.override[rid][0] < 6:
            command = self.override[rid][1]
            now = f"Heard “{command}”. Finishing this job first."
        relearn = None
        if (rl := latest(spec["relearn"], T)) and T < rl[1]:
            a, _, phase, step, probe, start = rl
            relearn = {"phase": phase, "elapsed_s": round((a if phase == "done" else T) - start, 1), "probe": probe,
                       "step": step, "steps": 3 if rid == "airo-1" else 2}
        body = self.body(rid, T)
        return {
            "id": rid, "name": spec["name"], "color": spec["color"], "real": False, "now": now,
            "status": status, "command": command, "pose": {"x": round(x, 3), "y": round(y, 3), "theta": round(th, 3)},
            "cloud": self.cloud(rid, T, x, y, th, moving), "path": self.path(rid, T, x, y),
            "polygons": [{"label": "rack", "points": [[cx - 0.57, cy - 0.27], [cx + 0.57, cy - 0.27],
                                                      [cx + 0.57, cy + 0.27], [cx - 0.57, cy + 0.27]]} for cx, cy in RACKS],
            "body": body, "surprise": {"values": [round(v, 2) for v in self.surprise[rid]], "threshold": 3.0},
            "notebook": [{"t": t, "kind": k, "text": text, **extra} for t, k, text, extra in spec["notes"] if t <= T],
            "relearn": relearn, "memory": self.memory(rid, T, body["n"]),
        }

    def state(self, T, selected, only=None):
        while self.next_window <= T:
            for rid in ROBOTS:
                self.surprise[rid].append(self._surprise(rid, self.next_window))
            self.next_window += WINDOW
        zones = [{"kind": "slippery", "polygon": SPILL.tolist(), "label": "Oil spill", "found_by": "airo-2", "t": ZONE_T}]
        return {
            "t": round(T, 2), "mode": "real" if only else "sim",
            "site": {"name": "Factory", "floor": {"w": W, "h": H}, "stations": {k: list(v) for k, v in STATIONS.items()},
                     "zones": zones if T >= ZONE_T else []},
            "selected": only or selected,
            "robots": [self.robot(rid, T) for rid in ROBOTS if only in (None, rid)],
            "models": [m for m in self.models(T) if only is None or m["robot"] in (only, None)],
        }

    def scene(self, T):
        """The floor texture with contact shadows and ghost clouds, the 3D boxes and the robots' poses at T."""
        tex, boxes, poses = self.tex.copy(), list(self.statics), {}
        shadow = np.zeros(tex.shape[:2], np.uint8)
        for rid, spec in ROBOTS.items():
            if self.only not in (None, rid):
                continue
            x, y, th, moving = self.tracks[rid].at(T)
            poses[rid] = (x, y, th)
            c, s = math.cos(th), math.sin(th)
            foot = [tex_px((x + a * c - b * s, y + a * s + b * c)) for a, b in ((0.17, 0.1), (0.17, -0.1), (-0.15, -0.1), (-0.15, 0.1))]
            cv2.fillConvexPoly(shadow, np.int32(foot), 255, cv2.LINE_AA)
            cloud = self.cloud(rid, T, x, y, th, moving)
            if len(cloud):
                layer = tex.copy()
                for member in cloud:
                    end, prev = member[-1], member[-2]
                    h = math.atan2(end[1] - prev[1], end[0] - prev[0])
                    ch, sh = math.cos(h), math.sin(h)
                    rect = [tex_px((end[0] + a * ch - b * sh, end[1] + a * sh + b * ch))
                            for a, b in ((0.13, 0.085), (0.13, -0.085), (-0.13, -0.085), (-0.13, 0.085))]
                    cv2.fillConvexPoly(layer, np.int32(rect), hex_bgr(spec["color"]), cv2.LINE_AA)
                cv2.addWeighted(layer, 0.14, tex, 0.86, 0, tex)
            boxes += car(x, y, th, spec["color"], self.weighted(rid, T), self.cargo(rid, T))
        shadow = cv2.GaussianBlur(shadow, (0, 0), 5).astype(np.float32) / 255
        return (tex * (1 - 0.55 * shadow[..., None])).astype(np.uint8), boxes, poses

    def view(self, T, cam):
        tex, boxes, _ = self.scene(T)
        K, R, c = cam
        Hv = K @ np.c_[R[:, 0], R[:, 1], -R @ c] @ np.linalg.inv(TEX)
        img = cv2.warpPerspective(tex, Hv, VIEW_WH, flags=cv2.INTER_LINEAR, borderValue=(30, 28, 26))
        draw_boxes(img, cam, boxes)
        return (img * self.shade).astype(np.uint8)

    def topdown(self, T):
        tex, _, poses = self.scene(T)
        m = int(MARGIN * PPM)
        img = tex[m:m + int(H * PPM), m:m + int(W * PPM)].copy()
        for rid, (x, y, th) in poses.items():
            parts = car(x, y, th, ROBOTS[rid]["color"], self.weighted(rid, T), self.cargo(rid, T))
            for (bx, by, _z0), (l, w, _), yaw, bgr in sorted(parts, key=lambda b: b[0][2] + b[1][2]):
                c, s = math.cos(yaw), math.sin(yaw)
                pts = [(bx + a * c - b * s, by + a * s + b * c) for a, b in ((l / 2, w / 2), (l / 2, -w / 2), (-l / 2, -w / 2), (-l / 2, w / 2))]
                cv2.fillConvexPoly(img, np.int32([(px * PPM * 16, (H - py) * PPM * 16) for px, py in pts]), bgr, cv2.LINE_AA, 4)
        return img, poses

    def thumbnail(self, spec):
        """A model's input frame: the overhead view around the robot, with SAM's masks when it segmented."""
        if spec[0] == "portrait":
            return portrait(ROBOTS[spec[1]]["color"], spec[2])
        _, rid, t, mask = spec
        img, poses = self.topdown(max(t, 0.0))
        x, y, th = poses[rid]
        cx, cy = SPILL.mean(0) if mask == "spill" else STATIONS["packing"] if mask == "station" else (x, y)
        cw, ch = 1.6, 1.0
        x0 = int(np.clip((cx - cw / 2) * PPM, 0, img.shape[1] - cw * PPM))
        y0 = int(np.clip((H - cy - ch / 2) * PPM, 0, img.shape[0] - ch * PPM))
        crop = cv2.resize(img[y0:y0 + int(ch * PPM), x0:x0 + int(cw * PPM)], (480, 300), interpolation=cv2.INTER_AREA)
        k = 480 / (cw * PPM)
        to = lambda p: ((p[0] * PPM - x0) * k, ((H - p[1]) * PPM - y0) * k)
        square = lambda cx, cy, r, c=1.0, s=0.0: [to((cx + a * c - b * s, cy + a * s + b * c)) for a, b in ((r, r), (r, -r), (-r, -r), (-r, r))]
        polys = []
        if mask == "cargo":
            size = 0.16 if self.cargo(rid, t) == "crate" else 0.12
            c, s = math.cos(th), math.sin(th)
            a0 = 0.135 + size / 2
            polys = [square(x + a0 * c, y + a0 * s, size / 2 + 0.01, c, s)]
        elif mask == "spill":
            polys = [[to(p) for p in SPILL]]
        elif mask == "station":
            polys = [square(*STATIONS["packing"], 0.29)]
        if polys:
            pts = [np.int32(np.array(p) * 16) for p in polys]
            layer = crop.copy()
            cv2.fillPoly(layer, pts, (255, 132, 10), cv2.LINE_AA, 4)
            cv2.addWeighted(layer, 0.45, crop, 0.55, 0, crop)
            cv2.polylines(crop, pts, True, (255, 255, 255), 2, cv2.LINE_AA, 4)
        return crop


def portrait(color, weight, size=512):
    yy, xx = np.mgrid[:size, :size] / size
    glow = np.exp(-((xx - 0.5) ** 2 + (yy - 0.42) ** 2) * 4)[..., None]
    img = (np.array([30, 28, 27]) + glow * np.array([46, 44, 42]) + (1 - yy[..., None]) * 10).astype(np.uint8)
    cam = look((0.46, -0.36, 0.3), (0.01, 0, 0.05), (size, size), 640)
    shadow = np.zeros((size, size), np.uint8)
    foot = [(a, b, 0) for a, b in ((0.18, 0.11), (0.18, -0.11), (-0.16, -0.11), (-0.16, 0.11))]
    cv2.fillConvexPoly(shadow, np.int32(project(cam, foot)[0]), 255, cv2.LINE_AA)
    shadow = cv2.GaussianBlur(shadow, (0, 0), 12).astype(np.float32)[..., None] / 255
    img = (img * (1 - 0.6 * shadow)).astype(np.uint8)
    draw_boxes(img, cam, car(0, 0, 0, color, weight))
    return img


def jpeg(img, q=85):
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])[1].tobytes()


CHANGES = ("weight", "tow", "slippery", "flip", "spill", "unload")


class Demo:
    def __init__(self, dash: Dashboard, start: float = 0.0, hardware: bool = False):
        self.dash, self.hardware = dash, hardware
        self.ep, self.cam = Episode(int(start // CYCLE)), View()
        self.clock, self.wall, self.cycle = start, time.monotonic(), int(start // CYCLE)
        self.selected, self.changes = "airo-1", []
        self.images = {i: jpeg(self.ep.thumbnail(f[7])) for i, f in enumerate(FEED) if f[7]}
        dash.sim = {} if hardware else {"camera": True, "preset": "chase", "paused": False, "speed": 1,
                                        "presets": list(PRESETS), "parts": list(PART_INFO), "changes": []}

        @dash.app.get("/models/{name}")
        def model_image(name: str):
            stem = name.removesuffix(".jpg")
            img = self.images.get(int(stem) % 100 - 1) if stem.isdigit() else None
            return Response(img, media_type="image/jpeg") if img else Response(status_code=404)

    def command(self, text):
        sim, low = self.dash.sim, text.strip().lower()
        words = low.split()
        if words[:1] == ["select"]:
            rid = "-".join(words[1:])
            self.selected = rid if rid in ROBOTS else self.selected
        elif low in ("pause", "resume") and sim:
            sim["paused"] = low == "pause"
        elif words[:1] == ["speed"] and sim:
            sim["speed"] = float(words[1])
        elif low == "unload":
            self.changes = [c for c in self.changes if c != "weight"]
        elif low in CHANGES:
            self.changes = sorted({*self.changes, low})
        elif words:
            rid = f"airo-{words[1]}" if len(words) > 2 and words[0] == "airo" else None
            if rid in ROBOTS:
                self.ep.override[rid] = (self.clock % CYCLE, " ".join(words[2:]))
            else:
                self.ep.override[self.selected] = (self.clock % CYCLE, text.strip())

    def tick(self, n):
        sim = self.dash.sim
        now = time.monotonic()
        if not sim.get("paused"):
            self.clock += (now - self.wall) * sim.get("speed", 1)
        self.wall = now
        if int(self.clock // CYCLE) != self.cycle:
            self.cycle = int(self.clock // CYCLE)
            self.ep, self.changes = Episode(self.cycle), []
        self.ep.only = "airo-1" if self.hardware else None
        T = self.clock % CYCLE
        while not self.dash.commands.empty():
            self.command(self.dash.commands.get_nowait())
        while not self.dash.camera.empty():
            req, fut = self.dash.camera.get_nowait()
            if not fut.done():
                fut.set_result(self.cam.handle(req))
        state = self.ep.state(T, self.selected, "airo-1" if self.hardware else None)
        pose = next(r for r in state["robots"] if r["id"] == state["selected"])["pose"]
        if self.hardware:
            self.dash.set_view(self.ep.view(T, look((W / 2, -0.6, 4.4), (W / 2, H / 2 + 0.1, 0))))
        else:
            scripted = ["weight"] if self.ep.weighted(state["selected"], T) else []
            sim.update(preset=self.cam.name, changes=sorted({*scripted, *self.changes}))
            self.dash.set_view(self.ep.view(T, self.cam.camera((pose["x"], pose["y"]), T)))
        if n % 4 == 0:
            self.dash.set_frame(self.ep.topdown(T)[0])
        self.dash.publish(state)


async def run(demo: Demo):
    n = 0
    while True:
        t0 = time.monotonic()
        demo.tick(n)
        n += 1
        await asyncio.sleep(max(0.01, 0.066 - (time.monotonic() - t0)))


async def main(args):
    dash = Dashboard()
    demo = Demo(dash, args.start, args.hardware)
    print(f"dashboard on http://{args.host}:{args.port}")
    await asyncio.gather(dash.serve(args.host, args.port), run(demo))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--start", type=float, default=0.0, help="seconds into the episode")
    ap.add_argument("--hardware", action="store_true", help="one real robot, no sim controls")
    asyncio.run(main(ap.parse_args()))

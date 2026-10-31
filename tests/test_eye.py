import asyncio
import math
from types import SimpleNamespace

import numpy as np

from airo.body import KEY
from airo.control import recentre
from airo.eye import fit
from airo.mjsim import MjRig
from airo.probes import halt
from airo.scientist import Scientist, _where
from airo.sim import SimRig


def test_fit_recovers_marker_offset():
    rng = np.random.default_rng(0)
    d = np.array([0.06, -0.015])
    runs = []
    for c, ths in [((0.4, 0.6), np.linspace(1.5, 3.4, 30)), ((0.42, 0.61), np.linspace(3.4, 1.2, 30))]:
        pts = [(c[0] - (d[0] * math.cos(t) - d[1] * math.sin(t)), c[1] - (d[0] * math.sin(t) + d[1] * math.cos(t)), t)
               for t in ths]
        runs.append(np.array(pts) + rng.normal(0, [0.001, 0.001, 0.005], (len(pts), 3)))
    got, rms = fit(runs)
    assert np.allclose(got, d, atol=0.004) and rms < 0.003


def test_where_in_words():
    assert _where((-0.064, 0.002)) == "6.4 cm behind"
    assert _where((0.03, 0.012)) == "3.0 cm in front and 1.2 cm left of"
    assert _where((0.0, -0.02)) == "2.0 cm right of"


def test_no_tracker_no_eye_check():
    rig = SimRig(KEY, log_dir=None)
    sci = Scientist(rig)
    asyncio.run(sci.find_eye())
    assert not sci.notebook and not len(rig.history[0])


def test_eye_check_needs_room():
    rig = SimRig(KEY, pose0=(0.25, 0.6, 0.0), log_dir=None)
    rig.tracker = SimpleNamespace(offset=(0.0, 0.0))
    sci = Scientist(rig)
    asyncio.run(sci.find_eye())
    assert [e["kind"] for e in sci.notebook] == ["eye"] and "No room" in sci.notebook[0]["text"]
    assert not len(rig.history[0])


def test_centred_marker_is_left_alone():
    rig = SimRig(KEY, log_dir=None)
    rig.tracker = SimpleNamespace(offset=(0.0, 0.0))
    sci = Scientist(rig)
    asyncio.run(sci.find_eye())
    assert rig.tracker.offset == (0.0, 0.0) and not sci.segments
    assert "on my turning centre" in sci.notebook[-1]["text"] and not sci.notebook[-1].get("moved")


def test_twin_finds_a_marker_taped_off_centre():
    """The twin's turning centre sits ~0.4 cm behind its body origin, so a marker taped 6 cm behind the origin is
    ~5.6 cm behind the turning centre."""
    rig = MjRig(source="aruco", marker_at=(-0.06, 0.0), size=(960, 540), log_dir=None)
    asyncio.run(halt(rig, 0.2))
    sci = Scientist(rig, lambda r: recentre(r, sci.best), k=4)
    asyncio.run(sci.learn(budget=5))
    eye = [e for e in sci.notebook if e["kind"] == "eye"]
    assert len(eye) == 1 and eye[0]["moved"] and "behind my turning centre" in eye[0]["text"]
    fwd, left = eye[0]["marker_cm"]
    assert -6.5 < fwd < -5.0 and abs(left) < 0.8
    x, y, _ = rig.true_pose
    assert math.hypot(rig.pose.x - x, rig.pose.y - y) < 0.01
    assert not any("does not fit the others" in e["text"] for e in sci.notebook)
    pivots = [e for e in sci.notebook if e["kind"] == "result" and e["probe"].startswith("pivot")]
    assert pivots and all(abs(e["measured"]["lateral_cm"]) < 1.5 for e in pivots)
    assert len(sci.segments) == 5

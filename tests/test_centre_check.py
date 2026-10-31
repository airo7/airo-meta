import importlib.util
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location("centre_check", Path(__file__).parents[1] / "scripts/centre_check.py")
centre_check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(centre_check)


def test_fit_recovers_marker_offset():
    rng = np.random.default_rng(0)
    d = np.array([0.06, -0.015])
    runs = []
    for c, ths in [((0.4, 0.6), np.linspace(1.5, 3.4, 30)), ((0.42, 0.61), np.linspace(3.4, 1.2, 30))]:
        pts = [(c[0] - (d[0] * math.cos(t) - d[1] * math.sin(t)), c[1] - (d[0] * math.sin(t) + d[1] * math.cos(t)), t)
               for t in ths]
        runs.append(np.array(pts) + rng.normal(0, [0.001, 0.001, 0.005], (len(pts), 3)))
    got, rms = centre_check.fit(runs)
    assert np.allclose(got, d, atol=0.004) and rms < 0.003

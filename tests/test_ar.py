"""The hardware dashboard's overlay and click-to-go: web/map.js maps floor metres to the hero on screen and back."""

import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from airo.camera import to_floor
from airo.fleet import _camera

MAP_JS = Path(__file__).resolve().parent.parent / "web" / "map.js"
FLOOR_TO_IMG = np.array([[700.0, 20, 160], [10, -700, 1000], [1e-5, 2e-5, 1]])
SIZE = (1920, 1080)


def js(expr: str):
    script = (f"const vm = require('vm'), ctx = {{}}; vm.createContext(ctx);"
              f"vm.runInContext(require('fs').readFileSync({json.dumps(str(MAP_JS))}, 'utf8')"
              f" + ';globalThis.out = ' + {json.dumps(expr)}, ctx); console.log(JSON.stringify(ctx.out))")
    return json.loads(subprocess.run([shutil.which("node"), "-e", script], capture_output=True, text=True,
                                     check=True).stdout)


@pytest.fixture
def cam():
    rig = SimpleNamespace(H=np.linalg.inv(FLOOR_TO_IMG), inset=0.15, frame=lambda: np.zeros((*SIZE[::-1], 3)))
    c = _camera(rig)
    return json.loads(json.dumps(c, default=lambda o: o.tolist()))


def test_camera_state_maps_floor_to_frame_pixels_both_ways(cam):
    assert cam["size"] == list(SIZE) and cam["inset"] == 0.15
    pts = np.array([[0.0, 0.0], [0.81, 1.21], [0.4, 0.6]])
    assert np.allclose(to_floor(np.array(cam["to_image"]), pts), to_floor(FLOOR_TO_IMG, pts))
    assert _camera(SimpleNamespace(H=None, frame=lambda: None)) is None


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_floor_to_screen_through_the_cover_fitted_hero(cam):
    # a tall window crops the 16:9 frame's sides: the frame is 1600 px wide, 400 px of it off each side
    box = js("cover({left: 0, top: 0, width: 800, height: 900}, 16 / 9)")
    assert box == pytest.approx({"x": -400, "y": 0, "w": 1600, "h": 900})
    u, v = to_floor(FLOOR_TO_IMG, np.array([[0.4, 0.6]]))[0]
    got = js(f"floorToScreen({json.dumps(cam)}, {json.dumps(box)}, [0.4, 0.6])")
    assert got == pytest.approx([-400 + u * 1600 / 1920, v * 900 / 1080])


@pytest.mark.skipif(shutil.which("node") is None, reason="needs node")
def test_a_click_on_the_hero_lands_on_the_floor_point_under_it(cam):
    box = {"x": 0, "y": -60, "w": 1440, "h": 810}
    u, v = to_floor(FLOOR_TO_IMG, np.array([[0.25, 1.0]]))[0]
    click = (u * 1440 / 1920, -60 + v * 810 / 1080)
    assert js(f"screenToFloor({json.dumps(cam)}, {json.dumps(box)}, {click[0]}, {click[1]})") == \
        pytest.approx([0.25, 1.0])

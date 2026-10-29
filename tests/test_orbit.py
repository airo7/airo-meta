import asyncio

import pytest

from airo.mjsim import MjRig
from airo.orbit import PARTS, VIEWS, Orbit
from airo.sim import world


@pytest.fixture(scope="module")
def rig():
    r = MjRig(objects=world(), bumper=True, log_dir=None, pose0=(1.0, 0.7, 0.4))
    yield r
    asyncio.run(r.close())


def look(rig, orbit, req):
    out = orbit.handle(req)
    orbit._t = None
    rig.view(1280, 720, orbit.update())
    return out


def test_every_preset_renders_and_reports_itself(rig):
    orbit = Orbit(rig)
    for name in [*VIEWS, *PARTS]:
        assert look(rig, orbit, {"preset": name})["camera"]["preset"] == name


def test_pick_names_the_part_under_the_pointer_and_focus_follows_it(rig):
    orbit = Orbit(rig)
    look(rig, orbit, {"preset": "marker"})
    assert orbit.handle({"pick": [0.5, 0.5]})["part"]["name"] == "ArUco marker 0"
    out = look(rig, orbit, {"pick": [0.5, 0.5], "focus": True})
    assert out["camera"] == {"preset": None, "follow": True} and orbit.shot.distance <= 0.22
    look(rig, orbit, {"preset": "wheel"})
    assert orbit.handle({"pick": [0.5, 0.5]})["part"]["name"] == "Front-left wheel"
    look(rig, orbit, {"preset": "floor"})
    assert orbit.handle({"pick": [0.0, 1.0]})["part"]["name"] in ("Floor", "Corner marker")


def test_orbit_zoom_and_pan_are_bounded_and_scale_with_distance(rig):
    orbit = Orbit(rig)
    look(rig, orbit, {"preset": "iso"})
    orbit.handle({"orbit": [10, -90]})
    assert orbit.shot.elevation == -89.5 and orbit.preset is None
    orbit.handle({"zoom": 1e-3})
    assert orbit.shot.distance == 0.06
    before = orbit.shot.target
    orbit.handle({"pan": [0.5, 0]})
    shift = sum((a - b) ** 2 for a, b in zip(orbit.shot.target, before)) ** 0.5
    assert 0.01 < shift < 0.1


def test_pause_holds_a_realtime_rig_and_speed_scales_it():
    rig = MjRig(objects=(), log_dir=None, realtime=True)

    async def run():
        rig.paused = True
        with pytest.raises(TimeoutError):
            await asyncio.wait_for(rig.step(0, 0), 0.3)
        rig.paused = False
        rig.set_speed(4)
        loop = asyncio.get_running_loop()
        t0 = loop.time()
        for _ in range(20):
            await rig.step(0, 0)
        return loop.time() - t0

    assert asyncio.run(run()) < 0.6
    asyncio.run(rig.close())

import argparse
import asyncio
import importlib.util
from pathlib import Path

import cv2
import numpy as np

from airo.mission import Mission
from airo.mjsim import MjRig
from airo.rig import add_args, open_rig
from airo.server import Dashboard
from airo.sim import world

spec = importlib.util.spec_from_file_location("demo", Path(__file__).parent.parent / "scripts" / "demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


def jpeg(dash: Dashboard, name: str) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(dash._jpeg[name], np.uint8), cv2.IMREAD_COLOR)


def test_mujoco_demo_steps_publishes_the_view_and_routes_body_changes(tmp_path):
    p = argparse.ArgumentParser()
    add_args(p)

    async def run():
        rig = await open_rig(p.parse_args(["--mujoco"]), objects=world(), log_dir=None)
        sci = demo.scientist(rig, tmp_path / "none.json")
        dash = Dashboard()
        mission = Mission(rig, sci)
        jobs = [asyncio.create_task(j) for j in (mission.run(dash), demo.film(rig, sci, dash),
                                                 demo.route(rig, dash.commands, mission.inbox))]
        dash.commands.put_nowait("weight")
        dash.commands.put_nowait("how are you doing")
        dash.commands.put_nowait("speed 2")
        cam = asyncio.get_running_loop().create_future()
        dash.camera.put_nowait(({"preset": "iso"}, cam))
        while rig.t < 0.5:
            await asyncio.sleep(0.01)
        assert (await cam)["camera"]["preset"] == "iso"
        for j in jobs:
            j.cancel()
        await asyncio.gather(*jobs, return_exceptions=True)
        await rig.close()
        return rig, dash, mission

    rig, dash, mission = asyncio.run(run())
    assert isinstance(rig, MjRig) and rig.realtime and rig.tracker is not None
    assert rig.m.body_subtreemass[1] > 0.94 + demo.WEIGHT_KG - 0.01
    assert "Body #0 prior" in mission.notes[-1]["text"]
    assert rig.speed == 2 and dash.sim["camera"] and dash.sim["changes"] == ["weight"]
    assert jpeg(dash, "view").shape == (900, 1600, 3) and jpeg(dash, "video").shape == (560, 800, 3)
    assert dash.state["status"] == "idle"

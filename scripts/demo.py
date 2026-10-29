"""Demo entry point: dashboard, mission and optional voice on one event loop.

--sim rehearses on a simulated floor, --mujoco on the physics twin with the 3D ghost view. Type a command on stdin
or in the dashboard ("deliver to red", "learn", "stop"), or change the simulated body mid-run: "weight" or "tow",
and on --mujoco also "slippery" or "flip", or "pause", "resume", "speed 0.5". --dry shows the camera, pose and
dashboard without driving. On --mujoco the 3D view takes the mouse: drag to orbit, scroll to zoom, click a part.
"""

import argparse
import asyncio
import sys
import time
from dataclasses import replace

import numpy as np

from airo import meta_api
from airo.body import cloud, load
from airo.control import recentre
from airo.mission import Mission
from airo.mjsim import WEIGHT_KG, MjRig
from airo.orbit import PARTS, VIEWS, Orbit
from airo.rig import RigAbort, add_args, body_path, open_rig
from airo.scientist import Scientist
from airo.server import Dashboard
from airo.sim import SimRig, world

FPS = 24
VIEW = (1600, 900)
HORIZON = 1.2


def scientist(rig, path) -> Scientist:
    """Starts from the saved body when there is one, so learning carries over between runs."""
    sci = Scientist(rig, recentre=lambda r: recentre(r, sci.best))
    if saved := load(path):
        sci.best, sci.members, sci.name = saved
        sci.n = 1
        print(f"loaded {path}: {sci.name}")
    return sci


def change_body(rig, what: str) -> bool:
    if isinstance(rig, MjRig):
        if what in ("pause", "resume"):
            rig.paused = what == "pause"
            return True
        if what.startswith("speed "):
            try:
                rig.set_speed(float(np.clip(float(what[6:]), 0.1, 4)))
            except ValueError:
                return False
            return True
        change = {"weight": lambda: rig.add_weight(WEIGHT_KG), "tow": rig.attach_tow, "slippery": rig.slippery_patch,
                  "flip": lambda: rig.flip_motor("l")}.get(what)
        if change:
            change()
            print(f"physics: {what}")
        return change is not None
    if not isinstance(rig, SimRig):
        return False
    b = rig.truth
    if what == "weight":
        rig.truth = replace(b, g_l=b.g_l * 0.65, g_r=b.g_r * 0.65, eta=b.eta * 0.7)
    elif what == "tow":
        rig.truth = replace(b, g_l=b.g_l * 0.85, g_r=b.g_r * 0.85, lag=b.lag * 2)
    else:
        return False
    print(f"truth body is now {rig.truth}")
    return True


async def console(commands: asyncio.Queue):
    loop = asyncio.get_running_loop()
    while line := await loop.run_in_executor(None, sys.stdin.readline):
        if text := line.strip():
            commands.put_nowait(text)


async def route(rig, commands: asyncio.Queue, inbox: asyncio.Queue, dry: bool = False):
    """Body changes go to the simulated rig, everything else to the mission."""
    while True:
        text = await commands.get()
        if dry:
            print(f"dry run, not driving: {text}")
        else:
            try:
                handled = change_body(rig, text.lower())
            except Exception as e:  # a bad body change must not take the dashboard down with it
                print(f"{text}: {e}")
                handled = True
            if not handled:
                inbox.put_nowait(text)


async def film(rig: MjRig, sci: Scientist, dash: Dashboard):
    """The interactive 3D view plus the ghost cloud over the next HORIZON seconds. Runs on the event loop: the
    rig's GL context belongs to the thread that steps it."""
    orbit, last = Orbit(rig, VIEW), None
    while True:
        tick = time.monotonic()
        while not dash.camera.empty():
            req, fut = dash.camera.get_nowait()
            if not fut.done():
                fut.set_result(orbit.handle(req))
        dash.sim = {"camera": True, "preset": orbit.preset, "paused": rig.paused, "speed": rig.speed,
                    "presets": list(VIEWS), "parts": list(PARTS), "changes": rig.changes}
        p = rig.pose
        if fresh := p and p.t != last:
            last = p.t
            t = np.linspace(p.t, p.t + HORIZON, 13)
            rig.render_ghosts(cloud(sci.members, t, *rig.history, (p.x, p.y, p.theta)), 0.1)
        if fresh or orbit.moving:
            dash.set_view(rig.view(*VIEW, orbit.update()))
        await asyncio.sleep(max(0.0, tick + 1 / FPS - time.monotonic()))


async def listen(inbox: asyncio.Queue):
    from airo.voice import transcribe

    try:
        await transcribe(lambda text, final: final and inbox.put_nowait(text))
    except Exception as e:
        print(f"voice off: {e}; the text bar still works")


async def main(args):
    # From the floor centre the drive to the parcel starts far enough out to be watched: the parcel approach is not.
    pose0 = (1.0, 0.7, 0.0) if args.mujoco else (0.4, 0.7, 0.0)
    rig = await open_rig(args, objects=world(), pose0=pose0, log_dir="data/runs")
    sim = args.sim or args.mujoco
    if sim:
        meta_api.segment = rig.sam
    path = body_path(args)
    dash = Dashboard()
    sci = scientist(rig, path)
    mission = Mission(rig, sci, mode="sim" if sim else "real", body_path=path)
    print(f"dashboard on http://{args.host}:{args.port}")
    jobs = [dash.serve(args.host, args.port), mission.run(dash), console(dash.commands),
            route(rig, dash.commands, mission.inbox, args.dry and not sim)]
    if args.mujoco:
        jobs.append(film(rig, sci, dash))
    if args.voice:
        jobs.append(listen(dash.commands))
    try:
        await asyncio.gather(*jobs)
    finally:
        await rig.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    p.add_argument("--voice", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    try:
        asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt:
        pass
    except RigAbort as e:
        sys.exit(f"airo: {e}")

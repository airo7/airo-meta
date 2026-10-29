"""Demo entry point: dashboard, fleet, missions and optional voice on one event loop.

--site factory (MuJoCo, three robots by default, --robots N) runs a fleet on the factory floor; --site home is the
real car's marker floor: --sim rehearses on a simulated floor, --mujoco on the physics twin, neither drives the car.
Type a command on stdin or in the dashboard: "airo 2 deliver to packing", "select airo 3", "deliver to dock" (the
selected robot), "learn", "stop", or change a body mid-run: "weight", "remove weight", "tow", "untow", "flip"; the
place changes with "spill" and "clean"; "pause", "resume", "speed 0.5". --dry shows the camera, pose and dashboard
without driving. In the 3D view drag to orbit, scroll to zoom, click a part.
"""

import argparse
import asyncio
import sys
import time

import numpy as np

from airo import meta_api
from airo.body import cloud, load
from airo.control import recentre
from airo.fleet import Fleet, crew
from airo.mjsim import fleet as fleet_rigs
from airo.orbit import PARTS, VIEWS, Orbit
from airo.rig import MJ_BODY_PATH, RigAbort, add_args, body_path, open_rig
from airo.scientist import Scientist
from airo.server import Dashboard
from airo.sim import world
from airo.site import SITES

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


async def open_rigs(args, site) -> list:
    if site.scene == "factory":
        return fleet_rigs(site, min(args.robots, len(site.starts)), realtime=True, bumper=True, log_dir="data/runs")
    # From the floor centre the drive to the parcel starts far enough out to be watched: the parcel approach is not.
    pose0 = (1.0, 0.7, 0.0) if args.mujoco else (0.4, 0.7, 0.0)
    return [await open_rig(args, objects=world(), pose0=pose0, log_dir="data/runs")]


async def console(commands: asyncio.Queue):
    loop = asyncio.get_running_loop()
    while line := await loop.run_in_executor(None, sys.stdin.readline):
        if text := line.strip():
            commands.put_nowait(text)


async def film(fleet: Fleet, dash: Dashboard):
    """The interactive 3D view, following the selected robot, with its ghost cloud over the next HORIZON seconds. Runs on the event loop: the world's GL contexts belong to the thread that steps it."""
    world = fleet.world
    orbit, last = Orbit(fleet.robot.rig, VIEW), {}
    while True:
        tick = time.monotonic()
        if orbit.rig is not fleet.robot.rig:
            orbit.follow(fleet.robot.rig)
        while not dash.camera.empty():
            req, fut = dash.camera.get_nowait()
            if not fut.done():
                fut.set_result(orbit.handle(req))
        dash.sim = {"camera": True, "preset": orbit.preset, "paused": world.paused, "speed": world.speed,
                    "presets": list(VIEWS), "parts": list(PARTS), "changes": fleet.robot.rig.changes,
                    "robot": fleet.robot.id}
        fresh = False
        for r in fleet.robots:
            p = r.rig.pose
            if p and p.t != last.get(r.id):
                last[r.id], fresh = p.t, True
                # the full cloud for the robot in focus only: each cloud is a few hundred transparent geoms
                t = np.linspace(p.t, p.t + HORIZON, 13)
                r.rig.render_ghosts(cloud(r.sci.members, t, *r.rig.history, (p.x, p.y, p.theta))
                                    if r is fleet.robot else [], 0.1)
        if fresh or orbit.moving:
            dash.set_view(world.view(*VIEW, orbit.update()))
        await asyncio.sleep(max(0.0, tick + 1 / FPS - time.monotonic()))


async def listen(inbox: asyncio.Queue):
    from airo.voice import transcribe

    try:
        await transcribe(lambda text, final: final and inbox.put_nowait(text))
    except Exception as e:
        print(f"voice off: {e}; the text bar still works")


async def main(args):
    site = SITES[args.site]
    if site.scene == "factory" and args.sim:
        sys.exit("airo: the factory runs on the MuJoCo twin; drop --sim")
    rigs = await open_rigs(args, site)
    sim = args.sim or args.mujoco or site.scene == "factory"
    if sim:
        meta_api.segment = rigs[0].sam
    path = MJ_BODY_PATH if site.scene == "factory" else body_path(args)
    robots = crew(site, rigs, lambda rig: scientist(rig, path), mode="sim" if sim else "real",
                  body_path=path if len(rigs) == 1 else None, real=not sim)
    fleet = Fleet(site, robots, mode="sim" if sim else "real", dry=args.dry and not sim)
    dash = Dashboard()
    print(f"{site.name}, {len(robots)} robot{'s' * (len(robots) > 1)}: dashboard on http://{args.host}:{args.port}")
    jobs = [dash.serve(args.host, args.port), fleet.run(dash), console(dash.commands),
            *(r.mission.run() for r in robots)]
    if fleet.world is not None:
        jobs.append(film(fleet, dash))
    if args.voice:
        jobs.append(listen(dash.commands))
    try:
        await asyncio.gather(*jobs)
    finally:
        for rig in rigs:
            await rig.close()
        if fleet.world is not None:
            fleet.world.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(p)
    p.add_argument("--site", choices=sorted(SITES), default="home")
    p.add_argument("--robots", type=int, default=3, help="robots on the factory floor")
    p.add_argument("--voice", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    try:
        asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt:
        pass
    except RigAbort as e:
        sys.exit(f"airo: {e}")

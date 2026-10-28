"""Demo entry point: dashboard, mission and optional voice on one event loop.

--sim rehearses on a simulated floor. Type a command on stdin ("deliver to red", "learn", "stop"),
or "weight" / "tow" to change the simulated body mid-run.
"""

import argparse
import asyncio
import math
import os
import sys
from dataclasses import replace

from airo import meta_api
from airo.body import KEY, wrap
from airo.control import face, goto
from airo.mission import Mission
from airo.rig import Rig
from airo.scientist import Scientist
from airo.server import Dashboard
from airo.sim import SimRig, world


async def open_rig(args):
    if args.sim:
        rig = SimRig(KEY, floor=(2.0, 1.4), objects=world(), pose0=(0.4, 0.7, 0.0), realtime=True)
        meta_api.segment = rig.sam
        return rig
    return await Rig.open(args.floor, int(args.camera), args.marker_h, args.cam_h)


def scientist(rig) -> Scientist:
    """Recentring ends facing along the floor's long axis, where the long steps fit."""
    centre = (rig.floor[0] / 2, rig.floor[1] / 2)
    axis = 0.0 if rig.floor[0] >= rig.floor[1] else math.pi / 2

    async def recentre(r):
        await goto(r, sci.best, centre)
        await face(r, sci.best, axis if abs(wrap(r.pose.theta - axis)) <= math.pi / 2 else axis + math.pi)

    sci = Scientist(rig, recentre=recentre)
    return sci


def change_body(rig, what: str) -> bool:
    b = rig.truth
    if what == "weight":
        rig.truth = replace(b, g_l=b.g_l * 0.65, g_r=b.g_r * 0.65, eta=b.eta * 0.7)
    elif what == "tow":
        rig.truth = replace(b, g_l=b.g_l * 0.85, g_r=b.g_r * 0.85, lag=b.lag * 2)
    else:
        return False
    print(f"truth body is now {rig.truth}")
    return True


async def console(rig, inbox: asyncio.Queue):
    loop = asyncio.get_running_loop()
    while line := await loop.run_in_executor(None, sys.stdin.readline):
        text = line.strip()
        if text and not (isinstance(rig, SimRig) and change_body(rig, text)):
            inbox.put_nowait(text)


async def listen(inbox: asyncio.Queue):
    from airo.voice import transcribe

    try:
        await transcribe(lambda text, final: final and inbox.put_nowait(text))
    except Exception as e:
        print(f"voice off: {e}; the text bar still works")


async def main(args):
    rig = await open_rig(args)
    dash = Dashboard()
    mission = Mission(rig, scientist(rig), inbox=dash.commands, mode="sim" if args.sim else "real")
    print(f"dashboard on http://{args.host}:{args.port}")
    jobs = [dash.serve(args.host, args.port), mission.run(dash), console(rig, dash.commands)]
    if args.voice:
        jobs.append(listen(dash.commands))
    try:
        await asyncio.gather(*jobs)
    finally:
        await rig.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sim", action="store_true")
    p.add_argument("--voice", action="store_true")
    p.add_argument("--floor", nargs=2, type=float, default=(2.0, 1.4), metavar=("W", "H"))
    p.add_argument("--camera", default=os.getenv("AIRO_CAMERA_INDEX", "0"))
    p.add_argument("--marker-h", type=float, default=0.0)
    p.add_argument("--cam-h", type=float, default=1.8)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    try:
        asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt:
        pass

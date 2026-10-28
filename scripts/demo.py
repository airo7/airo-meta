"""Demo entry point: dashboard, mission and optional voice on one event loop.

--sim rehearses on a simulated floor. Type a command on stdin ("deliver to red", "learn", "stop"),
or "weight" / "tow" to change the simulated body mid-run.
"""

import argparse
import asyncio
import sys
from dataclasses import replace

from airo import meta_api
from airo.body import BODY_PATH, SIM_BODY_PATH, load
from airo.control import recentre
from airo.mission import Mission
from airo.rig import add_args, open_rig
from airo.scientist import Scientist
from airo.server import Dashboard
from airo.sim import SimRig, world


def scientist(rig, path) -> Scientist:
    """Starts from the saved body when there is one, so learning carries over between runs."""
    sci = Scientist(rig, recentre=lambda r: recentre(r, sci.best))
    if saved := load(path):
        sci.best, sci.members, sci.name = saved
        sci.n = 1
        print(f"loaded {path}: {sci.name}")
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
    rig = await open_rig(args, objects=world(), pose0=(0.4, 0.7, 0.0), log_dir="data/runs")
    if args.sim:
        meta_api.segment = rig.sam
    path = SIM_BODY_PATH if args.sim else BODY_PATH
    dash = Dashboard()
    mission = Mission(rig, scientist(rig, path), inbox=dash.commands, mode="sim" if args.sim else "real",
                      body_path=path)
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
    add_args(p)
    p.add_argument("--voice", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    try:
        asyncio.run(main(p.parse_args()))
    except KeyboardInterrupt:
        pass

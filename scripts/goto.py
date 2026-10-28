"""D2 gate: click a floor point in the window, the car drives there and the final error is printed."""

import argparse
import asyncio
import math

import cv2
import numpy as np

from airo.body import BODY_PATH, KEY, SIM_BODY_PATH, load
from airo.camera import to_floor
from airo.control import goto
from airo.rig import RigAbort, add_args, open_rig

WIN = "airo goto"


def draw(frame, rig, goal, scale):
    inv = np.linalg.inv(rig.H)
    if goal is not None:
        cv2.circle(frame, tuple(to_floor(inv, goal[None])[0].astype(int)), 12, (0, 0, 255), 3)
    p = rig.pose
    if p:
        tip = (p.x + 0.15 * math.cos(p.theta), p.y + 0.15 * math.sin(p.theta))
        a, b = to_floor(inv, np.array([[p.x, p.y], tip])).astype(int)
        cv2.arrowedLine(frame, tuple(a), tuple(b), (255, 0, 0), 3, tipLength=0.3)
    return cv2.resize(frame, None, fx=scale, fy=scale)


async def main(args):
    saved = load(args.body or (SIM_BODY_PATH if args.sim else BODY_PATH))
    body = saved[0] if saved else KEY
    print(f"body: {saved[2] if saved else 'answer key'}")
    rig = await open_rig(args)
    scale = min(1.0, 1280 / rig.frame().shape[1])
    clicks, errors, goal, task = [], [], None, None
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, lambda ev, u, v, *_: ev == cv2.EVENT_LBUTTONDOWN and clicks.append((u, v)))
    try:
        while cv2.waitKey(1) not in (27, ord("q")):
            cv2.imshow(WIN, draw(rig.frame().copy(), rig, goal, scale))
            if task is None and clicks:
                xy = to_floor(rig.H, np.array([clicks.pop()]) / scale)[0]
                clicks.clear()
                if rig.inside(*xy):
                    goal, task = xy, asyncio.create_task(goto(rig, body, xy))
                else:
                    print(f"({xy[0]:.2f}, {xy[1]:.2f}) is outside the floor inset")
            if task is None:
                try:
                    await rig.step(0, 0)
                except RigAbort as e:
                    print(e)
                continue
            await asyncio.sleep(0.03)
            if task.done():
                try:
                    errors.append(task.result())
                    print(f"final error {errors[-1] * 100:.1f} cm, "
                          f"{sum(e < 0.05 for e in errors)}/{len(errors)} within 5 cm")
                except RigAbort as e:
                    print(f"aborted: {e}")
                task = None
    finally:
        if task:
            task.cancel()
        await rig.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    add_args(p)
    p.add_argument("--body", help="body json; default is the learned body if saved, else the answer key")
    asyncio.run(main(p.parse_args()))

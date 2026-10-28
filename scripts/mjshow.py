"""Film the MuJoCo twin: the car follows an S-curve under pure pursuit while a fitted ensemble draws its ghost cloud.

uv run scripts/mjshow.py --out data/mj
"""

import argparse
import asyncio
from pathlib import Path

import cv2
import numpy as np

from airo.body import cloud, ensemble, fit
from airo.control import follow
from airo.mjsim import MjRig

PATH = [(0.35, 0.3), (0.75, 0.6), (1.1, 0.35), (1.55, 0.45), (1.75, 0.8)]
PROBES = [(40, 40), (60, 60), (110, 110), (150, 150), (-110, -110), (100, -100), (-100, 100), (120, 60), (60, 120)]


async def learn(k: int):
    segs = []
    for l, r in PROBES:
        rig = MjRig(objects=(), log_dir=None)
        for i in range(36):
            await rig.step(*((l, r) if 4 <= i < 24 else (0, 0)))
        segs.append(rig.segment())
    best = fit(segs)
    return best, ensemble(segs, best, k=k)


class Film:
    """Wraps a rig: every step also predicts the cloud one horizon ahead and writes a chase frame."""

    def __init__(self, rig: MjRig, members, video: cv2.VideoWriter, still: Path, horizon: float = 1.2):
        self.rig, self.members, self.video, self.still, self.horizon = rig, members, video, still, horizon
        self.n = 0

    def __getattr__(self, name):
        return getattr(self.rig, name)

    async def step(self, l: int, r: int):
        pose = await self.rig.step(l, r)
        p = self.rig.pose
        t = np.linspace(p.t, p.t + self.horizon, 13)
        self.rig.render_ghosts(cloud(self.members, t, *self.rig.history, (p.x, p.y, p.theta)), 0.1)
        self.video.write(self.rig.view(1280, 720))
        self.n += 1
        if self.n == 70:
            cv2.imwrite(str(self.still), self.rig.view(1920, 1080))
        return pose


async def main(out: Path, source: str):
    out.mkdir(parents=True, exist_ok=True)
    best, members = await learn(12)
    print("fitted", best)
    rig = MjRig(pose0=(0.3, 0.25, 0.7), source=source, log_dir=None)
    video = cv2.VideoWriter(str(out / "arc.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 20, (1280, 720))
    film = Film(rig, members, video, out / "chase.png")
    for _ in range(10):
        await film.step(0, 0)
    err = await follow(film, best, PATH)
    for _ in range(30):
        await film.step(0, 0)
    video.release()
    cv2.imwrite(str(out / "overhead.png"), rig.frame())
    cv2.imwrite(str(out / "overhead_ghosts.png"), rig.view(1920, 1080, "overhead"))
    print(f"final error {err * 100:.1f} cm, wrote {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("data/mj"))
    ap.add_argument("--source", choices=("truth", "aruco"), default="aruco")
    args = ap.parse_args()
    asyncio.run(main(args.out, args.source))

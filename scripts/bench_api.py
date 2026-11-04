"""SAM 3.1 and Muse Spark latency on a real overhead frame. Run with: uv run --env-file .env"""

import argparse
import asyncio
import os
import statistics
import time

import cv2
import numpy as np

from airo.camera import open_camera
from airo.meta_api import describe, segment
from airo.paths import DATA

OUT = DATA


def grab(camera: int | str) -> np.ndarray:
    cap = open_camera(camera)
    for _ in range(10):  # let auto exposure settle
        ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"camera {camera} gave no frame")
    return frame


async def timed(n: int, fn, *args) -> tuple[list[float], object]:
    ms, out = [], None
    for _ in range(n):
        t0 = time.perf_counter()
        out = await fn(*args)
        ms.append((time.perf_counter() - t0) * 1000)
    return ms, out


def summary(name: str, ms: list[float]) -> str:
    s = sorted(ms)
    return f"{name}: median {statistics.median(s):.0f} ms, max {s[-1]:.0f} ms over {len(s)}"


async def main(args):
    OUT.mkdir(exist_ok=True)
    frame = cv2.imread(args.image) if args.image else grab(args.camera)
    cv2.imwrite(str(OUT / "bench_frame.jpg"), frame)

    sam_ms, segs = await timed(args.n, segment, frame, args.concept)
    print(summary("sam-3.1", sam_ms), f"| {len(segs)} x '{args.concept}'")
    for s in segs:
        print(f"  box {s.box} score {s.score} area {int(s.mask.sum())} px")

    spark_ms, text = await timed(args.n, describe, frame, "Describe the objects on the floor in one sentence.")
    print(summary("muse-spark-1.3", spark_ms))
    print(" ", text.strip())

    overlay = frame.copy()
    for s in segs:
        overlay[s.mask] = (0.5 * overlay[s.mask] + (0, 128, 0)).astype(np.uint8)
        cv2.rectangle(overlay, s.box[:2], s.box[2:], (0, 255, 0), 2)
    cv2.imwrite(str(OUT / "bench_overlay.jpg"), overlay)
    print(f"overlay: {OUT / 'bench_overlay.jpg'}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--image")
    p.add_argument("--camera", default=os.getenv("AIRO_CAMERA", os.getenv("AIRO_CAMERA_INDEX", "0")))
    p.add_argument("--concept", default="box")
    p.add_argument("--n", type=int, default=5)
    asyncio.run(main(p.parse_args()))

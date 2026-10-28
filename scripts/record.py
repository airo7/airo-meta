"""Record a running demo for the video edit: the chase view (/view) to an MP4 at a fixed wall-clock rate, and each
new dashboard state with its frame index to a .jsonl beside it. Stop with Ctrl-C.

uv run scripts/record.py data/rec/take1.mp4 --url http://127.0.0.1:8000
"""

import argparse
import json
import signal
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np
from websockets.sync.client import connect


def mjpeg(url: str, latest: dict):
    buf = b""
    with urllib.request.urlopen(url) as r:
        while chunk := r.read1(1 << 16):
            buf += chunk
            # JPEG entropy data stuffs every 0xFF, so the first EOI after an SOI ends the frame
            while (a := buf.find(b"\xff\xd8")) >= 0 and (b := buf.find(b"\xff\xd9", a)) >= 0:
                latest["view"] = cv2.imdecode(np.frombuffer(buf[a:b + 2], np.uint8), cv2.IMREAD_COLOR)
                buf = buf[b + 2:]


def states(url: str, latest: dict):
    with connect(url, max_size=None) as ws:
        for msg in ws:
            latest["state"] = msg


def main(out: Path, url: str, fps: float):
    out.parent.mkdir(parents=True, exist_ok=True)
    latest: dict = {}
    threading.Thread(target=mjpeg, args=(f"{url}/view", latest), daemon=True).start()
    threading.Thread(target=states, args=(url.replace("http", "ws", 1) + "/ws", latest), daemon=True).start()
    while "view" not in latest:
        time.sleep(0.05)
    video = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, latest["view"].shape[1::-1])
    # uv forwards Ctrl-C too, so a second SIGINT must not interrupt release() and leave the MP4 unreadable
    stop = threading.Event()
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    n, last, tick = 0, None, time.monotonic()
    with open(out.with_suffix(".jsonl"), "w") as log:
        while not stop.is_set():
            video.write(latest["view"])
            if (s := latest.get("state")) is not None and s != last:
                last = s
                log.write(json.dumps({"frame": n, "state": json.loads(s)}) + "\n")
            n += 1
            tick += 1 / fps
            time.sleep(max(0.0, tick - time.monotonic()))
    video.release()
    print(f"wrote {n} frames ({n / fps:.1f} s) to {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("out", type=Path)
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--fps", type=float, default=15)
    main(**vars(p.parse_args()))

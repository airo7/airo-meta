"""Cue sheets for filming the warehouse: fires commands and camera moves at a running dashboard on a timeline, so
every take of a scene is the same. Start the screen recording during the countdown.

  uv run scripts/director.py memory --port 8001
  uv run scripts/director.py --list
"""

import argparse
import json
import sys
import time
import urllib.request

# (seconds from action, "cam <preset>" or a dashboard command)
SCENES = {
    "deck": [
        (0, "cam deck"), (0, "calm"),
        (2, "airo 1 bring a heavy tote to packing"), (4, "airo 2 bring a light tote to qa"),
        (6, "airo 3 bring a heavy tote to assembly"), (30, "normal"),
    ],
    "memory": [
        (0, "cam deck"), (0, "calm"), (0, "select airo 1"),
        (1, "airo 1 bring a heavy tote to packing"), (14, "cam chase"),
        (40, "select airo 2"), (41, "airo 2 bring a heavy tote to qa"), (60, "cam chase"),
    ],
    "people": [
        (0, "cam deck"), (0, "normal"), (1, "airo 1 bring a light tote to packing"),
        (2, "airo 2 bring a heavy tote to dock"), (8, "send a worker across"), (18, "drop a carton"),
        (30, "send a worker across"), (40, "clear the aisle"),
    ],
    "change": [
        (0, "select airo 1"), (0, "cam chase"), (1, "airo 1 bring a light tote to packing"),
        (12, "airo 1 weight"), (40, "cam iso"), (55, "airo 1 remove weight"),
    ],
    "voice": [
        (0, "cam deck"), (0, "busy"), (2, "airo 3 bring a heavy tote to packing"), (6, "jam at packing"),
        (20, "spill"), (35, "clean"),
    ],
}


def post(base: str, path: str, body: dict) -> None:
    req = urllib.request.Request(base + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=3).read()


def run(base: str, cues: list[tuple[float, str]], countdown: int) -> None:
    for n in range(countdown, 0, -1):
        print(f"{n}…", flush=True)
        time.sleep(1)
    print("action")
    t0 = time.monotonic()
    for at, cue in sorted(cues, key=lambda c: c[0]):
        time.sleep(max(0.0, t0 + at - time.monotonic()))
        if cue.startswith("cam "):
            post(base, "/camera", {"preset": cue[4:]})
        else:
            post(base, "/cmd", {"text": cue})
        print(f"{time.monotonic() - t0:5.1f}s  {cue}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("scene", nargs="?", choices=sorted(SCENES))
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--countdown", type=int, default=5)
    p.add_argument("--list", action="store_true")
    a = p.parse_args()
    if a.list or not a.scene:
        for name, cues in SCENES.items():
            print(f"{name:7} {max(t for t, _ in cues):3.0f}s  " + " · ".join(c for _, c in cues))
        sys.exit(0)
    try:
        run(f"http://127.0.0.1:{a.port}", SCENES[a.scene], a.countdown)
    except OSError as e:
        sys.exit(f"airo: no dashboard on port {a.port}: {e}")

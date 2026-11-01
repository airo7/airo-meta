"""Body library: every body AIRO has been, so a body it meets again is recalled with one probe instead of relearned.
One JSON file per robot model; entries name the robot that learned them, so a fleet can share the pool."""

import copy
import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from airo.body import THRESHOLD, Body, Segment, load, surprise
from airo.paths import DATA

MEMORY_DIR = DATA / "memory"
MODEL = "elegoo-v4"
WINDOW = 0.5


@dataclass
class Entry:
    n: int
    name: str
    cause: str
    body: Body
    members: list[Body]
    verified: bool
    learned_t: float = field(default_factory=time.time)
    recalls: int = 0
    robot: str | None = None
    portrait: str | None = None
    tested: list[str] = field(default_factory=list)

    @classmethod
    def from_json(cls, d: dict) -> "Entry":
        return cls(**{**d, "body": Body(**d["body"]), "members": [Body(**m) for m in d["members"]]})


class Memory:
    def __init__(self, path: Path | None = None, robot: str = "airo-1"):
        self.path, self.robot = path and Path(path), robot
        self.entries: list[Entry] = []
        self.active: dict[str, int] = {}
        if self.path and self.path.exists():
            d = json.loads(self.path.read_text())
            self.entries, self.active = [Entry.from_json(e) for e in d["entries"]], d["active"]

    @classmethod
    def open(cls, body_path: Path, robot: str = "airo-1", model: str = MODEL) -> "Memory":
        """The library for this rig kind (data/body_sim.json -> data/memory/elegoo-v4_sim.json). A body saved
        before memory existed becomes the first, active entry."""
        mem = cls(MEMORY_DIR / f"{model}{Path(body_path).stem.removeprefix('body')}.json", robot)
        if not mem.entries and (saved := load(body_path)):
            body, members, name = saved
            mem.add(name, "learned from scratch", body, members, True)
        return mem

    def view(self, robot: str) -> "Memory":
        """Another robot's handle on the same pool: shared entries and file, its own active body."""
        v = copy.copy(self)
        v.robot = robot
        return v

    def inherit(self) -> Entry | None:
        """A robot new to the pool starts from the fleet's first confirmed body, untested on itself."""
        if self.current is None and (e := min((e for e in self.entries if e.verified), key=lambda e: e.n,
                                              default=None)):
            self.active[self.robot] = e.n
            self.save()
        return self.current

    @property
    def current(self) -> Entry | None:
        return self.get(self.active.get(self.robot))

    def get(self, n: int | None) -> Entry | None:
        return next((e for e in self.entries if e.n == n), None)

    def next_n(self) -> int:
        return max((e.n for e in self.entries), default=0) + 1

    def add(self, name: str, cause: str, body: Body, members: list[Body], verified: bool,
            n: int | None = None) -> Entry:
        e = Entry(n or self.next_n(), name, cause, body, list(members), verified, robot=self.robot,
                  tested=[self.robot] if verified else [])
        self.entries.append(e)
        self.active[self.robot] = e.n
        self.save()
        return e

    def update(self, e: Entry, name: str, body: Body, members: list[Body], verified: bool):
        """A partly learned entry completed by a later relearn keeps its number."""
        e.name, e.body, e.members, e.verified = name, body, list(members), verified
        if verified and self.robot not in e.tested:
            e.tested.append(self.robot)
        self.active[self.robot] = e.n
        self.save()

    def recall(self, e: Entry):
        e.recalls += 1
        self.active[self.robot] = e.n
        if self.robot not in e.tested:
            e.tested.append(self.robot)
        self.save()

    def match(self, seg: Segment, current: list[Body] | None = None, exclude: int | None = None,
              threshold: float = THRESHOLD) -> list[tuple[Entry, float]]:
        """Entries whose ensemble explains seg (median window surprise under threshold), best first. Windows the
        current body still explains come from before the change, so they are left out. The change falls inside the
        first window it cannot explain or before it, so each entry is scored from both starts and keeps the better."""
        starts = [since_change(current, seg, threshold, strict=s) for s in (False, True)] if current else [seg]
        out = []
        for e in self.entries:
            if e.n != exclude and (m := min(_score(e.members, cut) for cut in starts)) < threshold:
                out.append((e, m))
        return sorted(out, key=lambda c: c[1])

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"active": self.active, "entries": [asdict(e) for e in self.entries]},
                                            indent=1))


def since_change(members: list[Body], seg: Segment, threshold: float = THRESHOLD, strict: bool = False) -> Segment:
    """The end of seg from the first window members cannot explain; windows are counted back from the end so the
    latest motion is scored. strict also drops that window when a later one exists, as the change may fall inside
    it: right for scoring, but a refit needs the data."""
    seg = _since(seg, seg.t[-1] - int((seg.t[-1] - seg.t[0]) / WINDOW + 1e-6) * WINDOW)
    s = surprise(members, seg)
    bad = [i for i, v in enumerate(s) if v > threshold]
    return _since(seg, seg.t[0] + min(bad[0] + strict, len(s) - 1) * WINDOW) if bad else seg


def _score(members: list[Body], seg: Segment) -> float:
    s = surprise(members, seg)
    s = s[np.isfinite(s)]
    return float(np.median(s)) if len(s) else math.inf


def _since(seg: Segment, t: float) -> Segment:
    i = int(np.searchsorted(seg.t, t - 1e-6))
    return Segment(seg.cmd_t, seg.cmd, seg.t[i:], seg.pose[i:])

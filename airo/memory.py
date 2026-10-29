"""Body library: every body AIRO has been, so a body it meets again is recalled with one probe instead of relearned.
One JSON file per robot model; entries name the robot that learned them, so a fleet can share the pool."""

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from airo.body import THRESHOLD, Body, Segment, load, surprise

MEMORY_DIR = Path("data/memory")
MODEL = "elegoo-v4"


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

    @property
    def current(self) -> Entry | None:
        return self.get(self.active.get(self.robot))

    def get(self, n: int | None) -> Entry | None:
        return next((e for e in self.entries if e.n == n), None)

    def next_n(self) -> int:
        return max((e.n for e in self.entries), default=0) + 1

    def add(self, name: str, cause: str, body: Body, members: list[Body], verified: bool,
            n: int | None = None) -> Entry:
        e = Entry(n or self.next_n(), name, cause, body, list(members), verified, robot=self.robot)
        self.entries.append(e)
        self.active[self.robot] = e.n
        self.save()
        return e

    def recall(self, e: Entry):
        e.recalls += 1
        self.active[self.robot] = e.n
        self.save()

    def match(self, seg: Segment, exclude: int | None = None, threshold: float = THRESHOLD) -> list[tuple[Entry, float]]:
        """Entries whose ensemble explains seg (median window surprise under threshold), best first."""
        out = []
        for e in self.entries:
            s = surprise(e.members, seg) if e.n != exclude else []
            s = np.asarray(s)[np.isfinite(s)]
            if len(s) and (m := float(np.median(s))) < threshold:
                out.append((e, m))
        return sorted(out, key=lambda c: c[1])

    def save(self):
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps({"active": self.active, "entries": [asdict(e) for e in self.entries]},
                                            indent=1))

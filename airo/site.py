"""Sites: the floor a fleet works on. Station names map to SAM concepts, so missions, voice and the planner speak
the site's language ("deliver to packing" on the factory, "deliver to blue" at home)."""

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from airo.planner import OBSTACLE, PARCEL, STATIONS
from airo.sim import Thing, world

COLOURS = ("#ffb340", "#45c4ff", "#ff6fa8", "#8fe36b")
TAGS = ("orange", "blue", "pink", "green")
# Named by what SAM sees from above: painted squares, not words.
FACTORY_STATIONS = {"assembly": "blue square", "qa": "green square", "packing": "orange square", "dock": "red square"}
TOTE = "yellow tote"
KINDS = {"light": 0.2, "heavy": 0.8}
UP, RIGHT = math.pi / 2, 0.0


@dataclass(frozen=True)
class Bay:
    """Where a car parks to be loaded or unloaded: pose (x, y, heading). The loader stands beside the car, on its
    left for side 1 and its right for side -1; wait holds the queue spots in front of the bay, nearest first."""
    x: float
    y: float
    theta: float
    side: int = 1
    wait: tuple[tuple[float, float, float], ...] = ()

    @property
    def pose(self) -> tuple[float, float, float]:
        return self.x, self.y, self.theta

    def beside(self, d: float) -> tuple[float, float]:
        """The floor point d metres to the loader's side of the parked car's centre."""
        s = self.side * d
        return self.x - s * math.sin(self.theta), self.y + s * math.cos(self.theta)


@dataclass(frozen=True)
class Site:
    """starts are robot poses (on a carry site, their charging docks); totes[k] is robot k's parcel concept when
    pushing; spill is the slippery zone (x0, y0, x1, y1). mode "push" moves a parcel with the bumper, "carry" has
    loaders put a tote on the car at the pick bay and take it off at each station's drop bay. pose "aruco" tracks a
    single car's marker in rendered frames, "truth" reads physics plus noise."""
    name: str
    floor: tuple[float, float]
    stations: dict[str, str]
    objects: Callable[[], list[Thing]]
    starts: tuple[tuple[float, float, float], ...]
    totes: tuple[str, ...]
    spill: tuple[float, float, float, float]
    spill_label: str = "Slippery patch"
    pose: str = "truth"
    cam_h: float = 1.8
    frame: tuple[int, int] = (1920, 1080)
    scene: str = "home"
    max_robots: int = 1
    mode: str = "push"
    pick: Bay | None = None
    drops: dict[str, Bay] = field(default_factory=dict)
    kinds: dict[str, str] = field(default_factory=dict)

    def station_xy(self, objects) -> dict[str, tuple[float, float]]:
        by_concept = {o.concept: o.xy for o in objects}
        return {name: by_concept[c] for name, c in self.stations.items() if c in by_concept}


def robot(k: int) -> dict:
    """Identity of robot k (0-based), as the dashboard shows it."""
    return {"id": f"airo-{k + 1}", "name": f"AIRO {k + 1}", "color": COLOURS[k % len(COLOURS)]}


def tote(k: int) -> str:
    return f"yellow box with a {TAGS[k % len(TAGS)]} tag"


# The fulfilment cell: totes come in on a conveyor at the left wall, an arm loads them onto the car in the pick bay,
# and an arm at each station's drop bay takes them off. Cars charge at docks along the bottom edge and queue below
# the pick bay. Every loader stands outside the floor's driving inset.
PICK = Bay(0.36, 1.05, UP, 1, wait=((0.36, 0.64, UP), (0.36, 0.3, UP)))
DROPS = {"assembly": Bay(1.15, 1.68, RIGHT, 1, wait=((0.72, 1.6, RIGHT),)),
         "qa": Bay(2.05, 1.68, RIGHT, 1, wait=((1.6, 1.42, RIGHT),)),
         "packing": Bay(2.66, 0.95, UP, -1, wait=((2.66, 0.5, UP),)),
         "dock": Bay(2.15, 0.32, RIGHT, -1, wait=((1.78, 0.68, RIGHT),))}
DOCKS = ((0.7, 0.3, UP), (1.06, 0.3, UP), (1.42, 0.3, UP))
PAD_BGR = {"assembly": (176, 110, 38), "qa": (96, 150, 52), "packing": (36, 120, 226), "dock": (58, 58, 196)}


def factory() -> list[Thing]:
    """The station pads under the drop bays, and a pallet left in the aisle that the planner has to go around."""
    return [*(Thing(FACTORY_STATIONS[n], (b.x, b.y), (0.3, 0.3), PAD_BGR[n], kind="pad") for n, b in DROPS.items()),
            Thing(OBSTACLE, (2.2, 1.24), (0.2, 0.16), (60, 110, 160), kind="pallet")]


HOME = Site("Home", (2.0, 1.4), STATIONS, world, starts=((1.0, 0.7, 0.0),), totes=(PARCEL,),
            spill=(1.2, 0.35, 1.6, 0.85), pose="aruco")
FACTORY = Site("Factory", (3.0, 2.0), FACTORY_STATIONS, factory, starts=DOCKS, totes=(), spill=(1.7, 0.66, 2.2, 1.06),
               spill_label="Oil spill", cam_h=2.6, frame=(1280, 720), scene="factory", max_robots=len(DOCKS),
               mode="carry", pick=PICK, drops=DROPS,
               kinds={"assembly": "heavy", "qa": "light", "packing": "heavy", "dock": "light"})
SITES = {"home": HOME, "factory": FACTORY}

"""Sites: the floor a fleet works on. Station names map to SAM concepts, so missions, voice and the planner speak
the site's language ("deliver to packing" on the factory, "deliver to blue" at home)."""

from collections.abc import Callable
from dataclasses import dataclass

from airo.planner import OBSTACLE, PARCEL, STATIONS
from airo.sim import Thing, world

COLOURS = ("#ffb340", "#45c4ff", "#ff6fa8", "#8fe36b")
TAGS = ("orange", "blue", "pink", "green")
LANES = (1.45, 1.0, 0.55)
TOTE_X = 1.15
FACTORY_STATIONS = {n: f"{n} station" for n in ("assembly", "qa", "packing", "dock")}


@dataclass(frozen=True)
class Site:
    """starts are robot poses; totes[k] is robot k's parcel concept; spill is the slippery zone (x0, y0, x1, y1).
    pose "aruco" tracks a single car's marker in rendered frames, "truth" reads physics plus noise."""
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

    def station_xy(self, objects) -> dict[str, tuple[float, float]]:
        by_concept = {o.concept: o.xy for o in objects}
        return {name: by_concept[c] for name, c in self.stations.items() if c in by_concept}


def robot(k: int) -> dict:
    """Identity of robot k (0-based), as the dashboard shows it."""
    return {"id": f"airo-{k + 1}", "name": f"AIRO {k + 1}", "color": COLOURS[k % len(COLOURS)]}


def tote(k: int) -> str:
    return f"yellow box with a {TAGS[k % len(TAGS)]} tag"


def factory() -> list[Thing]:
    """Three lanes: each robot parks at the left with its tote 0.9 m ahead in the pick zone, far enough that the run-in
    behind the tote is a gentle arc. The natural deliveries (1 to assembly, 2 to packing, 3 to the dock) don't cross."""
    pads = [("assembly", (1.85, 1.72), (176, 110, 38)), ("qa", (2.6, 1.72), (96, 150, 52)),
            ("packing", (2.72, 1.05), (36, 120, 226)), ("dock", (2.55, 0.3), (58, 58, 196))]
    return [*(Thing(FACTORY_STATIONS[n], xy, (0.3, 0.3), bgr, kind="pad") for n, xy, bgr in pads),
            *(Thing(tote(k), (TOTE_X, y), (0.1, 0.1), (40, 195, 235), kind="tote") for k, y in enumerate(LANES)),
            Thing(OBSTACLE, (2.2, 1.32), (0.2, 0.16), (60, 110, 160), kind="pallet")]


HOME = Site("Home", (2.0, 1.4), STATIONS, world, starts=((1.0, 0.7, 0.0),), totes=(PARCEL,),
            spill=(1.2, 0.35, 1.6, 0.85), pose="aruco")
FACTORY = Site("Factory", (3.0, 2.0), FACTORY_STATIONS, factory, starts=tuple((0.25, y, 0.0) for y in LANES),
               totes=tuple(tote(k) for k in range(len(LANES))), spill=(1.7, 0.72, 2.2, 1.22), spill_label="Oil spill",
               cam_h=2.6, frame=(1280, 720), scene="factory", max_robots=len(LANES))
SITES = {"home": HOME, "factory": FACTORY}

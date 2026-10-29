"""Interactive camera for the MuJoCo twin: view and part presets, orbit, zoom, pan and picking, eased on the wall
clock so it stays live while the simulation is paused."""

import math
import re
import time
from dataclasses import dataclass, replace

import mujoco
import numpy as np

from airo.site import robot


@dataclass(frozen=True)
class Shot:
    """target is in the car frame when follow, else in the world; spin keeps the azimuth relative to the heading."""
    target: tuple[float, float, float]
    azimuth: float
    elevation: float
    distance: float
    follow: bool = True
    spin: bool = True


VIEWS = {
    "chase": Shot((0.25, 0, 0.03), 30, -27, 1.15),
    "iso": Shot((0, 0, 0.05), 45, -35.26, 0.95, spin=False),
    "top": Shot((0, 0, 0), 90, -89.5, 0.8, spin=False),
    "floor": Shot((0, 0, 0), 60, -48, 2.9, follow=False, spin=False),
}
PARTS = {
    "wheel": Shot((0.065, 0.0725, 0.0325), -110, -12, 0.24),
    "motor": Shot((0, 0.05, 0.035), -90, -4, 0.17),
    "battery": Shot((-0.035, 0, 0.0585), -90, -8, 0.26),
    "board": Shot((-0.008, 0, 0.088), -135, -8, 0.2),
    "sensor": Shot((0.11, 0, 0.105), 200, -18, 0.2),
    "marker": Shot((0, 0, 0.1), 0, -89, 0.32),
}
LABELS = {
    "car": ("Elegoo Smart Robot Car V4", "skid steer, four TT motors, no wheel encoders"),
    "chassis": ("Acrylic decks", "two plates on brass standoffs"),
    "motors": ("TT gear motors", "1:48 DC gear motors; one PWM channel drives each side"),
    "battery": ("Battery pack", "2 × 18650 cells, 7.4 V"),
    "line_tracker": ("IR line tracker", "three reflectance sensors; AIRO doesn't use them"),
    "wiring": ("Jumper wires", "sensor ribbon, servo and battery leads"),
    "board": ("Arduino UNO + SmartCar shield", "stock firmware: BLE commands, 500 ms watchdog"),
    "marker_card": ("ArUco marker 0", "95 mm; the overhead camera tracks the car by it"),
    "sensor": ("HC-SR04 on an SG90 servo", "ultrasonic eyes; AIRO doesn't use them"),
    "bumper": ("V bumper", "cardboard; cups the parcel toward the centre line"),
    "tow": ("Towed box", "on a 25 cm string from the rear hitch"),
    "tray": ("Tote tray", "lipped deck; a latch holds the tote while it rides"),
}
DECOR = {
    "rack": ("Pallet racking", "selective racking, pallets of stretch-wrapped cartons and crates"),
    "conveyor": ("Inbound conveyor", "powered rollers; totes arrive from the line"),
    "portal": ("Conveyor portal", "strip curtain where the line comes through the wall"),
    "pillar": ("Column", "hazard-wrapped at the base"),
    "goods": ("Pallet of goods", "stretch-wrapped cartons waiting for the dock"),
    "spill_sign": ("Wet floor sign", "the spill has been flagged"),
    "wall": ("Wall", "insulated cladding"),
    "fence": ("Safety guarding", "polycarbonate panels around a loader cell"),
    "cell": ("Loader cell", "light curtain at the open side, stack light, the station's table"),
    "charger": ("Charging dock", "contacts behind the parking bay"),
    "lamp": ("High-bay lamp", "LED, under the roof"),
    "door": ("Fire exit", "the hatched floor in front stays clear"),
    "sign": ("Station sign", ""),
}
WHEELS = {"fl": "Front-left", "rl": "Rear-left", "fr": "Front-right", "rr": "Rear-right"}
EASE = 0.15


class Orbit:
    def __init__(self, rig, size: tuple[int, int] = (1280, 720), preset: str = "chase"):
        self.rig, self.size = rig, size
        self.cam = mujoco.MjvCamera()
        self.preset: str | None = None
        self._t: float | None = None
        self._active = 0.0
        self.set(preset)

    def set(self, name: str):
        shot = VIEWS.get(name) or PARTS[name]
        if name == "floor":
            w, h = self.rig.floor
            shot = replace(shot, target=(w / 2, h / 2, 0), distance=shot.distance * max(1.0, w / 2.0, h / 1.4))
        self.shot, self.preset = shot, name

    def follow(self, rig):
        """Follow another car in the same world; the camera eases across to it."""
        self.rig = rig

    @property
    def moving(self) -> bool:
        return time.monotonic() < self._active

    def handle(self, req: dict) -> dict:
        part = None
        s = self.shot
        if (name := req.get("preset")) in VIEWS or name in PARTS:
            self.set(name)
            s = self.shot
        elif "orbit" in req:
            daz, de = req["orbit"]
            s = replace(s, azimuth=s.azimuth + daz, elevation=float(np.clip(s.elevation + de, -89.5, -1)))
        elif "zoom" in req:
            s = replace(s, distance=float(np.clip(s.distance * req["zoom"], 0.06, 6)))
        elif "pan" in req:
            s = replace(s, target=tuple(map(float, np.array(s.target) + self._pan(*req["pan"]))))
        elif "pick" in req:
            part, s = self._pick(*req["pick"], focus=req.get("focus", False))
        if s is not self.shot:
            self.shot, self.preset = s, None
        self._active = time.monotonic() + 1.0
        return {"camera": {"preset": self.preset, "follow": self.shot.follow}, "part": part}

    def update(self) -> mujoco.MjvCamera:
        x, y, th = self.rig.true_pose
        s, c = self.shot, self.cam
        lookat = np.array(s.target, float)
        if s.follow:
            lookat[:2] = _rot(th) @ lookat[:2] + (x, y)
        az = s.azimuth + (math.degrees(th) if s.spin else 0)
        now = time.monotonic()
        k = 1.0 if self._t is None else 1 - math.exp(-(now - self._t) / EASE)
        self._t = now
        c.lookat[:] += k * (lookat - c.lookat)
        c.azimuth += k * ((az - c.azimuth + 180) % 360 - 180)
        c.elevation += k * (s.elevation - c.elevation)
        c.distance += k * (s.distance - c.distance)
        return c

    def _pan(self, dx: float, dy: float) -> np.ndarray:
        """Drag in image fractions to a target shift: the scene follows the pointer."""
        s, th = self.shot, self.rig.true_pose[2]
        a, e = math.radians(s.azimuth) + (th if s.spin else 0), math.radians(s.elevation)
        forward = np.array([math.cos(e) * math.cos(a), math.cos(e) * math.sin(a), math.sin(e)])
        right = np.array([math.sin(a), -math.cos(a), 0])
        up = np.cross(right, forward)
        h = 2 * s.distance * math.tan(math.radians(self.rig.m.vis.global_.fovy) / 2)
        w = h * self.size[0] / self.size[1]
        d = -dx * w * right + dy * h * up
        if s.follow:
            d[:2] = _rot(-th) @ d[:2]
        return d

    def _pick(self, u: float, v: float, focus: bool) -> tuple[dict | None, Shot]:
        hit = self.rig.select(u, v, *self.size)
        if hit is None:
            return None, self.shot
        body, geom, point = hit
        part = self._label(body, geom, point)
        if not focus:
            return part, self.shot
        x, y, th = self.rig.true_pose
        m = self.rig.m
        if m.body_rootid[body] == m.body(self.rig.prefix + "car").id:
            local = tuple(map(float, (*(_rot(-th) @ (point[:2] - (x, y))), point[2])))
            shot = Shot(local, self.cam.azimuth - math.degrees(th), self.cam.elevation, min(self.shot.distance, 0.22))
        else:
            shot = Shot(tuple(map(float, point)), self.cam.azimuth, self.cam.elevation, min(self.shot.distance, 0.7),
                        follow=False, spin=False)
        return part, shot

    def _label(self, body: int, geom: int, point) -> dict:
        """Names the part under the pointer, and on a fleet which robot it belongs to ("AIRO 2 · Front-left wheel")."""
        m, world = self.rig.m, self.rig.world
        car = world.car_of(body)
        name = m.body(body).name.removeprefix(car.prefix if car else "")
        geom_name = m.geom(geom).name
        if name in WHEELS:
            label = (f"{WHEELS[name]} wheel", "65 mm rubber tyre on a TT gear motor")
        elif name.startswith("weight"):
            label = ("Steel weight", f"{m.body_mass[body]:.1f} kg, strapped on with velcro")
        elif name.startswith("thing") or geom_name.startswith("thing"):
            label = self._thing(int((name if name.startswith("thing") else geom_name)[5:]))
        elif name in LABELS:
            label = LABELS[name]
        elif world.cell is not None and (part := world.cell.label(name)):
            label = part
        elif (kind := re.sub(r"(_back|_left|\d+)$", "", name if name != "world" else geom_name)) in DECOR:
            label = DECOR[kind]
        elif (kind := kind.split("_")[0]) in DECOR:
            label = DECOR[kind]
        else:
            mat = m.geom_matid[geom]
            material = m.material(mat).name if mat >= 0 else ""
            x0, y0, x1, y1 = world.zone or (0, 0, -1, -1)
            label = (("Slippery patch", "low-friction glossy floor") if material == "glossy" else
                     ("Corner marker", "ArUco; fixes the floor frame") if material.startswith(("aruco", "paper")) else
                     (world.site.spill_label, "low friction: the tyres slip here")
                     if x0 <= point[0] <= x1 and y0 <= point[1] <= y1 else ("Floor", ""))
        if car and car.prefix:
            label = (f"{car.name} · {label[0]}", label[1])
        return {"name": label[0], "detail": label[1]}

    def _thing(self, i: int) -> tuple[str, str]:
        site = self.rig.world.site
        o = self.rig.world.objects[i]
        station = next((k for k, v in site.stations.items() if v == o.concept), None)
        if station:
            title = station.upper() if len(station) <= 2 else station.capitalize()
            return f"{title} station", "painted pad; deliveries end here" if o.kind == "pad" else \
                "A4 sheet; deliveries end here"
        if o.concept in site.totes and o.kind == "tote":
            return "Tote", f"{robot(site.totes.index(o.concept))['name']}'s; pushed with the bumper"
        if o.concept in site.totes:
            return "Parcel", "yellow box, pushed with the bumper"
        if o.kind == "pallet":
            return "Pallet", "stretch-wrapped cartons; the planner keeps clear of it"
        return "Obstacle", "a box the planner keeps clear of"


def _rot(th: float) -> np.ndarray:
    c, s = math.cos(th), math.sin(th)
    return np.array([[c, -s], [s, c]])

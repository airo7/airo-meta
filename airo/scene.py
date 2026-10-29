"""MuJoCo scenes for a site: the home marker floor the real car drives on, and a scale-model factory floor. Every
texture (ArUco markers, painted concrete, station labels, signs) is drawn with cv2."""

import math
from itertools import pairwise
from pathlib import Path

import cv2
import mujoco
import numpy as np

from airo.camera import CORNER_MARKER_M, DICT, floor_layout

ASSETS = Path(__file__).parent / "assets"
GEOM = mujoco.mjtGeom
TEX2D = mujoco.mjtTexture.mjTEXTURE_2D
PX = 400
MARGIN = 1.0
STATIC = {"contype": 0, "conaffinity": 0}
YELLOW, WHITE, BLACK = (20, 190, 245), (236, 238, 238), (24, 24, 24)


def marker(i: int, cell: int = 40) -> np.ndarray:
    """Marker i with a one-cell white quiet zone: a geom of side 8/6 m carries a marker of side m."""
    img = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(DICT, i, 6 * cell), *[cell] * 4, cv2.BORDER_CONSTANT,
                             value=255)
    return np.repeat(img[..., None], 3, axis=2)


def quat(theta: float) -> list[float]:
    return [math.cos(theta / 2), 0, 0, math.sin(theta / 2)]


def rgba(bgr, a=1.0) -> list[float]:
    return [bgr[2] / 255, bgr[1] / 255, bgr[0] / 255, a]


def rounded(spec, name: str, half) -> str:
    spec.add_mesh(name=name, scale=list(half)).make_supersphere(16, 0.12, 0.08)
    return name


def texture(spec, name: str, bgr: np.ndarray, **material) -> str:
    """A 2D texture and a material of the same name. Row 0 of the image lands at the texture's top (+v)."""
    rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    spec.add_texture(name=name, type=TEX2D, width=rgb.shape[1], height=rgb.shape[0], nchannel=3, data=rgb.ravel())
    spec.add_material(name=name, textures=["", name], **material)
    return name


def build(site, objects) -> mujoco.MjSpec:
    if site.scene == "factory":
        spec = mujoco.MjSpec.from_file(str(ASSETS / "factory.xml"))
        _factory(spec, site.floor)
    else:
        spec = mujoco.MjSpec.from_file(str(ASSETS / "world.xml"))
        _home(spec, site.floor)
    _things(spec, objects, site)
    cam = spec.camera("overhead")
    w, h = site.floor
    cam.pos = [w / 2, h / 2, site.cam_h]
    aspect = site.frame[1] / site.frame[0]
    cam.fovy = math.degrees(2 * math.atan(max(h / 2 + 0.2, (w / 2 + 0.2) * aspect) / site.cam_h))
    return spec


def add_car(spec, prefix: str, pose, marker_id: int, accent=None):
    """The home world includes the one real car unprefixed; fleet cars are attached copies of car.xml."""
    if prefix:
        car = mujoco.MjSpec.from_file(str(ASSETS / "car.xml"))
        img = marker(marker_id)
        car.add_texture(name="aruco0", type=TEX2D, width=img.shape[1], height=img.shape[0], nchannel=3,
                        data=img.ravel())
        if accent is not None:
            car.material("accent").rgba = accent
            car.material("label").rgba = accent
            plate = next(g for g in car.geoms if g.material == "accent")
            plate.size = [0.072, 0.072, plate.size[2]]
        body = spec.worldbody.add_frame().attach_body(car.body("car"), prefix, "")
    else:
        body = spec.body("car")
    x, y, th = pose
    body.pos, body.quat = [x, y, 0.0005], quat(th)
    return body


def _home(spec, floor):
    for i in range(5):
        img = marker(i)
        spec.add_texture(name=f"aruco{i}", type=TEX2D, width=img.shape[1], height=img.shape[0], nchannel=3,
                         data=img.ravel())
    side = CORNER_MARKER_M * 8 / 6
    for i, (x, y) in floor_layout(*floor).items():
        spec.add_material(name=f"aruco{i}", textures=["", f"aruco{i}"])
        spec.worldbody.add_geom(type=GEOM.mjGEOM_BOX, size=[0.105, 0.1485, 0.0002], pos=[x, y, 0.0002],
                                material="paper")
        spec.worldbody.add_geom(type=GEOM.mjGEOM_PLANE, size=[side / 2, side / 2, 1], pos=[x, y, 0.0005],
                                material=f"aruco{i}", **STATIC)


def _things(spec, objects, site):
    wb = spec.worldbody
    stations = set(site.stations.values())
    for i, o in enumerate(objects):
        sx, sy = np.array(o.size) / 2
        if o.kind == "pad":
            name = next(k for k, v in site.stations.items() if v == o.concept)
            texture(spec, f"pad{i}", pad(name, o.colour), specular=0.2, shininess=0.3)
            wb.add_geom(name=f"thing{i}", type=GEOM.mjGEOM_PLANE, size=[sx, sy, 1], pos=[*o.xy, 0.0004],
                        material=f"pad{i}", **STATIC)
        elif o.concept in stations:
            spec.add_material(name=f"sheet{i}", rgba=rgba(o.colour), specular=0.15, shininess=0.2)
            wb.add_geom(name=f"thing{i}", type=GEOM.mjGEOM_BOX, size=[sx, sy, 0.0002], pos=[*o.xy, 0.0002],
                        material=f"sheet{i}")
        elif o.kind == "pallet":
            _pallet(spec, wb.add_body(name=f"thing{i}", pos=[*o.xy, 0]), sx, sy, collide=True)
        else:
            _box(spec, i, o, sx, sy)


def _box(spec, i, o, sx, sy):
    """A free box: cardboard, the yellow parcel with a paper label, or a yellow plastic tote with its robot's tag."""
    h = float(np.clip(0.6 * max(o.size), 0.06, 0.15)) / 2
    b = spec.worldbody.add_body(name=f"thing{i}", pos=[*o.xy, h])
    b.add_freejoint()
    parcel = o.concept.startswith("yellow box")
    if parcel:
        spec.add_material(name=f"box{i}", rgba=rgba(o.colour), specular=0.35 if o.kind else 0.1,
                          shininess=0.4 if o.kind else 0.15)
    b.add_geom(type=GEOM.mjGEOM_MESH, meshname=rounded(spec, f"thing{i}", (sx, sy, h)),
               material=f"box{i}" if parcel else "cardboard", mass=0.12 if parcel else 0.35,
               friction=[0.45, 0.005, 0.0001])
    if o.kind == "tote":
        tag = spec.add_material(name=f"tag{i}", rgba=rgba(_tag(o.concept)), specular=0.3, shininess=0.4).name
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx + 0.002, sy + 0.002, 0.004], pos=[0, 0, h - 0.004],
                   material="tote_rim", mass=0, **STATIC)
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx * 0.7, sy * 0.3, 0.0004], pos=[0, 0, h], material=tag, mass=0,
                   **STATIC)
        for side in (1, -1):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.0004, sy * 0.5, h * 0.3], pos=[side * (sx + 0.0002), 0, 0],
                       material=tag, mass=0, **STATIC)
        return
    b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx + 0.0004, 0.024, h + 0.0004], material="tape", mass=0, **STATIC)
    if parcel:
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx * 0.55, sy * 0.4, 0.0003], pos=[-sx * 0.1, sy * 0.4, h],
                   material="paper", mass=0, **STATIC)


def _tag(concept: str):
    from airo.site import COLOURS, TAGS

    c = COLOURS[TAGS.index(concept.split(" a ")[-1].split()[0])]
    return int(c[5:7], 16), int(c[3:5], 16), int(c[1:3], 16)


# the factory


def _factory(spec, floor):
    w, h = floor
    wb = spec.worldbody
    rng = np.random.default_rng(7)
    texture(spec, "hall", concrete((512, 512), rng, joints=False), specular=0.25, shininess=0.3, texrepeat=[1, 1],
            texuniform=True)
    wb.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=[12, 12, 0.05], pos=[w / 2, h / 2, 0], material="hall",
                friction=[0.3, 0.001, 0.00001])
    sw, sh = w + 2 * MARGIN, h + 2 * MARGIN
    texture(spec, "workfloor", workfloor(floor, rng), specular=0.35, shininess=0.45)
    wb.add_geom(name="workfloor", type=GEOM.mjGEOM_PLANE, size=[sw / 2, sh / 2, 1], pos=[w / 2, h / 2, 0.0001],
                material="workfloor", **STATIC)
    texture(spec, "wall", wall((1024, 384), rng), specular=0.15, shininess=0.2)
    for name, pos, size, euler in [("wall_back", [w / 2, h + MARGIN, 0.45], [w / 2 + MARGIN, 0.45, 1], [90, 0, 0]),
                                   ("wall_left", [-MARGIN, h / 2, 0.45], [h / 2 + MARGIN, 0.45, 1], [90, 90, 0])]:
        wb.add_geom(name=name, type=GEOM.mjGEOM_PLANE, size=size, pos=pos, euler=euler, material="wall", **STATIC)
    for k, x0 in enumerate(np.arange(-0.55, w - 0.2, 0.75)):
        _rack(spec, wb.add_body(name=f"rack{k}", pos=[x0, h + 0.62, 0]), rng)
    _conveyor(spec, wb.add_body(name="conveyor", pos=[-0.06, 1.0, 0]))
    for k, (x, y) in enumerate([(-0.35, -0.35), (w + 0.35, -0.35), (w + 0.35, h + 0.3), (-0.35, h + 0.3)]):
        _pillar(wb.add_body(name=f"pillar{k}", pos=[x, y, 0]))
    for k, (x, y, th) in enumerate([(w + 0.42, 0.25, 0.1), (w + 0.4, 0.62, -0.05), (w - 0.35, -0.45, 1.6)]):
        _pallet(spec, wb.add_body(name=f"goods{k}", pos=[x, y, 0], quat=quat(th)), 0.11, 0.09, rng=rng)


def _rack(spec, b, rng, bay=(0.72, 0.28), levels=(0.03, 0.27, 0.5)):
    """One bay of pallet racking: blue uprights, orange beams, wire decks, cartons and crates."""
    bx, by = bay
    for x in (0.006, bx - 0.006):
        for y in (-by / 2, by / 2):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.008, 0.008, 0.37], pos=[x, y, 0.37], material="upright",
                       **STATIC)
    for x in (0.006, bx - 0.006):
        for z in np.arange(0.08, 0.7, 0.12):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.003, by / 2, 0.003], pos=[x, 0, z], material="upright",
                       **STATIC)
    for z in levels:
        for y in (-by / 2, by / 2):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[bx / 2, 0.006, 0.012], pos=[bx / 2, y, z + 0.012],
                       material="beam", **STATIC)
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[bx / 2 - 0.01, by / 2, 0.002], pos=[bx / 2, 0, z + 0.024],
                   material="deck", **STATIC)
        x = 0.03
        while True:
            crate = rng.random() < 0.25
            s = np.array([0.1, 0.16, 0.08]) if crate else rng.uniform([0.08, 0.12, 0.06], [0.16, 0.2, 0.14])
            if x + s[0] > bx - 0.02:
                break
            if rng.random() < 0.85:
                pos = [x + s[0] / 2, rng.uniform(-0.02, 0.02), z + 0.026 + s[2] / 2]
                b.add_geom(type=GEOM.mjGEOM_BOX, size=list(s / 2), pos=pos,
                           material=rng.choice(["crate_blue", "crate_grey"]) if crate else "cardboard", **STATIC)
                if not crate:
                    b.add_geom(type=GEOM.mjGEOM_BOX, size=[s[0] / 2 + 0.0005, 0.02, s[2] / 2 + 0.0005], pos=pos,
                               material="tape", **STATIC)
            x += s[0] + rng.uniform(0.01, 0.03)


def _conveyor(spec, b, length=0.9, width=0.18, top=0.12):
    """The end of a roller conveyor running in from the left wall, a tote waiting at its end."""
    for y in (-width / 2, width / 2):
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[length / 2, 0.006, 0.018], pos=[-length / 2, y, top - 0.01],
                   material="steel", **STATIC)
        for x in (-0.06, -length + 0.06):
            b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.01, 0.01, top / 2], pos=[x, y, top / 2], material="steel",
                       **STATIC)
    for x in np.arange(-length + 0.03, 0, 0.045):
        b.add_geom(type=GEOM.mjGEOM_CYLINDER, size=[0.014, width / 2 - 0.006], pos=[x, 0, top],
                   euler=[90, 0, 0], material="roller", **STATIC)
    b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.006, width / 2 + 0.01, 0.02], pos=[0.002, 0, top + 0.012],
               material="hazard_y", **STATIC)
    b.add_geom(type=GEOM.mjGEOM_MESH, meshname=rounded(spec, "conveyor_tote", (0.06, 0.05, 0.035)),
               pos=[-0.28, 0, top + 0.05], material="tote", **STATIC)


def _pillar(b, side=0.07, height=1.6):
    b.add_geom(type=GEOM.mjGEOM_BOX, size=[side, side, height / 2], pos=[0, 0, height / 2],
               material="concrete_col", **STATIC)
    for k in range(6):
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[side + 0.004] * 2 + [0.0125], pos=[0, 0, 0.0125 + 0.025 * k],
                   material="hazard_k" if k % 2 else "hazard_y", **STATIC)


def _pallet(spec, b, sx, sy, collide=False, rng=None):
    """A wooden pallet with stretch-wrapped cartons; the one on the floor is an obstacle the planner keeps clear of."""
    kw = {} if collide else STATIC
    for y in (-sy + 0.012, 0, sy - 0.012):
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx, 0.012, 0.009], pos=[0, y, 0.009], material="wood", **kw)
    for x in np.linspace(-sx + 0.012, sx - 0.012, 5):
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.011, sy, 0.0025], pos=[x, 0, 0.0205], material="wood", **STATIC)
    top, n = 0.023, 2
    for i in range(n):
        for j in range(n):
            for k in range(2 if rng is None else int(rng.integers(1, 4))):
                c = [(-0.5 + i) * sx, (-0.5 + j) * sy, top + (k + 0.5) * 0.06]
                b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx / 2 - 0.003, sy / 2 - 0.003, 0.029], pos=c,
                           material="cardboard", **(kw if k == 0 else STATIC))
    b.add_geom(type=GEOM.mjGEOM_BOX, size=[sx + 0.002, sy + 0.002, 0.062], pos=[0, 0, top + 0.06],
               material="wrap", **STATIC)


def add_spill(spec, rect, friction: float, floor=None) -> bytes | None:
    """Physics: a low-friction box that overrides the tyres' friction inside rect. Looks: on the factory (floor
    given) an oil stain painted into the work floor and a wet floor sign, returning the clean floor texture; at home
    a glossy sheet."""
    x0, y0, x1, y1 = rect
    size, pos = [(x1 - x0) / 2, (y1 - y0) / 2, 4e-4], [(x0 + x1) / 2, (y0 + y1) / 2, 4e-4]
    wb = spec.worldbody
    wb.add_geom(name="spill", type=GEOM.mjGEOM_BOX, size=size, pos=pos, priority=1, condim=6,
                friction=[friction, 0.001, 0.0001], group=3, rgba=[0, 0, 0, 0])
    if floor is None:
        wb.add_geom(name="spill_gloss", type=GEOM.mjGEOM_PLANE, size=size, pos=[*pos[:2], 8.5e-4],
                    material="glossy", **STATIC)
        return None
    if _find(spec.material, "wetfloor") is None:
        texture(spec, "wetfloor", wet_floor_sign(), specular=0.4, shininess=0.5)
    sign = wb.add_body(name="spill_sign", pos=[x1 + 0.07, y0 + 0.06, 0], quat=quat(math.radians(200)))
    for side in (1, -1):
        board = sign.add_body(pos=[side * 0.024, 0, 0.072], euler=[0, -side * 18, 0])
        board.add_geom(type=GEOM.mjGEOM_BOX, size=[0.0015, 0.045, 0.075], material="sign", **STATIC)
        board.add_geom(type=GEOM.mjGEOM_PLANE, size=[0.039, 0.066, 1], pos=[side * 0.0017, 0, 0],
                       quat=_facing(side), material="wetfloor", **STATIC)
    return paint_oil(spec, floor, rect)


def _facing(side: int) -> list[float]:
    """A plane that faces +x (side 1) or -x with its texture upright: local x across, y up, z out."""
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([(0, side, 0), (0, 0, 1), (side, 0, 0)]).ravel().astype(float))
    return q.tolist()


def _find(get, name: str):
    try:
        return get(name)
    except KeyError:
        return None


def remove_spill(spec, clean: bytes | None):
    for g in [g for g in spec.geoms if g.name.startswith("spill")]:
        spec.delete(g)
    if (b := _find(spec.body, "spill_sign")) is not None:
        spec.delete(b)
    if clean is not None:
        spec.texture("workfloor").data = clean


# textures


def _noise(rng, shape, cell: int) -> np.ndarray:
    small = rng.normal(size=(shape[0] // cell + 3, shape[1] // cell + 3)).astype(np.float32)
    big = cv2.resize(small, (small.shape[1] * cell, small.shape[0] * cell), interpolation=cv2.INTER_CUBIC)
    return big[cell:cell + shape[0], cell:cell + shape[1]]


def concrete(shape, rng, joints: bool = True) -> np.ndarray:
    """Polished concrete: broad mottling, fine aggregate, a few worn stains."""
    v = 0.58 + 0.035 * _noise(rng, shape, 160) + 0.02 * _noise(rng, shape, 36) + 0.012 * _noise(rng, shape, 6)
    v += cv2.GaussianBlur(rng.normal(0, 0.03, shape).astype(np.float32), (0, 0), 0.7)
    dots = rng.random(shape) < 0.004
    v[dots] *= rng.uniform(0.75, 1.15, dots.sum())
    stains = np.zeros(shape, np.float32)
    for _ in range(shape[0] * shape[1] // 120_000):
        c = (int(rng.integers(shape[1])), int(rng.integers(shape[0])))
        cv2.ellipse(stains, c, (int(rng.integers(20, 90)), int(rng.integers(12, 50))), float(rng.uniform(0, 180)), 0,
                    360, float(rng.uniform(0.5, 1)), -1)
    v -= 0.05 * cv2.GaussianBlur(stains, (0, 0), 18)
    if joints:
        for x in range(PX, shape[1], PX):
            v[:, x - 1:x + 1] *= 0.72
        for y in range(PX, shape[0], PX):
            v[y - 1:y + 1, :] *= 0.72
    img = np.clip(v, 0, 1)[..., None] * np.array([0.94, 0.97, 1.0], np.float32)
    return (img * 255).astype(np.uint8)


class Canvas:
    """Paint on a floor texture, in floor metres when given an origin; paint wears through a fixed noise mask."""

    def __init__(self, img: np.ndarray, rng, wear: float = 0.12, origin=(0.0, 0.0), px: float = PX):
        self.img = img.astype(np.float32)
        self.origin, self.px = origin, px
        shape = img.shape[:2]
        self.wear = np.clip(0.92 - wear * np.abs(_noise(rng, shape, 12)) - wear * (rng.random(shape) < 0.04), 0, 1)

    def p(self, x: float, y: float) -> tuple[int, int]:
        """Floor metres to pixels; image rows run down, floor y runs up."""
        return round((x - self.origin[0]) * self.px), round((self.origin[1] - y) * self.px)

    def w(self, m: float) -> int:
        return max(1, round(m * self.px))

    def paint(self, colour, draw):
        mask = np.zeros(self.img.shape[:2], np.uint8)
        draw(mask)
        x, y, w, h = cv2.boundingRect(mask)
        if not w:
            return
        part = self.img[y:y + h, x:x + w]
        m = (mask[y:y + h, x:x + w] * self.wear[y:y + h, x:x + w])[..., None]
        part[:] = (part * (1 - m) + np.array(colour, np.float32) * m * (0.75 + part / 255 * 0.4)).clip(0, 255)

    def text(self, text: str, centre, height: float, colour, angle: int = 0):
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = height * self.px / cv2.getTextSize(text, font, 1.0, 2)[0][1]
        thick = max(2, round(scale * 2.2))
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        stamp = np.zeros((th + 8, tw + 8), np.uint8)
        cv2.putText(stamp, text, (4, th + 4), font, scale, 1, thick, cv2.LINE_AA)
        stamp = np.ascontiguousarray(np.rot90(stamp, angle // 90))
        x, y = self.p(*centre)
        x, y = x - stamp.shape[1] // 2, y - stamp.shape[0] // 2

        def put(m):
            m[y:y + stamp.shape[0], x:x + stamp.shape[1]] = stamp
        self.paint(colour, put)

    def arrow(self, a, b, colour=WHITE):
        self.paint(colour, lambda m: cv2.arrowedLine(m, self.p(*a), self.p(*b), 1, self.w(0.028), cv2.LINE_AA,
                                                     tipLength=0.45))

    def done(self) -> np.ndarray:
        return self.img.astype(np.uint8)


def workfloor(floor, rng) -> np.ndarray:
    """The painted work floor: control joints, a hazard-striped border, lane lines, the pick zone, parking bays with
    robot numbers, flow arrows and a walkway, on concrete that runs MARGIN beyond the floor."""
    from airo.site import LANES, TOTE_X

    w, h = floor
    shape = (round((h + 2 * MARGIN) * PX), round((w + 2 * MARGIN) * PX))
    c = Canvas(concrete(shape, rng), rng, origin=(-MARGIN, h + MARGIN))
    p, lw = c.p, c.w
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    stripes = (((xx + yy) // lw(0.025)) % 2).astype(np.uint8)

    def band(m, keep):
        cv2.rectangle(m, p(-0.06, h + 0.06), p(w + 0.06, -0.06), 1, -1)
        cv2.rectangle(m, p(0, h), p(w, 0), 0, -1)
        m &= stripes == keep
    c.paint(YELLOW, lambda m: band(m, 0))
    c.paint(BLACK, lambda m: band(m, 1))
    c.paint((70, 120, 60), lambda m: cv2.rectangle(m, p(-0.4, -0.22), p(w + 0.6, -0.62), 1, -1))
    c.paint(YELLOW, lambda m: [cv2.line(m, p(-0.4, y), p(w + 0.6, y), 1, lw(0.02)) for y in (-0.22, -0.62)])
    c.paint(WHITE, lambda m: [cv2.line(m, p(x, -0.27), p(x + 0.08, -0.57), 1, lw(0.04))
                              for x in np.arange(0.2, 0.9, 0.14)])
    dividers = [(a + b) / 2 for a, b in pairwise(LANES)]
    c.paint(YELLOW, lambda m: [cv2.line(m, p(x, y), p(x + 0.1, y), 1, lw(0.022))
                               for y in dividers for x in np.arange(0.05, TOTE_X + 0.4, 0.18)])
    c.paint(YELLOW, lambda m: cv2.rectangle(m, p(TOTE_X - 0.17, LANES[0] + 0.2), p(TOTE_X + 0.17, LANES[-1] - 0.2), 1,
                                            lw(0.02)))
    c.text("PICK", (TOTE_X, LANES[-1] - 0.29), 0.09, YELLOW)
    def bay(m, cx, cy, hx=0.17, hy=0.13, arm=0.06):
        for sx in (-1, 1):
            for sy in (-1, 1):
                x, y = cx + sx * hx, cy + sy * hy
                cv2.polylines(m, [np.array([p(x, y - sy * arm), p(x, y), p(x - sx * arm, y)])], False, 1, lw(0.015))

    for k, y in enumerate(LANES):
        c.paint(WHITE, lambda m, y=y: bay(m, 0.25, y))
        c.text(str(k + 1), (0.04, y), 0.07, WHITE, angle=90)
        c.arrow((TOTE_X + 0.45, y), (TOTE_X + 0.67, y))
    c.arrow((2.15, 0.5), (2.35, 0.38))
    c.arrow((2.3, 1.58), (2.5, 1.66))
    _tyres(c, [(0.3, y, 2.6, y + rng.uniform(-0.1, 0.1)) for y in LANES])
    return c.done()


def _tyres(c: Canvas, lines):
    """Faint rubber marks where the robots run most."""
    m = np.zeros(c.img.shape[:2], np.float32)
    for x0, y0, x1, y1 in lines:
        for off in (-0.07, 0.07):
            pts = np.array([c.p(x, np.interp(x, [x0, x1], [y0, y1]) + off + 0.01 * math.sin(7 * x))
                            for x in np.linspace(x0, x1, 40)], np.int32)
            cv2.polylines(m, [pts], False, 1.0, max(2, c.w(0.02)), cv2.LINE_AA)
    c.img *= 1 - 0.07 * cv2.GaussianBlur(m, (0, 0), 3)[..., None]


def paint_oil(spec, floor, rect) -> bytes:
    """Stain the work floor with a dark, sheened oil puddle inside rect; returns the clean texture to restore."""
    tex = spec.texture("workfloor")
    clean = bytes(tex.data)
    img = np.frombuffer(clean, np.uint8).reshape(tex.height, tex.width, 3).astype(np.float32)
    rng = np.random.default_rng(3)
    c = Canvas(img, rng, origin=(-MARGIN, floor[1] + MARGIN))
    x0, y0, x1, y1 = rect
    blob = np.zeros(img.shape[:2], np.float32)
    for _ in range(9):
        r = rng.uniform(0.35, 0.6) * min(x1 - x0, y1 - y0) / 2
        cx, cy = rng.uniform(x0 + r, x1 - r), rng.uniform(y0 + r, y1 - r)
        cv2.ellipse(blob, c.p(cx, cy), (c.w(r), c.w(r * rng.uniform(0.6, 1))), float(rng.uniform(0, 180)), 0, 360,
                    1.0, -1)
    blob = cv2.GaussianBlur(blob, (0, 0), c.w(0.02))
    edge = (blob + 0.15 * _noise(rng, blob.shape, c.w(0.025)) - 0.4) * 8 * (blob > 0.1)
    alpha = cv2.GaussianBlur(np.clip(edge, 0, 1), (0, 0), 1.5)[..., None]
    sheen = np.clip(_noise(rng, blob.shape, c.w(0.06)), 0, None)[..., None]
    oil = np.array([20, 17, 13], np.float32) + sheen * np.array([26, 26, 30], np.float32)
    tex.data = (img * (1 - 0.95 * alpha) + oil * 0.95 * alpha).astype(np.uint8).tobytes()
    return clean


def pad(name: str, bgr, px: int = 384) -> np.ndarray:
    """A painted station square: colour field, white keyline, the station name across it."""
    rng = np.random.default_rng(len(name))
    c = Canvas(concrete((px, px), rng, joints=False), rng, wear=0.08, px=1)
    c.paint(bgr, lambda m: m.fill(1))
    c.paint(WHITE, lambda m: cv2.rectangle(m, (px // 16, px // 16), (px - px // 16, px - px // 16), 1,
                                           max(3, px // 40)))
    font, text = cv2.FONT_HERSHEY_DUPLEX, name.upper()
    (tw, th), _ = cv2.getTextSize(text, font, 1, 2)
    scale = min(0.74 * px / tw, 0.26 * px / th)
    thick = max(2, round(scale * 2.4))
    (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
    c.paint(WHITE, lambda m: cv2.putText(m, text, ((px - tw) // 2, (px + th) // 2), font, scale, 1, thick,
                                         cv2.LINE_AA))
    return c.done()


def wet_floor_sign(px=(256, 400)) -> np.ndarray:
    w, h = px
    img = np.full((h, w, 3), (12, 200, 250), np.uint8)
    tri = np.array([[w // 2, int(h * 0.08)], [int(w * 0.9), int(h * 0.5)], [int(w * 0.1), int(h * 0.5)]])
    cv2.polylines(img, [tri], True, BLACK, 10, cv2.LINE_AA)
    cv2.putText(img, "!", (w // 2 - 14, int(h * 0.44)), cv2.FONT_HERSHEY_DUPLEX, 3, BLACK, 9, cv2.LINE_AA)
    for text, y, s in (("CAUTION", 0.66, 1.25), ("WET", 0.79, 1.45), ("FLOOR", 0.92, 1.45)):
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, s, 4)
        cv2.putText(img, text, ((w - tw) // 2, int(h * y)), cv2.FONT_HERSHEY_DUPLEX, s, BLACK, 4, cv2.LINE_AA)
    return img


def wall(px, rng) -> np.ndarray:
    """Corrugated cladding over a painted dado, the hall name stencilled on it."""
    w, h = px
    x = np.arange(w)
    ribs = 0.8 + 0.2 * np.cos(2 * np.pi * x / 24) ** 2
    v = np.tile(ribs, (h, 1)).astype(np.float32) * (1 + 0.04 * _noise(rng, (h, w), 40))
    img = v[..., None] * np.array([92, 84, 76], np.float32)
    img[int(h * 0.72):] = np.array([60, 66, 70], np.float32) * (1 + 0.03 * _noise(rng, (h - int(h * 0.72), w), 20))[
        ..., None]
    img[int(h * 0.7):int(h * 0.72)] = YELLOW
    img = img.clip(0, 255).astype(np.uint8)
    cv2.putText(img, "HALL 2  -  INTRALOGISTICS", (40, int(h * 0.13)), cv2.FONT_HERSHEY_DUPLEX, 1.3, (205, 210, 214),
                3, cv2.LINE_AA)
    return img

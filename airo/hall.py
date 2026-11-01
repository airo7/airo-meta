"""The factory hall around the fleet's floor, at roughly 1:4: pallet racking along the back wall, the inbound
conveyor through the left wall, fenced loader cells with light curtains, charging docks, high-bay lamps, and a work
floor painted the way real ones are. Static parts are merged per material (meshes.Batch), so the whole hall is a
few dozen draw calls; the floor's paint, wear and contact shadows are baked into one texture."""

import math

import cv2
import numpy as np

from airo import mezzanine
from airo.meshes import Batch
from airo.scene import (BLACK, GEOM, MARGIN, PX, STATIC, WHITE, YELLOW, Canvas, _noise, concrete, quat, texture)
from airo.warehouse import BELT_Z, TABLE_Z, Cell, belt, cells, slots

WALL_H = 1.6
FENCE_H = 0.5
RACK_DEPTH = 0.28
# Building columns stand in the rack line, at these uprights; the forklift keeps a pallet at the ground level of
# the rack bay at FORKLIFT_SLOT (x) and stages it at STAGE on the right of the hall.
COLUMNS = (-0.13, 1.31, 3.47)
FORKLIFT_SLOT = 3.11
STAGE = (3.85, 0.25)
CAGES = ((-0.22, 1.98, 0.0), (2.78, -0.42, 0.0))
GOODS = ((3.32, -0.4, 0.05), (3.45, 1.75, -0.04), (-0.62, 0.35, 1.6))
# People cross onto the floor between the last charging dock and the dock cell: over the dock bay's run-in lane,
# clear of anywhere a robot parks.
CROSSING = 1.7


def rack_front(site) -> float:
    """y of the racks' aisle face; their pallets sit RACK_DEPTH / 2 behind it."""
    return site.floor[1] + MARGIN - 0.06 - RACK_DEPTH
GREEN = (60, 150, 70)


def build(spec, site):
    w, h = site.floor
    rng = np.random.default_rng(7)
    wb = spec.worldbody
    texture(spec, "hall", concrete((512, 512), rng, joints=False), specular=0.2, shininess=0.25, texrepeat=[1, 1],
            texuniform=True)
    wb.add_geom(name="floor", type=GEOM.mjGEOM_PLANE, size=[12, 12, 0.05], pos=[w / 2, h / 2, 0], material="hall",
                friction=[0.3, 0.001, 0.00001])
    cs = cells(site)
    texture(spec, "workfloor", workfloor(site, cs, rng), specular=0.3, shininess=0.4)
    sw, sh = w + 2 * MARGIN, h + 2 * MARGIN
    wb.add_geom(name="workfloor", type=GEOM.mjGEOM_PLANE, size=[sw / 2, sh / 2, 1], pos=[w / 2, h / 2, 0.0001],
                material="workfloor", **STATIC)
    _walls(spec, site, rng)
    _racks(spec, site, rng)
    _conveyor(spec, site, cs["pick"])
    _cells(spec, site, cs)
    _docks(spec, site)
    _building(spec, site, rng)
    mezzanine.build(spec)


def _walls(spec, site, rng):
    w, h = site.floor
    wb = spec.worldbody
    texture(spec, "wall", wall((2048, 640), rng), specular=0.15, shininess=0.2)
    # planes are one-sided: the right wall hides nothing from a camera outside the hall (a front wall would still
    # shade the walkway)
    for name, pos, size, euler in [
            ("wall_back", [w / 2, h + MARGIN, WALL_H / 2], [w / 2 + MARGIN, WALL_H / 2, 1], [90, 0, 0]),
            ("wall_left", [-MARGIN, h / 2, WALL_H / 2], [h / 2 + MARGIN, WALL_H / 2, 1], [90, 90, 0]),
            ("wall_right", [w + 1.3, h / 2, WALL_H / 2], [h / 2 + MARGIN, WALL_H / 2, 1], [90, -90, 0])]:
        wb.add_geom(name=name, type=GEOM.mjGEOM_PLANE, size=size, pos=pos, euler=euler, material="wall", **STATIC)
    b = Batch("door")
    y = 1.9
    b.box("door_frame", (0.005, 0.13, 0.27), (-MARGIN + 0.004, y, 0.27))
    b.box("door_leaf", (0.006, 0.11, 0.25), (-MARGIN + 0.006, y, 0.25))
    b.box("exit_green", (0.006, 0.07, 0.025), (-MARGIN + 0.008, y, 0.58))
    b.box("steel", (0.012, 0.004, 0.004), (-MARGIN + 0.016, y - 0.08, 0.26))
    b.emit(spec)


def _racks(spec, site, rng, bay=0.72, depth=RACK_DEPTH, levels=(0.0, 0.42, 0.84), top=1.3):
    """Selective pallet racking: braced upright frames, beam pairs, pallets of stretch-wrapped cartons and crates,
    column guards at the aisle; no pallets beside the columns, nor in the forklift's slot."""
    w, h = site.floor
    y0 = rack_front(site)
    b = Batch("rack")
    xs = np.arange(-0.85, w + 0.9, bay)
    for x in xs:
        for y in (y0, y0 + depth):
            b.box("upright", (0.009, 0.007, top / 2), (x, y, top / 2))
        for z in np.arange(0.12, top, 0.24):
            b.box("upright", (0.003, depth / 2, 0.003), (x, y0 + depth / 2, z))
            b.box("upright", (0.003, 0.003, 0.17), (x, y0 + depth / 2, z + 0.12), euler=(math.degrees(0.86), 0, 0))
        b.box("upright", (0.018, 0.016, 0.003), (x, y0, 0.003))
        b.box("guard", (0.02, 0.012, 0.06), (x, y0 - 0.02, 0.06))
    for x in xs[:-1]:
        for z in levels[1:]:
            for y in (y0 + 0.004, y0 + depth - 0.004):
                b.box("beam", (bay / 2 - 0.01, 0.005, 0.016), (x + bay / 2, y, z))
            for k in range(3):
                b.box("label_card", (0.02, 0.001, 0.008), (x + bay / 2 + (k - 1) * 0.225, y0 - 0.0015, z))
        for z in levels:
            for k in range(3):
                px = x + bay / 2 + (k - 1) * 0.225
                if (rng.random() < 0.12 or min(abs(px - c) for c in COLUMNS) < 0.17
                        or (z == 0 and abs(px - FORKLIFT_SLOT) < 0.01)):
                    continue
                _pallet(b, rng, (px, y0 + depth / 2, z + (0.017 if z else 0)), lo=z == 0)
    b.emit(spec)


def _pallet(b: Batch, rng, pos, lo: bool = False, load: bool = True, yaw: float = 0.0):
    """A 1200 x 800 pallet at 1:4 with a stretch-wrapped load: layers of cartons in a 2 x 3 pattern, or crates. At
    yaw 0 its long side runs along y."""
    x, y, z = pos
    c, s = math.cos(yaw), math.sin(yaw)

    def put(mat, half, dx, dy, dz):
        b.box(mat, half, (x + c * dx - s * dy, y + s * dx + c * dy, z + dz), yaw=yaw)
    for dx in (-0.085, 0, 0.085):
        put("wood", (0.012, 0.15, 0.009), dx, 0, 0.009)
    for dy in np.linspace(-0.135, 0.135, 5):
        put("wood_light", (0.1, 0.014, 0.0025), 0, dy, 0.0205)
    if not load:
        return
    top = 0.023
    if rng.random() < 0.22:
        for k in range(int(rng.integers(2, 5 if not lo else 4))):
            put("crate_blue" if rng.random() < 0.6 else "crate_grey", (0.097, 0.145, 0.021), 0, 0,
                top + 0.045 * k + 0.021)
        return
    layers = int(rng.integers(2, 6 if not lo else 4))
    for k in range(layers):
        for i in (-0.5, 0.5):
            for j in (-1, 0, 1):
                if k == layers - 1 and rng.random() < 0.15:
                    continue
                at = (i * 0.098, j * 0.0975, top + 0.03 * (2 * k + 1))
                put(rng.choice(["carton", "carton", "carton_dark"]), (0.0475, 0.0465, 0.0292), *at)
                if k == layers - 1:
                    put("tape", (0.048, 0.008, 0.0294), *at)
    put("wrap", (0.1, 0.148, 0.03 * layers * 0.85), 0, 0, top + 0.03 * layers * 0.85)


def _conveyor(spec, site, c: Cell):
    """Powered roller conveyor in from the wall: side frames, rollers, legs, guide rails, end stop, photo eye."""
    end, entry = belt(c, site.floor[0], MARGIN)
    stop = end + 0.085 * (end - entry) / np.linalg.norm(end - entry)
    start = np.array([-MARGIN, end[1]])
    u = (stop - start) / np.linalg.norm(stop - start)
    n = np.array([-u[1], u[0]])
    length = np.linalg.norm(stop - start)
    yaw = math.atan2(u[1], u[0])
    mid = (start + stop) / 2
    b = Batch("conveyor")
    for s in (1, -1):
        b.box("conveyor_frame", (length / 2, 0.005, 0.016), (*(mid + s * 0.072 * n), BELT_Z - 0.01), yaw=yaw)
        b.box("steel", (length / 2, 0.003, 0.003), (*(mid + s * 0.066 * n), BELT_Z + 0.03), yaw=yaw)
        for t in np.arange(0.1, length, 0.32):
            p = start + t * u + s * 0.072 * n
            b.box("conveyor_frame", (0.008, 0.008, BELT_Z / 2 - 0.01), (*p, BELT_Z / 2 - 0.01))
            b.box("rubber", (0.014, 0.014, 0.003), (*p, 0.003))
            b.box("steel", (0.003, 0.003, 0.015), (*(start + t * u + s * 0.066 * n), BELT_Z + 0.015))
    for t in np.arange(0.02, length - 0.01, 0.034):
        b.cylinder("roller", 0.0115, 0.064, (*(start + t * u), BELT_Z - 0.0115), axis=(*n, 0), sides=10)
    b.box("hazard_y", (0.006, 0.08, 0.016), (*stop, BELT_Z + 0.012), yaw=yaw)
    b.box("arm_dark", (0.08, 0.04, 0.03), (*(start + 0.5 * u - 0.11 * n), 0.06), yaw=yaw)
    b.box("arm_dark", (0.004, 0.004, 0.035), (*(end + 0.13 * n), BELT_Z + 0.035))
    b.box("led_red", (0.004, 0.006, 0.004), (*(end + 0.13 * n - 0.006 * n), BELT_Z + 0.055))
    b.emit(spec)
    portal = Batch("portal")
    portal.box("portal_dark", (0.004, 0.13, 0.11), (-MARGIN + 0.003, end[1], 0.15))
    portal.box("hazard_y", (0.005, 0.14, 0.008), (-MARGIN + 0.004, end[1], 0.265))
    for k in range(7):
        portal.box("strip_curtain", (0.0015, 0.016, 0.1), (-MARGIN + 0.012, end[1] - 0.105 + 0.035 * k, 0.15))
    portal.emit(spec)


def _cells(spec, site, cs: dict[str, Cell]):
    """Around every loader: polycarbonate guarding on three sides, light-curtain posts at the open front, a stack light, the
    station's table or outbound pallet, and a sign."""
    f, kit = Batch("fence"), Batch("cell")
    for name, c in cs.items():
        gap = 0.1 if name == "pick" else 0.0
        _fence(f, c, gap)
        for s in (-c.side, c.side):
            p = c.at(c.front, s)
            kit.box("curtain_post", (0.007, 0.007, 0.2), (*p, 0.2))
            kit.box("fence_post", (0.0075, 0.0075, 0.01), (*p, 0.41))
            kit.box("curtain_lens", (0.0072, 0.0072, 0.17), (*p, 0.21))
        kit.cylinder("arm_dark", 0.012, 0.006, (*c.at(c.rear, c.side), FENCE_H + 0.012), sides=12)
        if name != "pick":
            _table(kit, c, name)
    f.emit(spec)
    kit.emit(spec)
    for name, c in cs.items():
        _sign(spec, c, name)


def _fence(b: Batch, c: Cell, gap: float):
    side = c.side
    rear = [(-side, side)] if not gap else [(-side, -gap), (gap, side)]
    runs = [(c.at(c.rear, s0), c.at(c.rear, s1)) for s0, s1 in rear]
    runs += [(c.at(c.front, s), c.at(c.rear, s)) for s in (-side, side)]
    for a, z in runs:
        length = np.linalg.norm(z - a)
        u = (z - a) / length
        yaw = math.atan2(u[1], u[0])
        k = max(1, round(length / 0.3))
        for i in range(k + 1):
            p = a + u * length * i / k
            b.box("fence_post", (0.009, 0.009, FENCE_H / 2), (*p, FENCE_H / 2))
            b.box("fence_post", (0.016, 0.016, 0.002), (*p, 0.002))
        for i in range(k):
            m = a + u * length * (i + 0.5) / k
            half = length / k / 2 - 0.012
            for z_ in (0.04, FENCE_H - 0.015):
                b.box("fence_frame", (half, 0.004, 0.006), (*m, z_), yaw=yaw)
            for s_ in (-1, 1):
                b.box("fence_frame", (0.004, 0.004, (FENCE_H - 0.055) / 2), (*(m + s_ * half * u), FENCE_H / 2 + 0.006))
            b.box("fence_panel", (half, 0.0008, (FENCE_H - 0.055) / 2), (*m, 0.04 + (FENCE_H - 0.055) / 2), yaw=yaw)


def _table(b: Batch, c: Cell, name: str):
    """Where the station's arm sets totes down: a roller staging table, a packing bench, or an outbound pallet."""
    ss = slots(c)
    centre = np.mean([s.xyz[:2] for s in ss], axis=0)
    yaw = math.atan2(c.along[1], c.along[0])
    along, out = np.array(c.along), np.array(c.out)
    if name == "dock":
        rng = np.random.default_rng(3)
        _pallet(b, rng, (*centre, 0.0), load=False, yaw=yaw + math.pi / 2)
        b.cylinder("wrap", 0.012, 0.055, (*(centre + 0.21 * along + 0.06 * out), 0.055), sides=10)
        return
    ha, ho = 0.23, 0.085
    top = "bench_top" if name == "packing" else "conveyor_frame"
    b.box(top, (ha, ho, 0.006), (*centre, TABLE_Z - 0.006), yaw=yaw)
    if name != "packing":
        for t in np.linspace(-ha + 0.02, ha - 0.02, 13):
            b.cylinder("roller", 0.007, ho - 0.007, (*(centre + t * along), TABLE_Z - 0.006), axis=(*out, 0), sides=8)
    for sa in (-ha + 0.02, ha - 0.02):
        for so in (-ho + 0.012, ho - 0.012):
            b.box("alu", (0.008, 0.008, (TABLE_Z - 0.012) / 2), (*(centre + sa * along + so * out),
                                                                  (TABLE_Z - 0.012) / 2))
    b.box("alu", (ha - 0.02, 0.006, 0.006), (*(centre + (ho - 0.012) * out), 0.03), yaw=yaw)
    if name == "packing":
        b.box("cardboard", (0.05, 0.04, 0.002), (*(centre - 0.27 * along), 0.002), yaw=yaw + 0.3)


def _sign(spec, c: Cell, name: str):
    """A hanging station sign above the cell's rear fence, readable from the floor."""
    from airo.site import PAD_BGR

    title = "PICK CELL · INBOUND" if name == "pick" else name.upper() if len(name) <= 2 else name.upper()
    colour = (40, 150, 240) if name == "pick" else PAD_BGR[name]
    texture(spec, f"sign_{name}", sign(title, colour), specular=0.2, shininess=0.3)
    p = c.at(c.rear + 0.01)
    face = math.atan2(-c.out[1], -c.out[0])
    b = spec.worldbody.add_body(name=f"sign_{name}", pos=[*p, FENCE_H + 0.13], quat=quat(face))
    b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.006, 0.17, 0.05], material="arm_dark", **STATIC)
    b.add_geom(type=GEOM.mjGEOM_PLANE, size=[0.16, 0.042, 1], pos=[0.0065, 0, 0], quat=_facing_x(),
               material=f"sign_{name}", **STATIC)
    for s in (-0.12, 0.12):
        b.add_geom(type=GEOM.mjGEOM_BOX, size=[0.003, 0.003, 0.07], pos=[0, s, -0.11], material="steel", **STATIC)


def _facing_x() -> list[float]:
    """A plane facing +x, texture upright."""
    import mujoco

    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.column_stack([(0, 1, 0), (0, 0, 1), (1, 0, 0)]).ravel().astype(float))
    return q.tolist()


def _docks(spec, site):
    """AMR charging docks behind each robot's parking bay: a charger unit, its contacts, a status light."""
    b = Batch("charger")
    for x, y, th in site.starts:
        back = np.array([x - 0.2 * math.cos(th), y - 0.2 * math.sin(th)])
        b.box("charger", (0.05, 0.02, 0.032), (*back, 0.032), yaw=th + math.pi / 2)
        b.box("arm_dark", (0.051, 0.021, 0.003), (*back, 0.066), yaw=th + math.pi / 2)
        b.box("arm_dark", (0.052, 0.024, 0.004), (*back, 0.004), yaw=th + math.pi / 2)
        b.box("led_green", (0.025, 0.0015, 0.002), (*(back + 0.0205 * np.array([math.cos(th), math.sin(th)])), 0.055),
              yaw=th + math.pi / 2)
        for s in (-0.02, 0.02):
            p = back + 0.035 * np.array([math.cos(th), math.sin(th)]) + s * np.array([-math.sin(th), math.cos(th)])
            b.box("brass_pad", (0.012, 0.006, 0.0008), (*p, 0.0008), yaw=th)
    b.emit(spec)


def _building(spec, site, rng):
    """Columns with hazard wraps, high-bay lamps under the roof, pallets of goods along the edges, roll cages."""
    w, h = site.floor
    b = Batch("pillar")
    y = rack_front(site) + RACK_DEPTH / 2
    for x in COLUMNS:
        b.box("concrete_col", (0.07, 0.07, 1.1), (x, y, 1.1))
        for k in range(6):
            b.box("hazard_k" if k % 2 else "hazard_y", (0.074, 0.074, 0.0125), (x, y, 0.0125 + 0.025 * k))
    b.emit(spec)
    lamps = Batch("lamp")
    for x in (0.2, 1.5, 2.8):
        for y in (0.3, 1.7):
            lamps.cylinder("lamp_body", 0.08, 0.02, (x, y, 1.95), sides=16)
            lamps.cylinder("lamp_glow", 0.07, 0.002, (x, y, 1.928), sides=16)
            lamps.box("steel", (0.002, 0.002, 0.15), (x, y, 2.12))
    lamps.emit(spec)
    goods = Batch("goods")
    for x, y, th in GOODS:
        _pallet(goods, rng, (x, y, 0.0), yaw=th)
    goods.emit(spec)
    cage = Batch("cage")
    for x, y, th in CAGES:
        _cage(cage, rng, (x, y), th)
    cage.emit(spec)


def _cage(b: Batch, rng, xy, yaw: float):
    """A roll cage at 1:4: mesh sides on a castored base, part filled with cartons."""
    x, y = xy
    c, s = math.cos(yaw), math.sin(yaw)
    put = lambda mat, half, dx, dy, dz: b.box(mat, half, (x + c * dx - s * dy, y + s * dx + c * dy, dz), yaw=yaw)
    hx, hy, hz = 0.1, 0.085, 0.21
    put("cage", (hx, hy, 0.005), 0, 0, 0.02)
    for sx in (-1, 1):
        for sy in (-1, 1):
            put("rubber", (0.008, 0.006, 0.008), sx * (hx - 0.012), sy * (hy - 0.012), 0.008)
            put("cage", (0.004, 0.004, hz / 2), sx * hx, sy * hy, 0.025 + hz / 2)
    for z in np.linspace(0.06, 0.025 + hz, 6):
        for sy in (-1, 1):
            put("cage", (hx, 0.0015, 0.0015), 0, sy * hy, z)
        put("cage", (0.0015, hy, 0.0015), -hx, 0, z)
    for i in np.linspace(-hx, hx, 7):
        for sy in (-1, 1):
            put("cage", (0.0012, 0.0012, hz / 2), i, sy * hy, 0.025 + hz / 2)
    for k in range(int(rng.integers(3, 6))):
        put(rng.choice(["carton", "carton_dark"]), (0.045, 0.038, 0.028), (k % 2 - 0.5) * 0.095 * 0.98,
            ((k // 2) % 2 - 0.5) * 0.08, 0.025 + 0.028 + 0.056 * (k // 4))


# textures


def workfloor(site, cs: dict[str, Cell], rng) -> np.ndarray:
    """The painted work floor: control joints, contact shadows, the hazard-striped AMR zone, the pedestrian
    walkway, hatched robot cells, charging bays, the loading bay with its queue boxes, flow arrows, tyre marks."""
    w, h = site.floor
    shape = (round((h + 2 * MARGIN) * PX), round((w + 2 * MARGIN) * PX))
    c = Canvas(concrete(shape, rng), rng, origin=(-MARGIN, h + MARGIN))
    p, lw = c.p, c.w
    yy, xx = np.mgrid[:shape[0], :shape[1]]
    stripes = (((xx + yy) // lw(0.025)) % 2).astype(np.uint8)
    hatch = (((xx - yy) // lw(0.03)) % 3 == 0).astype(np.uint8)

    def band(m, keep):
        cv2.rectangle(m, p(-0.06, h + 0.06), p(w + 0.06, -0.06), 1, -1)
        cv2.rectangle(m, p(0, h), p(w, 0), 0, -1)
        m &= stripes == keep
    c.paint(YELLOW, lambda m: band(m, 0))
    c.paint(BLACK, lambda m: band(m, 1))
    walk = (-0.62, -0.9)
    c.paint(GREEN, lambda m: cv2.rectangle(m, p(-MARGIN, walk[0]), p(w + MARGIN, walk[1]), 1, -1))
    c.paint(WHITE, lambda m: [cv2.line(m, p(-MARGIN, y), p(w + MARGIN, y), 1, lw(0.016)) for y in walk])
    for x in np.arange(-0.6, w + 0.8, 0.9):
        c.text("WALKWAY", (x, (walk[0] + walk[1]) / 2), 0.06, WHITE)

    for cl in cs.values():
        out, along, base = np.array(cl.out), np.array(cl.along), np.array(cl.base)
        quad = lambda f, r, s: np.array([p(*(base + f * out - s * along)), p(*(base + f * out + s * along)),
                                          p(*(base + r * out + s * along)), p(*(base + r * out - s * along))],
                                         np.int32)
        f, r, s = cl.front, cl.rear, cl.side
        c.paint((84, 88, 92), lambda m: cv2.fillPoly(m, [quad(f, r, s)], 1))

        def ring(m):
            cv2.fillPoly(m, [quad(f, r, s)], 1)
            cv2.fillPoly(m, [quad(f + 0.05, r - 0.05, s - 0.05)], 0)
            m &= hatch
        c.paint(YELLOW, ring)
        c.paint(YELLOW, lambda m: cv2.polylines(m, [quad(f, r, s)], True, 1, lw(0.012)))

    def box(m, x, y, th, hx, hy, dash=False):
        cs_, sn = math.cos(th), math.sin(th)
        pts = [np.array([x + a * hx * cs_ - b * hy * sn, y + a * hx * sn + b * hy * cs_])
               for a, b in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        for a, z in zip(pts, pts[1:] + pts[:1]):
            if not dash:
                cv2.line(m, p(*a), p(*z), 1, lw(0.018))
                continue
            for t in np.arange(0, 1, 0.2):
                cv2.line(m, p(*(a + (z - a) * t)), p(*(a + (z - a) * (t + 0.11))), 1, lw(0.016))

    pick = site.pick
    c.paint(YELLOW, lambda m: box(m, pick.x, pick.y, pick.theta, 0.18, 0.13))
    c.text("LOAD", (pick.x + 0.23, pick.y), 0.06, YELLOW, angle=90)
    for k, (x, y, th) in enumerate(pick.wait):
        c.paint(YELLOW, lambda m, x=x, y=y, th=th: box(m, x, y, th, 0.16, 0.11, dash=True))
        c.text(f"Q{k + 1}", (x + 0.21, y), 0.05, YELLOW, angle=90)
    for bay in site.drops.values():
        for x, y, th in bay.wait:
            c.paint(WHITE, lambda m, x=x, y=y, th=th: box(m, x, y, th, 0.16, 0.11, dash=True))
    for k, (x, y, th) in enumerate(site.starts):
        c.paint(WHITE, lambda m, x=x, y=y, th=th: box(m, x, y, th, 0.17, 0.12))
        c.text(f"C{k + 1}", (x, y + 0.22), 0.055, WHITE)
    for a, z in [((0.75, 1.32), (1.0, 1.42)), ((1.5, 0.62), (1.75, 0.5)), ((2.3, 0.62), (2.45, 0.78)),
                 ((1.55, 1.85 - 0.25), (1.75, 1.85 - 0.25))]:
        c.arrow(a, z)
    door = (-MARGIN, 1.9)
    c.paint(YELLOW, lambda m: cv2.rectangle(m, p(door[0], door[1] + 0.16), p(door[0] + 0.32, door[1] - 0.16), 1, -1))
    c.paint(BLACK, lambda m: (cv2.rectangle(m, p(door[0], door[1] + 0.16), p(door[0] + 0.32, door[1] - 0.16), 1, -1),
                              m.__iand__(stripes)))
    c.paint(WHITE, lambda m: [cv2.rectangle(m, p(CROSSING - 0.1, y), p(CROSSING + 0.1, y - 0.045), 1, -1)
                              for y in np.arange(-0.1, walk[0], -0.09)])
    sx, sy = STAGE
    c.paint(YELLOW, lambda m: box(m, sx, sy, math.pi / 2, 0.19, 0.14))
    c.text("PALLETS", (sx, sy - 0.27), 0.05, YELLOW)
    c.paint(YELLOW, lambda m: [cv2.line(m, p(sx + s_, sy + 0.45), p(sx + s_, rack_front(site) - 0.25), 1, lw(0.012))
                               for s_ in (-0.19, 0.19)])
    c.text("FORKLIFTS", (sx, 1.3), 0.06, YELLOW, angle=90)
    _tyres(c, [(pick.x, pick.y, b.x, b.y) for b in site.drops.values()] +
           [(x, y, pick.x, pick.y) for x, y, _ in site.starts])
    _contact(c, site, cs)
    return c.done()


def _tyres(c: Canvas, lines):
    """Faint rubber marks where the robots run most."""
    m = np.zeros(c.img.shape[:2], np.float32)
    for x0, y0, x1, y1 in lines:
        d = np.array([x1 - x0, y1 - y0])
        n = np.array([-d[1], d[0]]) / (np.linalg.norm(d) + 1e-9)
        for off in (-0.07, 0.07):
            pts = np.array([c.p(*(np.array([x0, y0]) + t * d + (off + 0.01 * math.sin(9 * t)) * n))
                            for t in np.linspace(0, 1, 40)], np.int32)
            cv2.polylines(m, [pts], False, 1.0, max(2, c.w(0.02)), cv2.LINE_AA)
    c.img *= 1 - 0.06 * cv2.GaussianBlur(m, (0, 0), 3)[..., None]


def _contact(c: Canvas, site, cs):
    """Baked contact shadows: soft darkening where racks, fences, the conveyor and walls meet the floor."""
    w, h = site.floor
    m = np.zeros(c.img.shape[:2], np.float32)
    rect = lambda x0, y0, x1, y1, v=1.0: cv2.rectangle(m, c.p(x0, y1), c.p(x1, y0), v, -1)
    rect(-1.0, h + MARGIN - 0.36, w + MARGIN, h + MARGIN)
    rect(-MARGIN, -MARGIN, -MARGIN + 0.05, h + MARGIN, 0.8)
    end = cs["pick"].at(0.27)
    rect(-MARGIN, end[1] - 0.08, end[0] + 0.08, end[1] + 0.08, 0.7)
    for cl in cs.values():
        base = np.array(cl.base)
        pts = [cl.at(f, s) for f, s in ((cl.front, -cl.side), (cl.front, cl.side), (cl.rear, cl.side),
                                        (cl.rear, -cl.side))]
        cv2.polylines(m, [np.array([c.p(*q) for q in pts], np.int32)], True, 0.8, c.w(0.03))
        cv2.circle(m, c.p(*base), c.w(0.07), 0.9, -1)
    for x, y, _ in site.starts:
        cv2.circle(m, c.p(x, y - 0.2), c.w(0.05), 0.8, -1)
    c.img *= 1 - 0.3 * cv2.GaussianBlur(m, (0, 0), c.w(0.03))[..., None]


def sign(text: str, bgr, px=(640, 168)) -> np.ndarray:
    w, h = px
    img = np.full((h, w, 3), (38, 36, 34), np.uint8)
    img[:, : h // 3] = bgr
    font = cv2.FONT_HERSHEY_DUPLEX
    scale = min(0.75 * (w - h // 3 - 40) / cv2.getTextSize(text, font, 1, 3)[0][0], 2.4)
    (tw, th), _ = cv2.getTextSize(text, font, scale, 3)
    cv2.putText(img, text, (h // 3 + 24, (h + th) // 2), font, scale, WHITE, 3, cv2.LINE_AA)
    return img


def wall(px, rng) -> np.ndarray:
    """Insulated cladding: vertical panels with seams, a painted dado with a yellow line, grime toward the floor,
    a clerestory band, and the hall name."""
    w, h = px
    x = np.arange(w)
    ribs = 0.86 + 0.14 * np.cos(2 * np.pi * x / 22) ** 2
    v = np.tile(ribs, (h, 1)).astype(np.float32) * (1 + 0.03 * _noise(rng, (h, w), 60))
    v[:, (x % 180) < 2] *= 0.7
    img = v[..., None] * np.array([150, 146, 140], np.float32)
    clere = int(h * 0.12)
    img[:clere] = np.array([70, 64, 58], np.float32) + 25 * _noise(rng, (clere, w), 30)[..., None]
    img[clere:clere + 4] = (60, 60, 60)
    dado = int(h * 0.8)
    img[dado:] = np.array([72, 76, 80], np.float32) * (1 + 0.04 * _noise(rng, (h - dado, w), 20))[..., None]
    img[dado - 6:dado] = YELLOW
    grime = np.linspace(0, 1, h)[:, None] ** 4 * (0.25 + 0.1 * _noise(rng, (h, w), 40))
    img *= (1 - grime)[..., None]
    img = img.clip(0, 255).astype(np.uint8)
    cv2.putText(img, "HALL 2  -  INTRALOGISTICS", (60, int(h * 0.3)), cv2.FONT_HERSHEY_DUPLEX, 2.2, (52, 54, 58), 5,
                cv2.LINE_AA)
    return img

"""Procedural meshes: an open-topped tote shell, and Batch, which merges static decor into one mesh per material so a
hall full of parts costs a handful of draw calls."""

import math
from collections import defaultdict

import mujoco
import numpy as np

GEOM = mujoco.mjtGeom
SHELL = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
STATIC = {"contype": 0, "conaffinity": 0}


def _quads(quads) -> tuple[np.ndarray, np.ndarray]:
    """quads: (three or four corners, outward normal). Triangles wound so their normals point outward."""
    verts, faces = [], []
    for corners, n in quads:
        p = [np.asarray(c, float) for c in corners]
        tri = [(0, 1, 2), (0, 2, 3)][:len(p) - 2]
        if np.cross(p[1] - p[0], p[2] - p[0]) @ n < 0:
            tri = [(i, k, j) for i, j, k in tri]
        base = len(verts)
        verts += p
        faces += [(base + i, base + j, base + k) for i, j, k in tri]
    return np.array(verts), np.array(faces)


def _ring(hx, hy, z):
    return [(hx, hy, z), (-hx, hy, z), (-hx, -hy, z), (hx, -hy, z)]


def shell(spec, name: str, half, wall: float, taper: float) -> str:
    """An open-topped box, outer half size half at the rim, its sides drawn in by taper at the bottom."""
    hx, hy, hz = half
    ot, ob = _ring(hx, hy, hz), _ring(hx - taper, hy - taper, -hz)
    it, ib = _ring(hx - wall, hy - wall, hz), _ring(hx - taper - wall, hy - taper - wall, -hz + wall)
    sides = [(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)]
    order = [(3, 0), (0, 1), (1, 2), (2, 3)]
    quads = [([ob[i], ob[j], ot[j], ot[i]], np.array(n)) for (i, j), n in zip(order, sides)]
    quads += [([ib[i], ib[j], it[j], it[i]], -np.array(n)) for (i, j), n in zip(order, sides)]
    quads += [([ot[i], ot[j], it[j], it[i]], np.array((0, 0, 1))) for i, j in order]
    quads += [(ob, np.array((0, 0, -1))), (ib, np.array((0, 0, 1)))]
    v, f = _quads(quads)
    m = spec.add_mesh(name=name, uservert=v.ravel().tolist(), userface=f.ravel().tolist())
    m.inertia = SHELL
    return name


def _rot(euler) -> np.ndarray:
    """Intrinsic x-y-z Euler angles in degrees, MuJoCo's default."""
    r = np.eye(3)
    for axis, deg in zip(range(3), euler):
        c, s = math.cos(math.radians(deg)), math.sin(math.radians(deg))
        k = [[1, 0, 0, 0, c, -s, 0, s, c], [c, 0, s, 0, 1, 0, -s, 0, c], [c, -s, 0, s, c, 0, 0, 0, 1]][axis]
        r = r @ np.array(k).reshape(3, 3)
    return r


class Batch:
    """Static boxes and prisms gathered per material, emitted as one mesh geom each. Edges stay sharp: MuJoCo leaves
    faces at large angles out of a vertex's normal."""

    def __init__(self, name: str):
        self.name = name
        self.parts: dict[str, list] = defaultdict(list)

    def box(self, material: str, half, pos, euler=(0, 0, 0), yaw: float | None = None):
        hx, hy, hz = half
        corners = np.array([[sx * hx, sy * hy, sz * hz] for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
        r = _rot((0, 0, math.degrees(yaw)) if yaw is not None else euler)
        p = corners @ r.T + pos
        faces = [((0, 2, 3, 1), (0, 0, -1)), ((4, 5, 7, 6), (0, 0, 1)), ((0, 1, 5, 4), (0, -1, 0)),
                 ((2, 6, 7, 3), (0, 1, 0)), ((0, 4, 6, 2), (-1, 0, 0)), ((1, 3, 7, 5), (1, 0, 0))]
        self.parts[material].append([([p[i] for i in idx], r @ n) for idx, n in faces])

    def cylinder(self, material: str, radius: float, half: float, pos, axis=(0, 0, 1), sides: int = 12):
        axis = np.asarray(axis, float) / np.linalg.norm(axis)
        u = np.cross(axis, (1, 0, 0) if abs(axis[0]) < 0.9 else (0, 1, 0))
        u /= np.linalg.norm(u)
        v = np.cross(axis, u)
        a = np.linspace(0, math.tau, sides, endpoint=False)
        ring = radius * (np.outer(np.cos(a), u) + np.outer(np.sin(a), v))
        lo, hi = ring + pos - half * axis, ring + pos + half * axis
        quads = [([lo[i], lo[(i + 1) % sides], hi[(i + 1) % sides], hi[i]], ring[i] + ring[(i + 1) % sides])
                 for i in range(sides)]
        c0, c1 = np.asarray(pos) - half * axis, np.asarray(pos) + half * axis
        quads += [([c0, lo[(i + 1) % sides], lo[i]], -axis) for i in range(sides)]
        quads += [([c1, hi[i], hi[(i + 1) % sides]], axis) for i in range(sides)]
        self.parts[material].append(quads)

    def emit(self, spec, body=None, **geom):
        body = body or spec.worldbody
        for material, groups in self.parts.items():
            v, f = _quads([q for g in groups for q in g])
            name = f"{self.name}_{material}"
            m = spec.add_mesh(name=name, uservert=v.ravel().tolist(), userface=f.ravel().tolist())
            m.inertia = SHELL
            body.add_geom(name=name, type=GEOM.mjGEOM_MESH, meshname=name, material=material, **{**STATIC, **geom})
        self.parts.clear()

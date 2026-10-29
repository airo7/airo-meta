"""Robots from DeepMind's MuJoCo Menagerie (Apache-2.0), fetched into the git-ignored third_party/menagerie by
scripts/fetch_models.py, and grafted into a site at the hall's 1:4 scale. Without them the site falls back to our
own procedural arm and leaves the humanoids out.

A model is compiled once on its own; its visual geoms are copied into the site's spec with every length scaled,
masses by s³ and inertias by s⁵. Meshes are decimated by vertex clustering (at 1:4 the full models carry far more
triangles than pixels) and shared between copies, so four arms cost one set of mesh buffers."""

import math
from functools import cache
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parent.parent / "third_party" / "menagerie"
SCALE = 0.25
UR5E, GRIPPER, G1 = "universal_robots_ur5e/ur5e.xml", "robotiq_2f85/2f85.xml", "unitree_g1/g1.xml"
GEOM = mujoco.mjtGeom
COLLISION = 3
HIDDEN = 5
# Parts smaller than this (metres, at scale) go in geom group 2, which the far views leave out.
FINE = 0.009
# Level of detail: vertex-cluster cells for a model seen from within NEAR of the camera, and from farther away.
CELL, FAR_CELL = 0.0012, 0.004
NEAR = 1.2


class Lod:
    """The geoms of one grafted model at two levels of detail, shown by the camera's distance to its root."""

    def __init__(self, root: str):
        self.root = root
        self.near: list[str] = []
        self.far: list[str] = []

    def bind(self, m: mujoco.MjModel):
        self._root = m.body(self.root).id
        self._near = np.array([m.geom(n).id for n in self.near], int)
        self._far = np.array([m.geom(n).id for n in self.far], int)
        self._groups = m.geom_group[self._near].copy()
        m.geom_group[self._near] = HIDDEN
        m.geom_group[self._far] = 1
        self._shown = False

    def show(self, m: mujoco.MjModel, d: mujoco.MjData, eye):
        near = eye is not None and float(np.linalg.norm(d.xpos[self._root] - eye)) < NEAR
        if near != self._shown:
            m.geom_group[self._near] = self._groups if near else HIDDEN
            m.geom_group[self._far] = HIDDEN if near else 1
            self._shown = near


def available(*models: str) -> bool:
    return all((ROOT / m).is_file() for m in models)


@cache
def model(path: str) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(path)


def _src(rel: str) -> mujoco.MjModel:
    return model(str(ROOT / rel))


def decimate(v: np.ndarray, f: np.ndarray, cell: float) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering: vertices in the same grid cell merge to their mean, collapsed and repeated faces go."""
    _, inv = np.unique(np.floor(v / cell).astype(np.int64), axis=0, return_inverse=True)
    inv = inv.ravel()
    verts = np.zeros((inv.max() + 1, 3))
    np.add.at(verts, inv, v)
    verts /= np.bincount(inv)[:, None]
    faces = inv[f]
    faces = faces[(faces[:, 0] != faces[:, 1]) & (faces[:, 1] != faces[:, 2]) & (faces[:, 0] != faces[:, 2])]
    _, keep = np.unique(np.sort(faces, axis=1), axis=0, return_index=True)
    return verts, faces[np.sort(keep)]


@cache
def _mesh(path: str, i: int, scale: float, cell: float) -> tuple[np.ndarray, np.ndarray]:
    m = model(path)
    a, n, fa, fn = m.mesh_vertadr[i], m.mesh_vertnum[i], m.mesh_faceadr[i], m.mesh_facenum[i]
    v, f = m.mesh_vert[a:a + n] * scale, m.mesh_face[fa:fa + fn]
    return decimate(v, f, cell) if cell else (v, f)


def subtree(m: mujoco.MjModel, root: int) -> list[int]:
    return [b for b in range(root, m.nbody) if b == root or _under(m, b, root)]


def _under(m, b: int, root: int) -> bool:
    while b > 0:
        if b == root:
            return True
        b = m.body_parentid[b]
    return False


def graft(spec, rel: str, root: str, parent, prefix: str, pos=None, quat=None, joints: bool = True,
          geoms: bool = True, scale: float = SCALE, lod: Lod | None = None) -> dict[str, object]:
    """Copy the body root of a Menagerie model, its subtree and visual geoms under parent, names prefixed; root
    keeps its own pose unless pos or quat is given. With lod, every mesh comes at both levels of detail. Returns the
    new spec bodies by their source names."""
    path = str(ROOT / rel)
    m, tag = model(path), rel.split("/")[0]
    s = scale
    r = m.body(root).id
    bodies = {}
    for b in subtree(m, r):
        name = m.body(b).name
        top = b == r
        up = parent if top else bodies[m.body(m.body_parentid[b]).name]
        p = m.body_pos[b] * s if not top or pos is None else np.asarray(pos, float)
        q = m.body_quat[b] if not top or quat is None else np.asarray(quat, float)
        body = up.add_body(name=prefix + name, pos=p.tolist(), quat=q.tolist(), gravcomp=1 if joints else 0)
        body.mass, body.ipos, body.iquat = max(m.body_mass[b] * s ** 3, 1e-4), (m.body_ipos[b] * s).tolist(), \
            m.body_iquat[b].tolist()
        body.inertia, body.explicitinertial = np.maximum(m.body_inertia[b] * s ** 5, 1e-9).tolist(), True
        bodies[name] = body
        for j in range(m.body_jntadr[b], m.body_jntadr[b] + m.body_jntnum[b]) if joints else ():
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
                continue
            body.add_joint(name=prefix + m.joint(j).name, type=mujoco.mjtJoint.mjJNT_HINGE, axis=m.jnt_axis[j].tolist(),
                           pos=(m.jnt_pos[j] * s).tolist())
        for g in range(m.body_geomadr[b], m.body_geomadr[b] + m.body_geomnum[b]) if geoms else ():
            if m.geom_group[g] == COLLISION or m.geom_contype[g] or m.geom_conaffinity[g]:
                continue
            _geom(spec, body, m, g, path, tag, s, lod)
    return bodies


def _geom(spec, body, m, g: int, path: str, tag: str, s: float, lod: Lod | None):
    kw = {"pos": (m.geom_pos[g] * s).tolist(), "quat": m.geom_quat[g].tolist(), "contype": 0, "conaffinity": 0,
          "density": 0, "material": _material(spec, m, g, tag)}
    if m.geom_type[g] != GEOM.mjGEOM_MESH:
        body.add_geom(type=m.geom_type[g], size=(m.geom_size[g] * s).tolist(), group=1, **kw)
        return
    i = m.geom_dataid[g]
    for far in (False, True) if lod else (False,):
        name = f"{tag}/{m.mesh(i).name}" + ("@far" if far else "")
        v, f = _mesh(path, i, s, FAR_CELL if far else CELL)
        size = float(np.ptp(v, axis=0).max()) if len(f) else 0.0
        if far and size < FINE:
            continue
        if spec.mesh(name) is None:
            mesh = spec.add_mesh(name=name, uservert=v.ravel().tolist(), userface=f.ravel().tolist())
            mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL
        geom = body.add_geom(type=GEOM.mjGEOM_MESH, meshname=name, group=2 if size < FINE else 1, **kw)
        if lod:
            geom.name = f"{body.name}#{g}" + ("@far" if far else "")
            (lod.far if far else lod.near).append(geom.name)


def _material(spec, m, g: int, tag: str) -> str:
    k = m.geom_matid[g]
    if k < 0:
        rgba = m.geom_rgba[g]
        name = f"{tag}/rgba_" + "_".join(f"{c:.2f}" for c in rgba)
        spec_, shin, refl = 0.5, 0.5, 0.0
    else:
        name, rgba = f"{tag}/{m.material(k).name}", m.mat_rgba[k]
        spec_, shin, refl = m.mat_specular[k], m.mat_shininess[k], m.mat_reflectance[k]
    if spec.material(name) is None:
        spec.add_material(name=name, rgba=rgba.tolist(), specular=float(spec_), shininess=float(shin),
                          reflectance=float(refl))
    return name


def yaw_quat(theta: float) -> list[float]:
    return [math.cos(theta / 2), 0, 0, math.sin(theta / 2)]

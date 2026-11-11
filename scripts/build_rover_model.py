"""
Build ``src/mppp/data/rover/perseverance_rmech.npz`` - the simplified Perseverance model of mppp.ibr.rover_model -
from NASA's glTF model (https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/, 25042_Perseverance.glb,
credit NASA/JPL-Caltech; Christian, 10 Nov 2026: "take this model of perseverance's, simplify it, add it to the mesh,
and project the images onto it. The arm is stowed, so you would need to rotate the arm joints").

    python scripts/build_rover_model.py 25042_Perseverance.glb [--out FILE]

* Frame: the model is in metres, x left, y up, z forward, the wheels on y = 0; RMECH (x forward, y right, z down,
  z 0 = the deck top) is x = z_model, y = -x_model, z = 1.135 - y_model, then the small rigid correction of an ICP of
  the Van Zyl S003D2046 rover cloud onto the body (0.12 deg, 2-3 mm; 60 % of the cloud within 3 cm of the body).
* Left out: the mast above its base and the head (they turn with the cameras), the high-gain antenna (it points
  differently every sol; the images give it), the parts under the belly.
* The arm: the chain arm.003 (shoulder azimuth, about its local y) - arm.002 (shoulder elevation, local z) - arm
  (elbow, z) - arm.004 (wrist, z) - turret_obj (turret, y), turned from the model's deployed pose by ARM_STOW_DEG, the
  fit of the five angles to the Van Zyl cloud of the stowed arm (78 % of the arm's cloud within 2 cm of it, 94 %
  within 5 cm; the turret's own angle is the least certain).
* Simplified per part by quadric decimation (TARGET_FACES).
"""
import argparse
import json
import struct
from pathlib import Path

import numpy as np

ARM_CHAIN = ("arm.003", "arm.002", "arm", "arm.004", "turret_obj")
ARM_AXES = ("y", "z", "z", "z", "y")
ARM_STOW_DEG = (89.2, 16.4, 71.9, -103.3, -90.1)
DECK_TOP_MODEL_Y = 1.135
ICP_MODEL_FROM_MECH = np.array([[1.0, -2.0e-4, -2.0e-3, 0.0], [2.0e-4, 1.0, 1.1e-3, 2.1e-3], [2.0e-3, -1.1e-3, 1.0, 2.7e-3],
                                [0.0, 0.0, 0.0, 1.0]])
TARGET_FACES = {"body": 32000, "arm": 10000, "arm_base": 2300, "suspension": 8000, "wheels": 9000}
PARTS = ("body", "arm", "arm_base", "suspension", "wheels")
MAST = ("bottom", "bottom.002", "bottom.003", "probe.001", "top", "probe", "head", "lens", "Mastcam_Z_cams",
        "microphones.001", "NavCams")
MOVING = ("antenna_hg",)                  # the high-gain antenna points differently every sol: left to the images
BELLY = ("arm_01", "arm_02", "arm_01_joint", "arm_02_joint", "arm_01_pivot", "arm_02_pivot", "arm_02.001", "pan_end cover")


def _local(n):
    from scipy.spatial.transform import Rotation as R
    T = np.eye(4)
    if "matrix" in n:
        return np.array(n["matrix"], float).reshape(4, 4).T
    if "rotation" in n:
        T[:3, :3] = R.from_quat(n["rotation"]).as_matrix()
    if "scale" in n:
        T[:3, :3] = T[:3, :3] * np.array(n["scale"], float)
    if "translation" in n:
        T[:3, 3] = n["translation"]
    return T


def build(glb: Path, out: Path) -> dict:
    import open3d as o3d
    import trimesh
    from scipy.spatial.transform import Rotation as R
    raw = glb.read_bytes()
    g = json.loads(raw[20:20 + struct.unpack("<I", raw[12:16])[0]])
    nodes = g["nodes"]
    name_of = {i: n.get("name", str(i)) for i, n in enumerate(nodes)}
    parent = {c: i for i, n in enumerate(nodes) for c in n.get("children", [])}

    def root_name(i):                         # the top-level node a node hangs under
        while i in parent:
            i = parent[i]
        return name_of[i]

    def under(i, names):                      # is the node, or an ancestor, one of names
        while True:
            if name_of[i] in names:
                return True
            if i not in parent:
                return False
            i = parent[i]

    idx = {n.get("name"): i for i, n in enumerate(nodes)}
    chain = [idx[n] for n in ARM_CHAIN]
    rest, stow = [], []
    Wr, Ws = np.eye(4), np.eye(4)
    for i, ax, deg in zip(chain, ARM_AXES, ARM_STOW_DEG):
        L = _local(nodes[i])
        D = np.eye(4)
        D[:3, :3] = R.from_euler(ax, np.radians(deg)).as_matrix()
        Wr, Ws = Wr @ L, Ws @ L @ D
        rest.append(Wr.copy())
        stow.append(Ws.copy())
    A = np.linalg.inv(ICP_MODEL_FROM_MECH) @ np.array([[0, 0, 1, 0], [-1, 0, 0, 0], [0, -1, 0, DECK_TOP_MODEL_Y], [0, 0, 0, 1.0]])

    def world(i):
        W = _local(nodes[i])
        while i in parent:
            i = parent[i]
            W = _local(nodes[i]) @ W
        return W

    def link_of(i):                           # the arm link a node moves with (the nearest chain ancestor), or None
        while True:
            if i in chain:
                return chain.index(i)
            if i not in parent:
                return None
            i = parent[i]

    sc = trimesh.load(str(glb), process=False)
    geo_nodes = {}                            # trimesh names its nodes <node>[_hash]; map them back by the mesh index
    mesh_names = [m.get("name") for m in g["meshes"]]
    parts = {p: [] for p in PARTS}
    for i, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        k = link_of(i)
        if under(i, MAST) or under(i, BELLY) or under(i, MOVING):
            continue
        if k is not None:
            kind, move = "arm", stow[k] @ np.linalg.inv(rest[k])
        else:
            nm = root_name(i)
            kind = ("arm_base" if nm.startswith(("arm.001", "arm_cable")) else "wheels" if nm.startswith("Wheels")
                    else "suspension" if nm.startswith("suspension") else "body")
            move = np.eye(4)
        W = move @ world(i)
        mesh = g["meshes"][n["mesh"]]
        for pi in range(len(mesh["primitives"])):
            cands = [gn for gn in sc.geometry if gn == mesh_names[n["mesh"]] or gn.startswith(mesh_names[n["mesh"]] + "_")]
            geo_nodes.setdefault(n["mesh"], sorted(cands))
        for gn in geo_nodes.get(n["mesh"], []):
            m = sc.geometry[gn]
            v = trimesh.transform_points(trimesh.transform_points(np.asarray(m.vertices, float), W), A)
            parts[kind].append(trimesh.Trimesh(v, np.asarray(m.faces), process=False))
    V, F, P, rep = [], [], [], {}
    off = 0
    for pid, kind in enumerate(PARTS):
        if not parts[kind]:
            continue
        m = trimesh.util.concatenate(parts[kind])
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
        om.remove_duplicated_vertices()
        om.remove_degenerate_triangles()
        n0 = len(om.triangles)
        if n0 > TARGET_FACES[kind]:
            om = om.simplify_quadric_decimation(TARGET_FACES[kind])
            om.remove_unreferenced_vertices()
        v, f = np.asarray(om.vertices), np.asarray(om.triangles)
        V.append(v)
        F.append(f + off)
        P.append(np.full(len(f), pid, np.uint8))
        off += len(v)
        rep[kind] = {"faces_in": int(n0), "faces": int(len(f)), "bounds_mech_m": np.round([v.min(0), v.max(0)], 3).tolist()}
    V, F, P = np.vstack(V).astype(np.float32), np.vstack(F).astype(np.int32), np.concatenate(P)
    meta = {"source": glb.name, "credit": "NASA/JPL-Caltech", "url": "https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/",
            "frame": "RMECH (x forward, y right, z down, z 0 = the deck top), metres", "parts": list(PARTS),
            "arm_chain": list(ARM_CHAIN), "arm_axes": list(ARM_AXES), "arm_stow_deg": list(ARM_STOW_DEG), "report": rep}
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, vertices=V, faces=F, part=P, meta=json.dumps(meta))
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("glb")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "src" / "mppp" / "data" / "rover" / "perseverance_rmech.npz"))
    a = ap.parse_args()
    m = build(Path(a.glb), Path(a.out))
    print(json.dumps(m["report"], indent=1))
    print(a.out, Path(a.out).stat().st_size, "bytes")

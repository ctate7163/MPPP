"""
Build ``src/mppp/data/rover/perseverance_rmech.npz`` - the simplified Perseverance model of mppp.ibr.rover_model -
from NASA's glTF model (https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/, 25042_Perseverance.glb,
credit NASA/JPL-Caltech; Christian, 10 Nov 2026: "take this model of perseverance's, simplify it, add it to the mesh,
and project the images onto it. The arm is stowed, so you would need to rotate the arm joints").

    python scripts/build_rover_model.py 25042_Perseverance.glb [--out FILE]

* Frame: the model is in metres, x left, y up, z forward, the wheels on y = 0; RMECH (x forward, y right, z down,
  z 0 = the deck top) is x = z_model, y = -x_model, z = 1.135 - y_model, then the small rigid correction of an ICP of
  the Van Zyl S003D2046 rover cloud onto the body (0.12 deg, 2-3 mm; 60 % of the cloud within 3 cm of the body).
* Left out: the mast above its base and the head (they turn with the cameras), the parts under the belly.
* The high-gain antenna (0.97.0): the part "antenna", in one neutral pose (HGA_AZ_DEG, HGA_TILT_DEG) - it points
  differently every sol, and the model does not follow it.
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
# 0.97.0 (Christian, 12 Nov 2026: "use those arm joint angles to estimate where the arm is in space"): the model's
# joint angles from the image labels' resolver angles (ARM_ARTICULATION_STATE, JOINT 1-5 RESOLVER, degrees):
#   model = ARM_LABEL_SIGN * label + ARM_LABEL_OFFSET_DEG.
# Signs from six frames with the arm out (sols 18, 46 x 2, 74, 530, 1995: edge distance of the rendered outline, 32
# sign sets tried; the turret's from sol 46 by eye).  The offsets are the round ones (0, 0, -90, +90, 0): with them
# the outline is 1.5-2.7 times closer to the image edges than with the stow pose fitted by eye before, in the six
# frames and in three stowed ones (sols 52, 461, 1463).  The stow pose is the median resolver stow of 443 stows,
# sols 12-2022 (90.09, -16.37, -161.66, 178.18, 278.52 deg; sd 7-29 mdeg).  Before (12 Nov, by eye): (90.0, 14.5,
# 68.8, -89.4, 77.5).
ARM_LABEL_SIGN = (1.0, -1.0, -1.0, -1.0, -1.0)
ARM_LABEL_OFFSET_DEG = (0.0, 0.0, -90.0, 90.0, 0.0)
ARM_LABEL_STOW_DEG = (90.09, -16.37, -161.66, 178.18, 278.52)
ARM_STOW_DEG = (90.09, 16.37, 71.66, -88.18, 81.48)
# that the drill points out and pixl toward the mast. the PERSEVERANCE plate on the arm can rotate up a little"): the
# turret turned by 175 deg (-90.1 -> 85.0), the wrist -103.3 -> -93.3 and the shoulder elevation 16.4 -> 20.4, chosen
# by eye from renders of the model through the label camera of NLF_0461_0707871340 (Rockytop S026D0630) against the
# frame: the drill's dome, PIXL's box, the bit and the top instruments within about 20 px (half-resolution frame).
# Then the shoulder azimuth 89.2 -> 96.0 (Christian: "align the PERSEVERANCE plate on the front forearm"): the forearm's
# slope in NRF_0052_0671568531 (Van Zyl S003D2046) - 38 deg below the image's horizontal at 94, about 43 in the
# frame - with the turret in the Rockytop frame still in place.
# Then (Christian: "az90 is more right for that joint", "the turret ... maybe down just a little"): the shoulder
# azimuth held at 90 and the other four angles fitted by least squares to marked points - the drill's dome, PIXL and
# the bit in NLF_0461_0707871340 (Rockytop), PIXL and the nameplate's centre line in NRF_0052_0671568531 (Van Zyl):
# (14.5, 66.3, -86.9, 77.5), rms 44 px at half resolution (the marks are good to 20-30 px); with the azimuth free the
# same fit gives 90.9.
# Then (Christian: "the arm turret is too high. Bring it down a little while leaving the far side of the forearm in
# place"): the elbow +2.5 deg with the wrist -2.5 deg - the forearm turns about its elbow end, the turret keeps its
# attitude and comes down 3 cm (about 45 px in the Rockytop frame).
# Before (10 Nov): the fit to the Van Zyl cloud, which left the turret's own angle the least certain - it was 180 deg off.
# 0.97.0 (Christian, 12 Nov 2026: "there's a black rubber thing on the far side of the middle deck at looks like it
# should be rotated 30-45 degrees" ... "rotate it in azimuth back 30 degrees"): the ribbed block of the body (a
# connected component 21 cm along y, 7 cm tall, centred at near_m) turned 30 deg about the vertical through its own
# centre, so that its ribbed face looks towards the mast as in NRF_0461_0707874458 (-30 in turn_components' sense;
# +30 shows it end on).  A lean about its long axis (tried first) was the wrong axis.  Then ("the back thing should
# be anchored on the right side and then rotated so that its right extent stays in place"): the pivot is the block's
# -y end (y = -0.194 m, the right end as the mast cameras see it looking aft, by the roller bracket), not its centre.
BODY_TURNS = ({"near_m": (-0.323, -0.088, -0.048), "pivot_xy_m": (-0.323, -0.194), "angle_deg": -30.0},)
STRAY_M = 0.012              # a decimated triangle this far from the model it was made from is a stray one
HGA_NODES = ("antenna_hg",)
HGA_AZ_DEG = 90.0            # 0.97.0 (Christian, 12 Nov 2026: "put the high gain antenna back in and have it point in a
HGA_TILT_DEG = -45.0         # neutral direction ... tilted forward at a 45 degree angle"): the gimbal's arm towards +y
#                              (the dish to the right of the gimbal), the dish turned 45 deg about that arm so that it
#                              faces forward and up.  Checked against NRF_0461_0707874458 (Rockytop) and
#                              NLF_0052_0671568487 (Van Zyl): the dish's outline falls on the dish in both.
HGA_AXIS_XY_M = (-0.46, -0.475)          # RMECH: the gimbal's vertical (azimuth) axis
HGA_GIMBAL_M = (-0.436, -0.463, -0.25)   # RMECH: the top of the gimbal, where the dish's arm starts
HGA_MODEL_AZ_DEG = 25.0                  # the direction of that arm in NASA's model (dish flat, pointing up)
HGA_FIXED_BELOW_Z_M = -0.13              # RMECH z (down): components centred below this height do not turn (the base)
DECK_TOP_MODEL_Y = 1.135
# 0.97.0 (Christian, 12 Nov 2026: "find the fiducials on the 3D model deck and say if they are consistent with the
# surveyed locations"): NASA's model carries the ten deck targets as discs of 24 mm radius.  The placement is the rigid
# fit of seven of them (TDF1-4, LDIFF, RDIFF, AFT1) to the survey (fiducials.SURVEY_MECH_MM, TDF4 at its fitted
# position): rms 2.2 mm.  RPFA1, RPFA2 and AFT2 are 6-18 mm off in the model and are left out.  Before: an ICP fit of
# the model to the Van Zyl cloud, [[1, -2.0e-4, -2.0e-3, 0], [2.0e-4, 1, 1.1e-3, 2.1e-3], [2.0e-3, -1.1e-3, 1, 2.7e-3]],
# which put the forward deck targets 4 mm high (8.8 px rms in NRF_1642_0812712250 through the Navcam law and the
# deck_joint mast chain; this placement 2.8 px; the survey itself 1.5 px).  The name is kept.
ICP_MODEL_FROM_MECH = np.array([[1.000002, -0.000654, 0.000622, -0.000243],
                                [0.000651, 1.000000, 0.000990, 0.000791],
                                [-0.000623, -0.000989, 1.000002, 0.000798],
                                [0.0, 0.0, 0.0, 1.0]])
TARGET_FACES = {"body": 32000, "arm": 10000, "arm_base": 2300, "suspension": 8000, "wheels": 9000, "mast": 6000, "antenna": 6000}
PARTS = ("body", "arm", "arm_base", "suspension", "wheels", "mast", "antenna")
MAST_PARTS = ("bottom", "bottom.002", "bottom.003", "probe.001", "top", "probe")   # turn with the head's azimuth: the
# mast tube and the wind-sensor booms on it; kept as the part "mast" (not for the mesh: for masking them in the
# frames, mppp.ibr.rover_model.mast_mask), in the model's own azimuth - the head looking forward (0 deg)
HEAD = ("head", "lens", "Mastcam_Z_cams", "microphones.001", "NavCams")
MAST = ("bottom", "bottom.002", "bottom.003", "probe.001", "top", "probe", "head", "lens", "Mastcam_Z_cams",
        "microphones.001", "NavCams")
MOVING = ()                               # 0.97.0: the high-gain antenna is the part "antenna", in a neutral pose (HGA_*)
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


def turn_components(v: np.ndarray, f: np.ndarray, turns=None) -> np.ndarray:
    """``v`` with the connected components named by BODY_TURNS (centroid within 2 cm of ``near_m``) turned by
    ``angle_deg`` about the vertical through ``pivot_xy_m`` (RMECH; positive: the -y end swings to -x), or - ``axis``
    "y" - about the line along y through (x, z) = ``pivot_xz_m``."""
    import trimesh
    turns = BODY_TURNS if turns is None else turns
    lo, hi = v.min(0), v.max(0)
    todo = [t for t in turns if np.all(np.asarray(t["near_m"]) > lo - 0.05) and np.all(np.asarray(t["near_m"]) < hi + 0.05)]
    if not todo:
        return v
    v = v.copy()
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    # components over coincident vertices (the primitives repeat a vertex per face corner)
    key = np.round(v, 5)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    inv = inv.ravel()
    fi = inv[f]
    n = int(inv.max()) + 1
    g = sp.coo_matrix((np.ones(3 * len(f)), (np.r_[fi[:, 0], fi[:, 1], fi[:, 2]], np.r_[fi[:, 1], fi[:, 2], fi[:, 0]])), shape=(n, n))
    _, lab = connected_components(g, directed=False)
    vlab = lab[inv]
    for t in todo:
        near = np.asarray(t["near_m"], float)
        for c in np.unique(vlab):
            sel = vlab == c
            if np.linalg.norm(v[sel].mean(0) - near) < 0.02:
                a = np.radians(t["angle_deg"])
                if t.get("axis", "z") == "y":             # about the line along y through (x, z) = pivot_xz_m
                    px, pz = t["pivot_xz_m"]
                    x, z = v[sel, 0] - px, v[sel, 2] - pz
                    v[sel, 0] = px + np.cos(a) * x + np.sin(a) * z
                    v[sel, 2] = pz - np.sin(a) * x + np.cos(a) * z
                    continue
                px, py = t["pivot_xy_m"]
                x, y = v[sel, 0] - px, v[sel, 1] - py
                v[sel, 0] = px + np.cos(a) * x + np.sin(a) * y
                v[sel, 1] = py - np.sin(a) * x + np.cos(a) * y
    return v


def pose_hga(X: np.ndarray, F: np.ndarray, az_deg: float = HGA_AZ_DEG, tilt_deg: float = HGA_TILT_DEG):
    """The high-gain antenna of NASA's model (vertices ``X`` in RMECH) in a pose: every connected component above the
    base turned about the gimbal's vertical axis so that the dish's arm points to ``az_deg`` (from +x towards +y),
    then the dish (the largest component, and what sits on it) turned by ``tilt_deg`` about that arm."""
    import trimesh
    from scipy.spatial.transform import Rotation as R
    ax = np.array([HGA_AXIS_XY_M[0], HGA_AXIS_XY_M[1], 0.0])
    g0 = np.asarray(HGA_GIMBAL_M, float)
    comps = trimesh.Trimesh(X, F, process=False).split(only_watertight=False)
    dish = max(comps, key=lambda c: c.area)
    Rz = R.from_euler("z", np.radians(az_deg - HGA_MODEL_AZ_DEG)).as_matrix()
    a = np.array([np.cos(np.radians(az_deg)), np.sin(np.radians(az_deg)), 0.0])
    g = (g0 - ax) @ Rz.T + ax
    Rt = R.from_rotvec(a * np.radians(tilt_deg)).as_matrix()
    Vs, Fs, off = [], [], 0
    for c in comps:
        v = np.asarray(c.vertices, float).copy()
        cz = float(c.centroid[2])
        if cz < HGA_FIXED_BELOW_Z_M:
            v = (v - ax) @ Rz.T + ax
            on_dish = c is dish or (np.linalg.norm(c.centroid[:2] - dish.centroid[:2]) < 0.12 and cz < -0.2
                                    and np.linalg.norm(c.centroid[:2] - ax[:2]) > 0.1)
            if on_dish:
                v = (v - g) @ Rt.T + g
        Vs.append(v)
        Fs.append(np.asarray(c.faces) + off)
        off += len(v)
    return np.vstack(Vs), np.vstack(Fs)


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
        if under(i, HEAD) or under(i, BELLY) or under(i, MOVING):
            continue
        is_mast = under(i, MAST_PARTS)
        is_hga = under(i, HGA_NODES)
        if k is not None:
            kind, move = "arm", stow[k] @ np.linalg.inv(rest[k])
        else:
            nm = root_name(i)
            kind = ("antenna" if is_hga else "mast" if is_mast else "arm_base" if nm.startswith(("arm.001", "arm_cable")) else "wheels" if nm.startswith("Wheels")
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
            fm = np.asarray(m.faces)
            if kind == "body":
                v = turn_components(v, fm)
            parts[kind].append(trimesh.Trimesh(v, fm, process=False))
    V, F, P, rep = [], [], [], {}
    off = 0
    for pid, kind in enumerate(PARTS):
        if not parts[kind]:
            continue
        m = trimesh.util.concatenate(parts[kind])
        if kind == "antenna":                 # the dish in its neutral pose
            hv, hf = pose_hga(np.asarray(m.vertices, float), np.asarray(m.faces))
            m = trimesh.Trimesh(hv, hf, process=False)
        om = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(m.vertices), o3d.utility.Vector3iVector(m.faces))
        om.remove_duplicated_vertices()
        om.remove_degenerate_triangles()
        n0 = len(om.triangles)
        stray = 0
        if n0 > TARGET_FACES[kind]:
            ref = o3d.t.geometry.RaycastingScene()
            ref.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(om))
            om = om.simplify_quadric_decimation(TARGET_FACES[kind])
            # 0.97.0 (Christian, 12 Nov 2026: "there seem to be some extra polygons cutting across the deck"): the
            # decimation joins parts that only touch (the deck's bar to the mast's base) with long triangles through
            # the air; a triangle with its centroid or an edge's middle farther than STRAY_M from the model goes
            v_, f_ = np.asarray(om.vertices), np.asarray(om.triangles)
            a_ = v_[f_]
            probes = np.stack([a_.mean(1), 0.5 * (a_[:, 0] + a_[:, 1]), 0.5 * (a_[:, 1] + a_[:, 2]), 0.5 * (a_[:, 2] + a_[:, 0])], 1)
            dist = ref.compute_distance(o3d.core.Tensor(probes.reshape(-1, 3).astype(np.float32))).numpy().reshape(-1, 4).max(1)
            bad = dist > STRAY_M
            stray = int(bad.sum())
            if stray:
                om.remove_triangles_by_mask(bad.tolist())
            om.remove_unreferenced_vertices()
        v, f = np.asarray(om.vertices), np.asarray(om.triangles)
        V.append(v)
        F.append(f + off)
        P.append(np.full(len(f), pid, np.uint8))
        off += len(v)
        rep[kind] = {"faces_in": int(n0), "faces": int(len(f)), "stray_faces_removed": stray, "bounds_mech_m": np.round([v.min(0), v.max(0)], 3).tolist()}
    V, F, P = np.vstack(V).astype(np.float32), np.vstack(F).astype(np.int32), np.concatenate(P)
    axis_xy = [0.716, 0.559]                                         # the pivot fit of the Van Zyl frames; the tube is centred on it
    meta = {"source": glb.name, "credit": "NASA/JPL-Caltech", "url": "https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/",
            "frame": "RMECH (x forward, y right, z down, z 0 = the deck top), metres", "parts": list(PARTS),
            "arm_chain": list(ARM_CHAIN), "arm_axes": list(ARM_AXES), "arm_stow_deg": list(ARM_STOW_DEG), "report": rep,
            "hga": {"az_deg": HGA_AZ_DEG, "tilt_deg": HGA_TILT_DEG, "note": "a neutral pose, not the sol's pointing"},
            "mast_axis_xy_m": axis_xy, "mast_model_azimuth_deg": 0.0}
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

"""
Build ``src/mppp/data/rover/perseverance_rmech.npz`` - the simplified Perseverance model of mppp.ibr.rover_model -
from NASA's glTF model (https://science.nasa.gov/resource/mars-perseverance-rover-3d-model/, 25042_Perseverance.glb,
credit NASA/JPL-Caltech; Christian, 10 Nov 2026: "take this model of perseverance's, simplify it, add it to the mesh,
and project the images onto it. The arm is stowed, so you would need to rotate the arm joints").

    python scripts/build_rover_model.py 25042_Perseverance.glb [--out FILE]

The changes to NASA's model are described for users in docs/rover-model.md; the numbers are the constants below.

* Frame: the model is in metres, x left, y up, z forward, the wheels on y = 0; RMECH (x forward, y right, z down,
  z 0 = the deck top) is x = z_model, y = -x_model, z = 1.135 - y_model, then the rigid fit of seven fiducial discs
  of the model's deck to their surveyed positions (2.2 mm rms; before 0.97.0 an ICP of the Van Zyl rover cloud).
* Left out: the mast above its base and the head (they turn with the cameras), the parts under the belly.
* The high-gain antenna (0.97.0): the part "antenna", in one neutral pose (HGA_AZ_DEG, HGA_TILT_DEG) - it points
  differently every sol, and the model does not follow it.
* The arm: the chain arm.003 (shoulder azimuth, about its local y) - arm.002 (shoulder elevation, local z) - arm
  (elbow, z) - arm.004 (wrist, z) - turret_obj (turret, y), turned from the model's deployed pose to the stow pose
  of the image labels' arm resolver angles (ARM_LABEL_STOW_DEG through ARM_LABEL_SIGN and ARM_LABEL_OFFSET_DEG).
* The UHF antenna can: its wall raised to the real rim and the domed cap replaced by a flat floor (UHF_CAN_*).
* Pieces of the body moved to where the images show them (BODY_COMPONENT_SHIFTS).
* Not simplified since 0.97.7: NASA's triangles as they are (TARGET_FACES; 0.96.1-0.97.6 decimated every part).
* scripts/export_rover_glb.py writes the result as docs/rover/perseverance_rmech.glb for use outside MPPP.
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
# 0.97.0 (Christian, 12 Nov 2026, in VR: "the grey cylinder above the caltargets ... looks a little out of place"):
# the UHF antenna's can is too small in NASA's model.  Through the label camera of NRF_0052_0671564560 its stand
# falls on the stand, but the can is 0.81 of the real one across and its top 27 px (1 deg) low.  The can (the
# geometry of the node above UHF_CAN_BELOW_Z_M) is scaled about its own axis and from its base.
UHF_NODES = ("antenna_uhf",)
UHF_CAN_BELOW_Z_M = -0.43         # RMECH z (down): the can is the geometry wholly above this height
# 13 Nov 2026 (Christian: "the gray cylinder uhf antenna looks mis-sized again"): that enlargement was wrong.  With
# the model placed by the deck discs, the silhouette through the label cameras of four frames (NLF / NRF_0052_0671564560,
# NLF_0474_0709026896, NLF_0688_0728025880) puts NASA's can on the real one at its original width; its wall ends
# about 10 % below the real rim (its far edge), and its domed cap is not there (the real can is open): width as modelled, height
# 1.10, the cap flattened and set just inside the rim.
UHF_CAN_RADIAL = 1.0
UHF_CAN_HEIGHT = 1.10              # 13 Nov, later: to the far top rim (it blends into the ground; found on contrast-stretched crops)
UHF_CAN_FLAT_TOP = True
UHF_CAN_CAP_BELOW_RIM_M = 0.05     # the open can's inside: a floor this far below the rim (not measured)
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
# 0.97.7 (Christian, 13 Nov 2026: "too many polygons were simplified.  Please restore the original polygon detail on
# everything that the mast cameras see ... the forearm plate and the wheels are too simplified"): no part is
# decimated any more - NASA's triangles as they are (197,000; before: body 32000, arm 10000, arm_base 2300,
# suspension 8000, wheels 9000, mast 6000 - 71,000 in all).  A number here decimates that part to it.
TARGET_FACES = {"body": None, "arm": None, "arm_base": None, "suspension": None, "wheels": None, "mast": None, "antenna": None, "head": None}
PARTS = ("body", "arm", "arm_base", "suspension", "wheels", "mast", "antenna", "head")
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


# 0.97.8 (Christian, 13 Nov 2026, looking at the model's outline on NLF_0052_0671564132: "I also noticed this dark
# object here.  In the model, it needs to move back"): connected pieces of the body that lie wholly inside a box
# (RMECH, m) are moved by a vector (m).  The dark block at the front right under the arm: NASA's model has it 5 cm
# too far forward and 2 cm too far left against the Van Zyl pair NLF / NRF_0052_0671564132 (its outline through a
# projection fitted to each frame's own depth map, tried at 0 to -9 cm in x and 0, +2 cm in y; good to about 2 cm).
BODY_COMPONENT_SHIFTS = [
    {"name": "dark block at the front right under the arm", "box_min": (1.05, 0.24, 0.19), "box_max": (1.24, 0.40, 0.42),
     "shift_m": (-0.05, 0.02, 0.0)},
]


def shift_components(v: np.ndarray, f: np.ndarray, shifts) -> list:
    """Move (in place) every connected piece of the mesh (v, f) that lies wholly inside a shift's box; the report."""
    import scipy.sparse as sp
    import scipy.sparse.csgraph as cg
    e = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n, lab = cg.connected_components(sp.coo_matrix((np.ones(len(e)), (e[:, 0], e[:, 1])), shape=(len(v), len(v))), directed=False)
    lo = np.full((n, 3), np.inf)
    hi = np.full((n, 3), -np.inf)
    np.minimum.at(lo, lab, v)
    np.maximum.at(hi, lab, v)
    used = np.zeros(n, bool)
    used[lab[np.unique(f)]] = True
    rep = []
    for sh in shifts:
        inside = used & np.all(lo >= np.asarray(sh["box_min"]), axis=1) & np.all(hi <= np.asarray(sh["box_max"]), axis=1)
        sel = inside[lab]
        v[sel] += np.asarray(sh["shift_m"], float)
        rep.append({"name": sh["name"], "pieces": int(inside.sum()), "vertices": int(sel.sum()), "shift_m": list(sh["shift_m"])})
        print(f"moved {int(inside.sum())} piece(s), {int(sel.sum())} vertices: {sh['name']} by {sh['shift_m']} m")
    return rep


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
    head_nodes = {}
    for i, n in enumerate(nodes):
        if "mesh" not in n:
            continue
        k = link_of(i)
        if under(i, BELLY) or under(i, MOVING):
            continue
        is_head = under(i, HEAD)
        is_mast = under(i, MAST_PARTS)
        is_hga = under(i, HGA_NODES)
        if k is not None:
            kind, move = "arm", stow[k] @ np.linalg.inv(rest[k])
        else:
            nm = root_name(i)
            kind = ("head" if is_head else "antenna" if is_hga else "mast" if is_mast else "arm_base" if nm.startswith(("arm.001", "arm_cable")) else "wheels" if nm.startswith("Wheels")
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
            if under(i, UHF_NODES) and len(v) and v[:, 2].max() < UHF_CAN_BELOW_Z_M:      # the can, not its stand
                c, zb = 0.5 * (v[:, :2].min(0) + v[:, :2].max(0)), v[:, 2].max()
                v = np.c_[c + UHF_CAN_RADIAL * (v[:, :2] - c), zb + UHF_CAN_HEIGHT * (v[:, 2] - zb)]
                if UHF_CAN_FLAT_TOP:                                   # nothing above the wall's top (z is down)
                    rr = np.linalg.norm(v[:, :2] - c, axis=1)
                    wall = rr > 0.97 * rr.max()
                    if wall.any():                                     # the cap: a flat floor a little inside the rim
                        zt = v[wall, 2].min()                          # (Christian, 13 Nov: "the sphere on the top
                        cap = ~wall & (v[:, 2] < zt + 1e-6)            # center of the uhfa should be lower")
                        v[cap, 2] = zt + UHF_CAN_CAP_BELOW_RIM_M
                        v[:, 2] = np.maximum(v[:, 2], zt)
            parts[kind].append(trimesh.Trimesh(v, fm, process=False))
            if kind == "head" and len(v):
                head_nodes.setdefault(name_of[i], []).append(v)
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
        if TARGET_FACES[kind] and n0 > TARGET_FACES[kind]:
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
        if kind == "body" and BODY_COMPONENT_SHIFTS:
            v = np.array(v, float)
            rep_shift = shift_components(v, f, BODY_COMPONENT_SHIFTS)
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
            "mast_axis_xy_m": axis_xy, "mast_model_azimuth_deg": 0.0,
            "head_nodes_bounds_mech_m": {k: np.round([np.vstack(v).min(0), np.vstack(v).max(0)], 4).tolist() for k, v in head_nodes.items()}}
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

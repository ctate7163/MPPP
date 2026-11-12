"""
mppp.ibr.raycast - per-image depth, range, normal and label maps cast from one surface (plan of 31 Oct, step 3).

A ray through every pixel centre of an image, through the image's **exact** camera model (``pycolmap.Camera.
cam_from_img``: THIN_PRISM_FISHEYE for the Navcam, FULL_OPENCV for Mastcam-Z), rotated into the block frame and
cast at the station's surface (``open3d.t.geometry.RaycastingScene``).  One surface, many views: the maps of two
images agree by construction, which the patch-match maps never do.

The surface (Christian, 10 Nov 2026: "cut off at 20 m from the axis of the mast rotation, beyond that, project onto a
flat plain out to 100 m, and beyond that onto a sphere at infinity"), in order of priority: the station's **mesh**
(within 30 m of the mast axis); where a ray misses it, the **ground plane** (fitted to the terrain, out to 100 m from
the axis; it also fills the mesh's holes); where it misses both, the **sky at infinity** (label "far": a direction,
no depth).  The mesh wins over the plane even where the plane is nearer (a hollow below it), and only the mesh
occludes.  A finite far sphere (the 0.96.0 prototype's) is still taken when one is given.

Per pixel: ``depth`` (z along the optical axis, metres, COLMAP's convention), ``range`` (along the ray; a fisheye
pixel near 90 degrees has z -> 0), ``normal`` (camera frame, facing the camera), ``label`` (0 none, 1 mesh, 2 far:
the sky at infinity - depth and range 0 - or the finite far sphere, 3 masked by the image's rover mask, 4 the ground
plane) and the triangle id.  Stored as ``maps/<image>.npz`` (depth float32,
range float16, normal int8 x 127, label uint8, tri int32) with PNG previews.

Resolution: ``scale`` of the frame's own size - 1 by default, its own pixel grid (10 Nov; 0.5 before); Mastcam-Z
``zcam_scale`` 1.  The
camera is never rescaled: the scaled grid's pixel centres are mapped to native pixel coordinates first.

Validation (``maps_report.json``): the observed sparse points of every image against the cast depth at their
keypoints (median |dz|/z and the 90th percentile in range bins), and the coverage per image (mesh / far / none /
masked).  Mastcam-Z frames, which took no part in the mesh, are an independent check of both the surface and the
Mastcam-Z calibration.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List, Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

log = logging.getLogger("mppp.ibr.raycast")
PathLike = Union[str, Path]

LABEL_NONE, LABEL_MESH, LABEL_FAR, LABEL_MASKED, LABEL_PLANE = 0, 1, 2, 3, 4
RANGE_BINS = ((0, 5), (5, 10), (10, 20), (20, 60), (60, 1e9))
SKY_RANGE_M = 1.0e6     # "infinity" for the rays that end on the sky: the 64 mm ODS baseline subtends 6e-8 rad there
INVALID_ID = 4294967295


class Scene:
    """The station's surface: the mesh (and a finite far sphere, if given) in one ray-casting scene, then the ground
    ``plane`` (a dict with ``normal``, ``offset`` - n . X = offset - ``centre``, ``up`` and ``radius_m``, the
    horizontal reach from the mast axis; :func:`mppp.ibr.mesh.fit_ground_plane`), then the ``sky`` at infinity."""

    def __init__(self, mesh: Tuple[np.ndarray, np.ndarray], sphere: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 plane: Optional[Dict[str, Any]] = None, sky: bool = False):
        import open3d as o3d
        self.plane = None
        if plane is not None:
            n = np.asarray(plane["normal"], float)
            self.plane = {"normal": n / np.linalg.norm(n), "offset": float(plane["offset"]),
                          "centre": np.asarray(plane["centre"], float), "up": np.asarray(plane.get("up", (0, 0, 1)), float),
                          "radius_m": float(plane["radius_m"])}
        self.sky = bool(sky)
        self.o3d = o3d
        self.scene = o3d.t.geometry.RaycastingScene()
        self.ids: Dict[int, int] = {}
        V, F = mesh
        self.n_mesh_triangles = int(len(F))
        if len(F):
            gid = self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(_legacy(o3d, V, F)))
            self.ids[int(gid)] = LABEL_MESH
        if sphere is not None and len(sphere[1]):
            gid = self.scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(_legacy(o3d, *sphere)))
            self.ids[int(gid)] = LABEL_FAR
        self.mesh_V, self.mesh_F = V, F
        self.rover_triangles: Optional[Tuple[int, int]] = None     # 0.97.0: [first, end) of the rover model in the mesh
        self.sky_radius_m: Optional[float] = None    # 0.97.0: the sky on a sphere of this radius about the plane's centre

    def set_sky_sphere(self, radius_m: Optional[float]) -> None:
        """0.97.0 (Christian, 12 Nov 2026: "the 3d terrain should probably stop at a 100 or 200 m radius sky sphere"
        ... "Extend the ground plane to the sphere so mesh, plane and sphere meet without a gap in depth"): the rays
        that miss mesh and plane end on a sphere of ``radius_m`` about the plane's centre, and the plane reaches to
        it.  None or 0: the sky at infinity, the plane as fitted.  Needs the plane (its centre)."""
        if not radius_m or self.plane is None or not self.sky:
            self.sky_radius_m = None
            return
        self.sky_radius_m = float(radius_m)
        self.plane["radius_m"] = float(radius_m)

    def on_rover(self, tri: np.ndarray) -> Optional[np.ndarray]:
        """True where a mesh triangle index is one of the rover model's; None when the mesh holds no model."""
        if self.rover_triangles is None:
            return None
        a, b = self.rover_triangles
        t = np.asarray(tri)
        return (t >= a) & (t < b)

    def _cast_geometry(self, origins: np.ndarray, dirs: np.ndarray) -> Dict[str, np.ndarray]:
        """The mesh and the finite sphere only (nearest hit)."""
        o = np.broadcast_to(np.asarray(origins, np.float32), np.asarray(dirs).shape)
        if not self.ids:
            n = len(o)
            return {"t": np.full(n, np.inf, np.float64), "label": np.zeros(n, np.uint8), "normal": np.zeros((n, 3), np.float32),
                    "tri": np.full(n, -1, np.int64)}
        rays = self.o3d.core.Tensor(np.ascontiguousarray(np.concatenate([o, np.asarray(dirs, np.float32)], axis=1)))
        ans = self.scene.cast_rays(rays)
        t = ans["t_hit"].numpy().astype(np.float64)
        gid = ans["geometry_ids"].numpy()
        label = np.zeros(len(t), np.uint8)
        for g, lab in self.ids.items():
            label[gid == g] = lab
        label[~np.isfinite(t)] = LABEL_NONE
        tri = ans["primitive_ids"].numpy().astype(np.int64)
        tri[label != LABEL_MESH] = -1
        return {"t": t, "label": label, "normal": ans["primitive_normals"].numpy().copy(), "tri": tri}

    def plane_hits(self, origins: np.ndarray, dirs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """(t, hit) of the rays on the ground plane within its horizontal reach (t inf where it misses)."""
        d = np.asarray(dirs, float)
        o = np.broadcast_to(np.asarray(origins, float), d.shape)
        pl = self.plane
        t = np.full(len(d), np.inf)
        if pl is None:
            return t, np.zeros(len(d), bool)
        n = pl["normal"]
        den = d @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            tt = (pl["offset"] - o @ n) / den
        ok = np.isfinite(tt) & (tt > 1e-6) & (np.abs(den) > 1e-9)
        X = o + np.where(ok, tt, 0.0)[:, None] * d
        v = X - pl["centre"]
        h = v - (v @ pl["up"])[:, None] * pl["up"]
        ok &= np.linalg.norm(h, axis=1) <= pl["radius_m"]
        t[ok] = tt[ok]
        return t, ok

    def cast(self, origins: np.ndarray, dirs: np.ndarray) -> Dict[str, np.ndarray]:
        """Rays (n, 3) + (n, 3) -> {"t" range (inf: no hit; SKY_RANGE_M on the sky), "label", "normal" (world, unit or
        0), "tri"}: the mesh first, the plane where the mesh is missed, the sky where both are."""
        hit = self._cast_geometry(origins, dirs)
        label, t = hit["label"], hit["t"]
        if self.plane is not None:
            tp, okp = self.plane_hits(origins, dirs)
            take = okp & (label != LABEL_MESH) & ((label == LABEL_NONE) | (tp < t))
            t[take] = tp[take]
            label[take] = LABEL_PLANE
            hit["normal"][take] = self.plane["normal"]
            hit["tri"][take] = -1
        if self.sky:
            miss = label == LABEL_NONE
            t[miss] = SKY_RANGE_M
            if self.sky_radius_m and miss.any():                 # the sphere, from inside it (else infinity)
                d = np.asarray(dirs, float)[miss]
                v = np.broadcast_to(np.asarray(origins, float), (len(label), 3))[miss] - self.plane["centre"]
                b = np.einsum("ij,ij->i", v, d)
                disc = b * b - (np.einsum("ij,ij->i", v, v) - self.sky_radius_m ** 2)
                ts = -b + np.sqrt(np.maximum(disc, 0.0))
                t[miss] = np.where((disc > 0) & (ts > 0), ts, SKY_RANGE_M)
            label[miss] = LABEL_FAR
            hit["normal"][miss] = 0.0
        return {"t": t, "label": label, "normal": hit["normal"], "tri": hit["tri"]}

    def occluded(self, origins: np.ndarray, targets: np.ndarray, tol: np.ndarray) -> np.ndarray:
        """True where the segment origin -> target meets the mesh more than ``tol`` before the target (the plane and
        the sky never occlude: the plane is a stand-in for terrain the mesh does not reach)."""
        d = np.asarray(targets, float) - np.asarray(origins, float)
        r = np.linalg.norm(d, axis=1)
        d = d / np.where(r > 0, r, 1.0)[:, None]
        hit = self._cast_geometry(origins, d)
        return (hit["label"] == LABEL_MESH) & (hit["t"] < r - np.asarray(tol, float))


def _legacy(o3d, V, F):
    m = o3d.geometry.TriangleMesh()
    m.vertices = o3d.utility.Vector3dVector(np.asarray(V, float))
    m.triangles = o3d.utility.Vector3iVector(np.asarray(F, np.int32))
    return m


# ----------------------------------------------------------------------------- rays of an image
def scaled_size(camera, scale: float) -> Tuple[int, int]:
    return max(1, int(round(camera.width * scale))), max(1, int(round(camera.height * scale)))


def pixel_rays(camera, R: np.ndarray, scale: float = 1.0) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Unit rays of the pixel centres of the image at ``scale``: (dirs_cam (H, W, 3), dirs_world (H, W, 3), valid
    (H, W)).  ``R`` is the world-to-camera rotation; a pixel the model cannot undistort is invalid."""
    W, H = scaled_size(camera, scale)
    xs = (np.arange(W) + 0.5) / scale
    ys = (np.arange(H) + 0.5) / scale
    uv = np.stack(np.meshgrid(xs, ys), axis=-1).reshape(-1, 2)
    xy = np.asarray(camera.cam_from_img(uv), float)
    d = np.concatenate([xy, np.ones((len(xy), 1))], axis=1)
    valid = np.all(np.isfinite(d), axis=1)
    d[~valid] = [0, 0, 1]
    d /= np.linalg.norm(d, axis=1)[:, None]
    dw = d @ np.asarray(R, float)                   # R^T d for every row
    return d.reshape(H, W, 3), dw.reshape(H, W, 3), valid.reshape(H, W)


def cast_image(scene: Scene, camera, R: np.ndarray, C: np.ndarray, scale: float = 1.0,
               mask: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
    """The maps of one image: depth, range, normal (camera frame, facing the camera), label, tri.  ``mask``: the
    rover mask at native size (True = terrain); masked pixels get label 3 and no depth."""
    d_cam, d_world, valid = pixel_rays(camera, R, scale)
    H, W = valid.shape
    hit = scene.cast(np.asarray(C, np.float32), d_world.reshape(-1, 3))
    t = hit["t"].reshape(H, W)
    label = hit["label"].reshape(H, W)
    label[~valid] = LABEL_NONE
    nw = hit["normal"].reshape(H, W, 3)
    ok = (label != LABEL_NONE) & (t < 0.5 * SKY_RANGE_M)             # the sky: label far, no depth
    rng = np.where(ok, t, 0.0).astype(np.float32)
    depth = (rng * d_cam[..., 2]).astype(np.float32)
    n_cam = nw @ np.asarray(R, float).T            # R n for every pixel
    flip = np.sum(n_cam * d_cam, axis=-1) > 0     # facing away from the camera: flip
    n_cam[flip] *= -1
    n_cam[~ok] = 0
    if mask is not None:
        m = np.asarray(mask)
        if m.shape != (H, W):
            import cv2
            m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST) > 0
        label[~m] = LABEL_MASKED
        depth[~m] = 0
        rng[~m] = 0
        n_cam[~m] = 0
    return {"depth": depth, "range": rng, "normal": n_cam.astype(np.float32), "label": label.astype(np.uint8),
            "tri": hit["tri"].reshape(H, W).astype(np.int32), "dirs_cam": d_cam.astype(np.float32)}


def save_maps(path: PathLike, maps: Dict[str, np.ndarray], **meta) -> Path:
    """``<image>.npz``: depth float32, range float16, normal int8 (x 127), label uint8, tri int32, and ``meta`` as JSON."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, depth=maps["depth"].astype(np.float32), range=maps["range"].astype(np.float16),
                        normal=np.clip(np.round(maps["normal"] * 127), -127, 127).astype(np.int8),
                        label=maps["label"].astype(np.uint8), tri=maps["tri"].astype(np.int32),
                        meta=np.array(json.dumps(meta, default=float)))
    return p


def load_maps(path: PathLike) -> Dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    out = {"depth": z["depth"].astype(np.float32), "range": z["range"].astype(np.float32),
           "normal": z["normal"].astype(np.float32) / 127.0, "label": z["label"], "tri": z["tri"]}
    out["meta"] = json.loads(str(z["meta"])) if "meta" in z else {}
    return out


PREVIEW_DIR = "photometry_jpg"   # 0.97.0: every per-image picture of a station in one folder
DEPTH_LINEAR_MAX_M = 50.0          # the linear depth picture: 0 to the mesh radius (0.97.0: 50 m, was 30)
DEPTH_LOG_RANGE_M = (0.5, 200.0)  # the log one
JPEG_QUALITY = 92


def depth_rgb(rng: np.ndarray, lab: np.ndarray, mode: str = "log") -> np.ndarray:
    """A range map as a BGR picture: turbo over ``mode`` "log" (DEPTH_LOG_RANGE_M) or "linear" (0 to
    DEPTH_LINEAR_MAX_M); the ground plane the same colours darker, the sky grey, masked dark red, none black."""
    import cv2
    rng = np.asarray(rng, np.float32)
    v = np.zeros_like(rng)
    ok = (lab == LABEL_MESH) | (lab == LABEL_PLANE)
    if mode == "linear":
        v[ok] = np.clip(rng[ok] / DEPTH_LINEAR_MAX_M, 0, 1)
    else:
        a, b = np.log10(DEPTH_LOG_RANGE_M[0]), np.log10(DEPTH_LOG_RANGE_M[1])
        v[ok] = np.clip((np.log10(np.clip(rng[ok], *DEPTH_LOG_RANGE_M)) - a) / (b - a), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    pl = lab == LABEL_PLANE
    img[pl] = (img[pl] * 0.55).astype(np.uint8)
    img[lab == LABEL_FAR] = (160, 160, 160)
    img[lab == LABEL_MASKED] = (40, 40, 120)
    img[lab == LABEL_NONE] = 0
    return img


def preview_jpg(maps: Dict[str, np.ndarray], out_prefix: PathLike) -> List[Path]:
    """0.97.0 (Christian, 12 Nov 2026: "Put the depth maps and surfaces normal map in the photometry_jpg folder ...
    Depth maps should be in two versions: linear and log"): ``<prefix>_depth_linear.jpg``, ``_depth_log.jpg``,
    ``_normal.jpg``."""
    import cv2
    q = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    lab = maps["label"]
    Path(str(out_prefix)).parent.mkdir(parents=True, exist_ok=True)
    outs = []
    for mode in ("linear", "log"):
        f = Path(str(out_prefix) + f"_depth_{mode}.jpg")
        cv2.imwrite(str(f), depth_rgb(maps["range"], lab, mode), q)
        outs.append(f)
    n = maps["normal"]
    nimg = ((n * 0.5 + 0.5) * 255).astype(np.uint8)[..., ::-1].copy()
    nimg[(lab != LABEL_MESH) & (lab != LABEL_PLANE)] = 0
    f = Path(str(out_prefix) + "_normal.jpg")
    cv2.imwrite(str(f), nimg, q)
    return outs + [f]


def preview_png(maps: Dict[str, np.ndarray], out_prefix: PathLike) -> Tuple[Path, Path]:
    """``<prefix>_depth.png`` (turbo over log range; far sphere grey, masked dark red, none black) and ``<prefix>_normal.png``."""
    import cv2
    rng = maps["range"].astype(np.float32)
    lab = maps["label"]
    v = np.zeros_like(rng)
    ok = lab == LABEL_MESH
    ok = ok | (lab == LABEL_PLANE)
    v[ok] = np.clip((np.log10(np.clip(rng[ok], 0.5, 200)) - np.log10(0.5)) / (np.log10(200) - np.log10(0.5)), 0, 1)
    img = cv2.applyColorMap((v * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    pl = lab == LABEL_PLANE
    img[pl] = (img[pl] * 0.55).astype(np.uint8)                        # the ground plane: the same colours, darker
    img[lab == LABEL_FAR] = (160, 160, 160)
    img[lab == LABEL_MASKED] = (40, 40, 120)
    img[lab == LABEL_NONE] = 0
    p1 = Path(str(out_prefix) + "_depth.png")
    p1.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(p1), img)
    n = maps["normal"]
    nimg = ((n * 0.5 + 0.5) * 255).astype(np.uint8)[..., ::-1]
    nimg[(lab != LABEL_MESH) & (lab != LABEL_PLANE)] = 0
    p2 = Path(str(out_prefix) + "_normal.png")
    cv2.imwrite(str(p2), nimg)
    return p1, p2


def coverage(maps: Dict[str, np.ndarray]) -> Dict[str, float]:
    lab = maps["label"]
    n = lab.size
    return {"mesh": float((lab == LABEL_MESH).sum() / n), "plane": float((lab == LABEL_PLANE).sum() / n),
            "far": float((lab == LABEL_FAR).sum() / n),
            "none": float((lab == LABEL_NONE).sum() / n), "masked": float((lab == LABEL_MASKED).sum() / n)}


# ----------------------------------------------------------------------------- validation
def validate_sparse(rec, image_id: int, maps: Dict[str, np.ndarray], scale: float) -> Dict[str, Any]:
    """The image's observed 3-D points against the cast depth at their keypoints: median |dz|/z and the 90th
    percentile per range bin, and the share of keypoints over no surface."""
    im = rec.images[image_id]
    cw = im.cam_from_world()
    R, t = np.asarray(cw.rotation.matrix(), float), np.asarray(cw.translation, float)
    H, W = maps["depth"].shape
    obs: Dict[int, Tuple[list, list]] = {LABEL_MESH: ([], []), LABEL_PLANE: ([], [])}
    none = 0
    for p2 in im.points2D:
        if not p2.has_point3D():
            continue
        X = np.asarray(rec.points3D[p2.point3D_id].xyz, float)
        z = float((R @ X + t)[2])
        u, v = int(p2.xy[0] * scale), int(p2.xy[1] * scale)
        if not (0 <= u < W and 0 <= v < H) or z <= 0:
            continue
        lab = int(maps["label"][v, u])
        if lab not in obs:
            none += 1
            continue
        obs[lab][0].append(z)
        obs[lab][1].append(float(maps["depth"][v, u]))

    def bins(z_obs, z_cast):
        z_obs, z_cast = np.asarray(z_obs), np.asarray(z_cast)
        rows = []
        for lo, hi in RANGE_BINS:
            m = (z_obs >= lo) & (z_obs < hi)
            if m.sum() < 3:
                continue
            rel = np.abs(z_cast[m] - z_obs[m]) / z_obs[m]
            rows.append({"range_m": [lo, hi if hi < 1e8 else None], "points": int(m.sum()), "median_rel_dz": float(np.median(rel)),
                         "p90_rel_dz": float(np.quantile(rel, 0.9)), "median_dz_m": float(np.median(z_cast[m] - z_obs[m]))})
        return rows
    # the tie points on the plane tell how far the flat stand-in is from the terrain it replaces (20-100 m)
    return {"image": im.name, "points_on_mesh": int(len(obs[LABEL_MESH][0])), "points_on_plane": int(len(obs[LABEL_PLANE][0])),
            "points_off_surface": int(none), "bins": bins(*obs[LABEL_MESH]), "plane_bins": bins(*obs[LABEL_PLANE])}


def panorama_range(scene: Scene, pivot: Sequence[float], width: int = 1440, up=(0, 0, 1), north=(0, 1, 0), east=(1, 0, 0)) -> Dict[str, np.ndarray]:
    """An equirect range map from the pivot (azimuth from north through east; the mesh and the far sphere)."""
    H = width // 2
    az = (np.arange(width) + 0.5) / width * 2 * np.pi
    el = np.pi / 2 - (np.arange(H) + 0.5) / H * np.pi
    A, E = np.meshgrid(az, el)
    e, n, u = (np.asarray(v, float) for v in (east, north, up))
    d = (np.cos(E) * np.sin(A))[..., None] * e + (np.cos(E) * np.cos(A))[..., None] * n + np.sin(E)[..., None] * u
    hit = scene.cast(np.asarray(pivot, np.float32), d.reshape(-1, 3))
    r = hit["t"].reshape(H, width)
    lab = hit["label"].reshape(H, width)
    r[lab == LABEL_NONE] = 0
    r[r >= 0.5 * SKY_RANGE_M] = np.inf
    return {"range": r.astype(np.float32), "label": lab}


# ----------------------------------------------------------------------------- a station's image set
def cast_station(project, rec, station, scene: Scene, out_dir: PathLike, families: str = "N", scale: float = 1.0,
                 zcam_scale: float = 1.0, images: Optional[Sequence[str]] = None, previews: bool = True,
                 validate: bool = True, use_masks: bool = True) -> Dict[str, Any]:
    """The maps of every registered image of ``station`` (``families`` letters; or the ``images`` named) at
    ``scale`` (Mastcam-Z at ``zcam_scale``), the previews and the validation -> ``out_dir/maps``, ``photometry_jpg``,
    ``maps_report.json``.  Returns the report."""
    out = Path(out_dir)
    by_name = {im.name: (iid, im) for iid, im in rec.images.items()}
    names = list(images) if images else [n for n in station.images if n[:1].upper() in set(families.upper())]
    rows = []
    for name in names:
        if name not in by_name:
            rows.append({"image": name, "error": "not in the model"})
            continue
        iid, im = by_name[name]
        if not im.has_pose:
            rows.append({"image": name, "error": "no pose"})
            continue
        cam = rec.cameras[im.camera_id]
        cw = im.cam_from_world()
        R = np.asarray(cw.rotation.matrix(), float)
        C = np.asarray(im.projection_center(), float)
        s = zcam_scale if name[:1].upper() == "Z" else scale
        mask = None
        mf = Path(project.root) / "masks" / (name + ".png")
        if use_masks and mf.is_file():
            import cv2
            mask = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE) > 0
        maps = cast_image(scene, cam, R, C, scale=s, mask=mask)
        stem = Path(name).stem
        save_maps(out / "maps" / f"{stem}.npz", maps, image=name, scale=s, camera_model=cam.model.name,
                  width=maps["depth"].shape[1], height=maps["depth"].shape[0], station=station.label)
        if previews:                                 # 0.97.0: jpg, with the photometry pictures
            preview_jpg(maps, out / PREVIEW_DIR / stem)
        row = {"image": name, "scale": s, "coverage": coverage(maps)}
        if validate:
            row["sparse"] = validate_sparse(rec, iid, maps, s)
        rows.append(row)
        log.info("depthmaps: %s %.0f%% mesh, %.0f%% plane, %.0f%% far", name, 100 * row["coverage"]["mesh"],
                 100 * row["coverage"]["plane"], 100 * row["coverage"]["far"])
    rep = {"station": station.label, "n_images": len(rows), "scale": scale, "zcam_scale": zcam_scale, "images": rows,
           "summary": _summary(rows), "summary_plane": _summary(rows, "plane_bins")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "maps_report.json").write_text(json.dumps(rep, indent=1, default=float), encoding="utf-8")
    (out / "maps_report.md").write_text(report_md(rep), encoding="utf-8")
    return rep


def _summary(rows, which: str = "bins") -> Dict[str, Any]:
    fam: Dict[str, Dict[str, list]] = {}
    for r in rows:
        if "sparse" not in r:
            continue
        f = "Z" if r["image"][:1].upper() == "Z" else r["image"][:1].upper()
        for b in r["sparse"].get(which, []):
            key = f"{b['range_m'][0]}-{b['range_m'][1] or 'inf'}"
            fam.setdefault(f, {}).setdefault(key, []).append((b["median_rel_dz"], b["median_dz_m"]))
    return {f: {k: {"images": len(v), "median_of_medians_rel_dz": float(np.median([a for a, _ in v])),
                    "median_of_medians_dz_m": float(np.median([b for _, b in v]))} for k, v in d.items()} for f, d in fam.items()}


def report_md(rep: Dict[str, Any]) -> str:
    lines = [f"# Depth maps of station {rep['station']}", "", f"{rep['n_images']} images; Navcam scale {rep['scale']}, "
             f"Mastcam-Z scale {rep['zcam_scale']}.", "", "| image | mesh | plane | far | none | masked | points on mesh | median |dz|/z (0-5, 5-10, 10-20, 20-60 m) | points on plane |",
             "|---|---:|---:|---:|---:|---:|---:|---|---:|"]
    for r in rep["images"]:
        if "error" in r:
            lines.append(f"| {r['image']} | | | | | | | {r['error']} | |")
            continue
        c = r["coverage"]
        sp = r.get("sparse") or {}
        bins = {f"{b['range_m'][0]}": b["median_rel_dz"] for b in sp.get("bins", [])}
        txt = ", ".join(f"{bins[k]:.3f}" if k in bins else "-" for k in ("0", "5", "10", "20"))
        lines.append(f"| {r['image']} | {c['mesh']:.2f} | {c.get('plane', 0):.2f} | {c['far']:.2f} | {c['none']:.2f} | {c['masked']:.2f} | "
                     f"{sp.get('points_on_mesh', '-')} | {txt} | {sp.get('points_on_plane', '-')} |")
    if rep.get("summary"):
        lines += ["", "## Sparse points against the cast depth, per camera family (median over the images of the per-image median |dz|/z)", ""]
        for f, d in rep["summary"].items():
            lines.append(f"- {f}: " + "; ".join(f"{k} m: {v['median_of_medians_rel_dz']:.3f} ({v['images']} images)" for k, v in d.items()))
    if rep.get("summary_plane"):
        lines += ["", "## Sparse points on the ground plane (how far the flat stand-in is from the terrain: median |dz|/z, "
                  "and the median signed dz, cast - observed; negative = the plane is nearer than the terrain)", ""]
        for f, d in rep["summary_plane"].items():
            lines.append(f"- {f}: " + "; ".join(f"{k} m: {v['median_of_medians_rel_dz']:.3f}, dz {v['median_of_medians_dz_m']:+.2f} m "
                                                 f"({v['images']} images)" for k, v in d.items()))
    return "\n".join(lines) + "\n"

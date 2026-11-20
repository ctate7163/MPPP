"""
Fusion and meshing variants, made and scored apart from the station products (1.0, Christian, 20 Nov 2026: "how could
we implement these meshing options without the danger of making anything worse?").

``rpp fusion-compare STATION`` takes a station of the station list whose block cloud is made (``rpp single-sites`` /
``rpv_process.bat``), and for each variant

1. fuses the block's existing depth maps into ``<block workspace>/compare/<variant>/fused.ply`` (no stereo, no GPU),
2. meshes that cloud for the station with the very arguments the station run uses (:func:`rpp.ibr.single_sites.plan`),
   only ``--points`` and ``--out`` changed: ``compare/<variant>/stations/<station>/`` (mesh.ply, its preview),
3. scores the mesh against the block's tie points (:func:`score_mesh`),

then compares every variant with ``current`` - the station run's own cloud, meshed the same way - and says for each
whether it is not worse, better, or worse (:func:`verdict`).  Nothing of the station products, the workspace's maps
or its fused.ply is changed; a variant becomes the default only by a later change of the code, once the comparison
on real stations is in.

THE VARIANTS (:data:`VARIANTS`)

* ``current``: the workspace's ``fused.ply``.
* ``two_pass``: the loose fusion and a strict one, merged (:func:`rpp.sfm.dense.merge_fusions`).
* ``filtered``: the depth maps filtered first (:mod:`rpp.sfm.depth_filter`: depth edges, speckles, back-facing
  normals), then the workspace's own fusion.
* ``filtered_two_pass``: both.
* ``delaunay`` / ``filtered_delaunay`` (with ``--colmap EXE``): COLMAP's Delaunay mesher, which uses which cameras saw
  each point (``fused.ply.vis``), over the ``current`` / ``filtered`` cloud.

THE SCORE

The block's tie points - points of the sparse model seen in at least ``min_track`` images with a reprojection error
of at most ``max_error_px`` - are triangulated from many views and are not part of the dense matching, so they are an
independent sample of the surface.  Per ring of horizontal distance from the mast axis: the share of tie points with
measured terrain (the mesh's ``terrain_triangles``, not the hole fill, the ground plane or the rover) within the
depth tolerance tau(range) (:func:`rpp.ibr.mesh.tau_m`) - this counts both holes and errors - and, of the points
within 3 tau, the median and 90th percentile distance.  A variant is "not worse" when no ring with enough tie points
loses more than 1 point of the agreeing share or grows its 90th percentile by more than 10 %.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

log = logging.getLogger(__name__)
PathLike = Union[str, os.PathLike]

COMPARE_DIR = "compare"
VARIANTS = ("current", "two_pass", "filtered", "filtered_two_pass")
DELAUNAY_VARIANTS = ("delaunay", "filtered_delaunay")
RINGS_M = (0.0, 2.0, 5.0, 10.0, 20.0, 40.0, 75.0)
MIN_RING_POINTS = 30
WORSE_AGREE = 0.01            # a ring losing more than this share of agreeing tie points is worse
WORSE_P90 = 0.10              # ... or a 90th percentile more than 10 % (and 5 mm) larger
BETTER_AGREE = 0.02
BETTER_P90 = 0.10


# ------------------------------------------------------------------------------------------- variant workspaces
def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)                                         # NTFS: a hard link costs no space
    except OSError:
        shutil.copyfile(src, dst)


def link_tree(src: Path, dst: Path, pattern: str = "*") -> int:
    n = 0
    for f in Path(src).rglob(pattern):
        if f.is_file():
            _link(f, Path(dst) / f.relative_to(src))
            n += 1
    return n


def variant_workspace(ws: PathLike, vws: PathLike, maps_from: Optional[PathLike] = None) -> Path:
    """A dense workspace for a variant's fusion: the images and normal maps of ``ws`` linked, its sparse model and
    configuration files copied, and the depth maps of ``maps_from`` (a folder holding ``stereo/depth_maps``; default
    ``ws`` itself) linked.  ``ws`` is not changed."""
    ws, vws = Path(ws), Path(vws)
    (vws / "stereo").mkdir(parents=True, exist_ok=True)
    if not (vws / "images").is_dir():
        link_tree(ws / "images", vws / "images")
    if (vws / "sparse").is_dir():
        shutil.rmtree(vws / "sparse")
    shutil.copytree(ws / "sparse", vws / "sparse")
    for f in ("patch-match.cfg", "fusion.cfg"):
        if (ws / "stereo" / f).is_file():
            shutil.copyfile(ws / "stereo" / f, vws / "stereo" / f)
    src = Path(maps_from) if maps_from else ws
    if src.resolve() != vws.resolve():
        link_tree(src / "stereo" / "depth_maps", vws / "stereo" / "depth_maps", "*.geometric.bin")
    nm = vws / "stereo" / "normal_maps"
    if not (nm.is_dir() and any(nm.glob("*.geometric.bin"))):
        link_tree(ws / "stereo" / "normal_maps", nm, "*.geometric.bin")
    return vws


def product_fusion(ws: PathLike) -> Dict[str, Any]:
    """The fusion settings of the station run's cloud: its options and working size (``rpp_fusion.json``) and the
    masks it used (``mppp_dense_run.json``; else ``masks_fusion/`` or ``masks/``)."""
    ws = Path(ws)
    out: Dict[str, Any] = {"fusion_options": {}, "max_image_size": 2000, "mask_path": None}
    try:
        rec = json.loads((ws / "rpp_fusion.json").read_text(encoding="utf-8"))
        out["fusion_options"] = dict(rec.get("fusion_options") or {})
        out["max_image_size"] = int(rec.get("max_image_size") or 2000)
    except (OSError, ValueError):
        from .dense import STATION_FUSION_OPTIONS
        out["fusion_options"] = dict(STATION_FUSION_OPTIONS)
    try:
        run = json.loads((ws / "mppp_dense_run.json").read_text(encoding="utf-8"))
        out["mask_path"] = ((run.get("steps") or {}).get("fusion") or {}).get("masks")
    except (OSError, ValueError):
        pass
    if not out["mask_path"] or not Path(out["mask_path"]).is_dir():
        from .dense import SKY_MASK_DIR
        for d in (ws / SKY_MASK_DIR, ws / "masks"):
            if d.is_dir():
                out["mask_path"] = str(d)
                break
    return out


def fuse(vws: PathLike, settings: Dict[str, Any], two_pass: bool = False) -> Dict[str, Any]:
    """The fusion of a variant workspace with the station run's settings (``fused.ply`` + ``.vis``); ``two_pass``: the
    strict fusion too, and the merge."""
    import pycolmap
    from .dense import FUSED_STRICT_PLY, STATION_FUSION_STRICT, merge_fusions, set_options
    vws = Path(vws)
    rep: Dict[str, Any] = {}
    passes = [("fused.ply", settings["fusion_options"])]
    if two_pass:
        passes.append((FUSED_STRICT_PLY, {**settings["fusion_options"], **STATION_FUSION_STRICT}))
    for ply, opts in passes:
        t0 = time.time()
        fo = pycolmap.StereoFusionOptions()
        if settings.get("mask_path"):
            fo.mask_path = str(settings["mask_path"])
        fo.max_image_size = int(settings["max_image_size"])
        set_options(fo, opts)
        r = pycolmap.stereo_fusion(str(vws / ply), str(vws), "COLMAP", input_type="geometric", options=fo, output_type="ply")
        rep[ply] = {"points": int(r.num_points3D()), "seconds": round(time.time() - t0, 1), "options": opts}
    if two_pass:
        rep["merge"] = merge_fusions(vws)
    return rep


# ------------------------------------------------------------------------------------------- scoring
def tie_points(sparse: PathLike, min_track: int = 3, max_error_px: float = 1.0) -> np.ndarray:
    import pycolmap
    rec = pycolmap.Reconstruction(str(sparse))
    P = [p.xyz for p in rec.points3D.values() if p.track.length() >= min_track and p.error <= max_error_px]
    return np.asarray(P, float).reshape(-1, 3)


def score_mesh(mesh_dir: PathLike, points: np.ndarray, rings: Sequence[float] = RINGS_M) -> Dict[str, Any]:
    """The agreement of a station mesh (``mesh.ply`` + ``mesh_report.json`` in ``mesh_dir``) with the reference
    ``points`` (the tie points), per ring of horizontal distance from the pivot (module docstring)."""
    import open3d as o3d
    from ..ibr.mesh import horizontal_distance, read_mesh, tau_m
    mesh_dir = Path(mesh_dir)
    rep = json.loads((mesh_dir / "mesh_report.json").read_text(encoding="utf-8"))
    V, F = read_mesh(mesh_dir / "mesh.ply")
    tt = rep.get("terrain_triangles")
    if tt and 0 <= int(tt[0]) < int(tt[1]) <= len(F):
        F = F[int(tt[0]):int(tt[1])]
    pivot = np.asarray(rep["pivot_block_m"], float)
    up = (0, 0, 1)
    h = horizontal_distance(points, pivot, up)
    sel = h < rings[-1]
    P, h = points[sel], h[sel]
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.core.Tensor(V.astype(np.float32)), o3d.core.Tensor(F.astype(np.uint32)))
    d = scene.compute_distance(o3d.core.Tensor(P.astype(np.float32))).numpy().astype(float) if len(P) else np.zeros(0)
    rng = np.linalg.norm(P - pivot, axis=1)
    tau = tau_m(rng)
    out = []
    for a, b in zip(rings[:-1], rings[1:]):
        m = (h >= a) & (h < b)
        n = int(m.sum())
        row: Dict[str, Any] = {"ring_m": [a, b], "tie_points": n}
        if n:
            dm, tm = d[m], tau[m]
            near = dm <= 3 * tm
            row["agree"] = round(float(np.mean(dm <= tm)), 4)
            row["median_m"] = round(float(np.median(dm[near])), 4) if near.any() else None
            row["p90_m"] = round(float(np.percentile(dm[near], 90)), 4) if near.any() else None
        out.append(row)
    return {"rings": out, "tie_points": int(len(P)), "triangles": int(len(F)),
            "agree_all": round(float(np.mean(d <= tau)), 4) if len(P) else None,
            "points_meshed": rep.get("points_meshed"), "surface": (rep.get("surface") or {}).get("method")}


def verdict(base: Dict[str, Any], var: Dict[str, Any], min_points: int = MIN_RING_POINTS) -> Dict[str, Any]:
    """"worse", "not worse" or "better" than ``base`` (both :func:`score_mesh`), and why, ring by ring."""
    worse, better = [], []
    for rb, rv in zip(base["rings"], var["rings"]):
        if rb.get("tie_points", 0) < min_points or "agree" not in rb or "agree" not in rv:
            continue
        ring = f"{rb['ring_m'][0]:g}-{rb['ring_m'][1]:g} m"
        da = rv["agree"] - rb["agree"]
        if da < -WORSE_AGREE:
            worse.append(f"{ring}: agreeing share {rb['agree']:.3f} -> {rv['agree']:.3f}")
        elif da >= BETTER_AGREE:
            better.append(f"{ring}: agreeing share {rb['agree']:.3f} -> {rv['agree']:.3f}")
        pb, pv = rb.get("p90_m"), rv.get("p90_m")
        if pb is not None and pv is not None:
            if pv > pb * (1 + WORSE_P90) + 0.005:
                worse.append(f"{ring}: p90 {pb:.3f} -> {pv:.3f} m")
            elif pv < pb * (1 - BETTER_P90):
                better.append(f"{ring}: p90 {pb:.3f} -> {pv:.3f} m")
    return {"verdict": "worse" if worse else ("better" if better else "not worse"), "worse": worse, "better": better}


# ------------------------------------------------------------------------------------------- the run
def _mesh_argv(label: str, root: PathLike, cfg: Dict[str, Any], block_cameras: Optional[str], terrain: Optional[str]) -> List[str]:
    from ..ibr.single_sites import plan
    runs = plan(cfg, root, labels=[label], steps=("mesh",), variants=["masked"], block_cameras=block_cameras, terrain=terrain)
    if not runs:
        raise KeyError(f"{label}: not a station of the list")
    for c in runs[0]["commands"]:
        if c["step"] == "mesh" and c["variant"] == "masked":
            return [str(x) for x in c["argv"]]
    raise KeyError(f"{label}: no mesh step")


def _replace(argv: List[str], flag: str, value: Optional[str]) -> List[str]:
    out, skip = [], False
    for i, x in enumerate(argv):
        if skip:
            skip = False
            continue
        if x == flag:
            skip = True
            continue
        out.append(x)
    return out + ([flag, value] if value is not None else [])


def compare_station(label: str, root: PathLike, cfg: Optional[Dict[str, Any]] = None, variants: Sequence[str] = VARIANTS,
                    colmap_exe: Optional[str] = None, block_cameras: Optional[str] = "NF", terrain: Optional[str] = "block",
                    keep_maps: bool = False, filter_options: Optional[Dict[str, Any]] = None,
                    python: Optional[str] = None, refuse: bool = False) -> Dict[str, Any]:
    """The variants of one station (module docstring); returns and writes ``compare/<station>_comparison.json`` and
    ``.md``.  ``refuse``: fuse the variants again even when their cloud is there with the same settings."""
    from ..ibr.single_sites import load_config
    from .depth_filter import FILTER_DEFAULTS
    cfg = cfg or load_config()
    argv = _mesh_argv(label, root, cfg, block_cameras, terrain)
    pts_i = argv.index("--points") if "--points" in argv else -1
    if pts_i < 0:
        raise ValueError(f"{label}: its mesh step names no block cloud (--points); a station of a block terrain is needed")
    ws = Path(argv[pts_i + 1]).parent
    if not (ws / "fused.ply").is_file() or not (ws / "stereo" / "depth_maps").is_dir():
        raise FileNotFoundError(f"{ws}: no fused.ply or depth maps yet - run the station first")
    variants = [v for v in variants if v in VARIANTS + DELAUNAY_VARIANTS]
    if not colmap_exe:
        variants = [v for v in variants if v not in DELAUNAY_VARIANTS]
    variants = ["current"] + [v for v in dict.fromkeys(variants) if v != "current"]     # the reference first
    cmp_dir = ws / COMPARE_DIR
    cmp_dir.mkdir(exist_ok=True)
    settings = product_fusion(ws)
    fopts = {**FILTER_DEFAULTS, **(filter_options or {})}
    report: Dict[str, Any] = {"station": label, "workspace": str(ws), "fusion": settings, "filter": fopts,
                              "variants": {}, "made": time.strftime("%Y-%m-%d %H:%M")}
    # the block's lock (0.99.53's station runs: one cloud maker per block) while the maps and fused.ply are read - a
    # 1.0 station run of this block waits; a 0.99.52 run takes no lock, so do not compare a block it is making
    from ..ibr.single_sites import BLOCK_LOCK, FileLock
    lock = FileLock(ws.parent / BLOCK_LOCK)
    lock.acquire(what=f"the block cloud of {ws.parent.parent.name}")
    try:
        return _compare(label, ws, cmp_dir, argv, variants, settings, fopts, report, colmap_exe, keep_maps, python,
                        refuse, lock)
    finally:
        lock.release()


def _compare(label, ws, cmp_dir, argv, variants, settings, fopts, report, colmap_exe, keep_maps, python, refuse, lock):
    from .depth_filter import filter_workspace
    clouds: Dict[str, Path] = {"current": ws}
    fdir = cmp_dir / "filtered"
    todo = {}                                                    # the variants whose cloud is to be fused
    for v in variants:
        if v in ("current",) + DELAUNAY_VARIANTS:
            continue
        want = {"settings": settings, "two_pass": v.endswith("two_pass"), "filter": fopts if v.startswith("filtered") else None}
        rec_f = cmp_dir / v / "rpp_compare_fusion.json"
        have = json.loads(rec_f.read_text(encoding="utf-8")).get("want") if rec_f.is_file() else None
        if refuse or have != json.loads(json.dumps(want, default=str)) or not (cmp_dir / v / "fused.ply").is_file():
            todo[v] = want
        clouds[v] = cmp_dir / v
    if any(v.startswith("filtered") for v in todo):
        frec = fdir / "rpp_depth_filter.json"
        old = json.loads(frec.read_text(encoding="utf-8")).get("options") if frec.is_file() else None
        maps = fdir / "stereo" / "depth_maps"
        if refuse or old != json.loads(json.dumps(fopts)) or not (maps.is_dir() and any(maps.glob("*.bin"))):
            log.info("fusion-compare: filtering the depth maps of %s", ws)
            r = filter_workspace(ws, fdir, **fopts)
            report["filter_result"] = {k: r[k] for k in ("images", "totals", "kept_fraction", "seconds")}
    if "filter_result" not in report and (fdir / "rpp_depth_filter.json").is_file():
        report["filter_result"] = {k: v for k, v in json.loads((fdir / "rpp_depth_filter.json").read_text(encoding="utf-8")).items()
                                   if k != "per_image"}
    for v, want in todo.items():
        vws = variant_workspace(ws, cmp_dir / v, fdir if v.startswith("filtered") else ws)
        log.info("fusion-compare: fusing %s", v)
        fr = fuse(vws, settings, two_pass=want["two_pass"])
        (vws / "rpp_compare_fusion.json").write_text(json.dumps({"want": want, "result": fr}, indent=1, default=str), encoding="utf-8")
    if "filtered_delaunay" in variants:
        clouds["filtered_delaunay"] = clouds.get("filtered") or cmp_dir / "filtered"
    if "delaunay" in variants:
        clouds["delaunay"] = ws
    py = python or sys.executable
    tie = tie_points(ws / "sparse")
    for v in variants:
        out = cmp_dir / v / "stations" / label
        a = _replace(_replace(list(argv), "--points", str(clouds[v] / "fused.ply")), "--out", str(out))
        if v in DELAUNAY_VARIANTS:
            a = _replace(_replace(a, "--colmap", colmap_exe), "--dense-dir", str(clouds[v]))
        t0 = time.time()
        log.info("fusion-compare: meshing %s", v)
        r = subprocess.run([py, "-m", "rpp", *a], capture_output=True, text=True)
        if r.returncode != 0:
            report["variants"][v] = {"error": (r.stderr or r.stdout)[-800:]}
            continue
        report["variants"][v] = {"mesh_dir": str(out), "seconds": round(time.time() - t0, 1), **score_mesh(out, tie)}
        if v == "current":                                       # the block's own cloud read: its lock let go
            lock.release()
    base = report["variants"].get("current")
    if base and "rings" in base:
        for v, s in report["variants"].items():
            if v != "current" and "rings" in s:
                s.update(verdict(base, s))
    if not keep_maps:
        drop_maps(cmp_dir)
    (cmp_dir / f"{label}_comparison.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (cmp_dir / f"{label}_comparison.md").write_text(markdown(report), encoding="utf-8")
    return report


def drop_maps(cmp_dir: PathLike) -> int:
    """The depth and normal maps of the variant workspaces deleted (the filtered ones, a few GB a block; the others are
    hard links, which deleting does not touch the workspace's own maps through).  The clouds and meshes stay."""
    n = 0
    for f in Path(cmp_dir).glob("*/stereo/*/*.bin"):
        f.unlink()
        n += 1
    return n


def markdown(report: Dict[str, Any]) -> str:
    """The comparison as a table: per variant and ring, the agreeing share and the 90th percentile."""
    vs = [v for v, s in report["variants"].items() if "rings" in s]
    lines = [f"# Fusion and mesh variants: {report['station']}", "",
             f"Workspace `{report['workspace']}`, {report.get('made', '')}. Tie points: track >= 3, error <= 1 px.", "",
             "Agreeing share (tie points within tau of the measured terrain) / 90th percentile distance (m):", "",
             "| ring | tie points | " + " | ".join(vs) + " |", "|---|---|" + "---|" * len(vs)]
    if vs:
        for i, rb in enumerate(report["variants"][vs[0]]["rings"]):
            cells = []
            for v in vs:
                r = report["variants"][v]["rings"][i]
                cells.append(f"{r['agree']:.3f} / {r['p90_m']:.3f}" if r.get("agree") is not None and r.get("p90_m") is not None else "-")
            lines.append(f"| {rb['ring_m'][0]:g}-{rb['ring_m'][1]:g} m | {rb['tie_points']} | " + " | ".join(cells) + " |")
    lines += ["", "| variant | verdict | triangles | why |", "|---|---|---|---|"]
    for v, s in report["variants"].items():
        if "error" in s:
            lines.append(f"| {v} | failed | - | {s['error'][-120:].strip()} |")
            continue
        why = "; ".join(s.get("worse") or s.get("better") or []) or ("the reference" if v == "current" else "")
        lines.append(f"| {v} | {s.get('verdict', 'reference')} | {s.get('triangles')} | {why} |")
    fr = report.get("filter_result") or {}
    if fr.get("totals"):
        t = fr["totals"]
        lines += ["", f"Depth filter: kept {fr.get('kept_fraction')} of {t['valid']} pixels (edges {t['edges']}, "
                  f"speckles {t['speckles']}, back-facing {t['back_facing']}, beyond range {t['beyond_range']})."]
    return "\n".join(lines) + "\n"


def main(a) -> int:
    from ..ibr.single_sites import load_config
    root = Path(a.root)
    cfg = load_config(a.config) if getattr(a, "config", None) else load_config()
    fo: Dict[str, Any] = {}
    if a.edge_rel is not None:
        fo["edge_rel"] = a.edge_rel
    if a.speckle_px is not None:
        fo["speckle_px"] = a.speckle_px
    if a.max_range:
        fo["max_range_m"] = {k.upper(): float(v) for k, v in (x.split("=", 1) for x in a.max_range)}
    dirs = set()
    for lab in a.stations:                                       # the block's maps kept until its last station
        rep = compare_station(lab, root, cfg, variants=a.variants or VARIANTS, colmap_exe=a.colmap,
                              block_cameras=a.block_cameras, terrain=a.terrain, keep_maps=True,
                              filter_options=fo, refuse=a.refuse)
        dirs.add(Path(rep["workspace"]) / COMPARE_DIR)
        print(markdown(rep))
    if not a.keep_maps:
        for d in dirs:
            drop_maps(d)
    return 0

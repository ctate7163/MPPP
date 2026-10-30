"""
MPPP -> Agisoft Metashape: build the project of a block from ``metashape_job.json`` (written by ``mppp metashape``,
:mod:`mppp.sfm.metashape`).  Runs INSIDE Metashape Professional (its Python, 2.x; no MPPP or numpy needed):

    metashape.exe -r build_metashape_project.py metashape_job.json            (the window stays open)
    metashape.exe -platform offscreen -r build_metashape_project.py metashape_job.json --quit

(``make_metashape_project.bat`` beside the job does this.)  Steps:

1. a chunk in a local metric coordinate system (MPPP's world frame of the block; the chunk transform adds the block
   offset), the photos linked from the COLMAP project's ``images/`` (no reference, XMP or EXIF GPS read: the EXIF
   GPS of the PNGs is Mars latitude/longitude, which Metashape would take for WGS84);
2. COLMAP's tie points through the Bundler import (provisional poses);
3. one sensor per COLMAP camera with the converted calibration, fixed; every camera's pose set from the job (world
   from camera; Metashape's camera axes are COLMAP's: x right, y down, z forward); the camera references (the PDS
   label priors: position with its sigma, yaw/pitch/roll); the tie points triangulated again with these cameras
   from COLMAP's observations (the Bundler import triangulates with pinhole cameras and renumbers its tracks);
4. the masks from the COLMAP project's ``masks/`` (black = masked in both);
5. the region (the tie points' 2-98 % box);
6. the check: sampled COLMAP observations reprojected by Metashape (pose + calibration + pixel convention) and the
   imported tie points' own reprojection errors (the Bundler conversion) -> ``metashape_check.json``;
7. the project saved as ``<block>.psx`` (before the check is logged).  Nothing is aligned or optimised.
"""
import json
import math
import os
import sys
import time

try:
    import Metashape as M
except ImportError:                                        # imported by MPPP's tests for the helpers below
    M = None


def log(*a):
    print("[mppp metashape]", *a)
    sys.stdout.flush()


def _try(calls):
    """The first of ``calls`` (callables) that does not raise; returns (result, None) or (None, last error)."""
    err = None
    for c in calls:
        try:
            return c(), None
        except Exception as e:                             # noqa: BLE001 - API names differ between versions
            err = e
    return None, err


def _norm(p):
    return os.path.normcase(os.path.abspath(p))


def median(xs):
    xs = sorted(xs)
    n = len(xs)
    if not n:
        return None
    return xs[n // 2] if n % 2 else 0.5 * (xs[n // 2 - 1] + xs[n // 2])


def stats(du, dv):
    e = [math.hypot(a, b) for a, b in zip(du, dv)]
    if not e:
        return {"n": 0}
    return {"n": len(e), "median_px": median(e), "rms_px": math.sqrt(sum(x * x for x in e) / len(e)),
            "max_px": max(e), "mean_du_px": sum(du) / len(du), "mean_dv_px": sum(dv) / len(dv)}


def add_photos(chunk, paths):
    kw_full = dict(load_reference=False, load_xmp_calibration=False, load_xmp_orientation=False,
                   load_xmp_accuracy=False, load_xmp_antenna=False, load_rpc_txt=False)
    kw_min = dict(load_reference=False, load_xmp_calibration=False, load_xmp_orientation=False)
    _, err = _try([lambda: chunk.addPhotos(paths, **kw_full), lambda: chunk.addPhotos(paths, **kw_min),
                   lambda: chunk.addPhotos(paths, load_reference=False)])
    if err is not None:
        raise RuntimeError(f"addPhotos: {err}")


def import_bundler(chunk, job):
    fmt = M.CamerasFormatBundler
    _, err = _try([lambda: chunk.importCameras(path=job["bundler"], format=fmt, load_image_list=True,
                                               image_list=job["bundler_images"]),
                   lambda: chunk.importCameras(job["bundler"], format=fmt, image_list=job["bundler_images"]),
                   lambda: chunk.importCameras(job["bundler"], format=fmt)])
    return err


def tie_points(chunk):
    return getattr(chunk, "tie_points", None) or getattr(chunk, "point_cloud", None)


def make_sensor(chunk, s):
    sen = chunk.addSensor()
    sen.label = s["label"]
    typ = M.Sensor.Type.Fisheye if s["type"] == "fisheye" else M.Sensor.Type.Frame
    sen.type = typ
    sen.width, sen.height = int(s["width"]), int(s["height"])
    cal = M.Calibration()
    cal.width, cal.height = int(s["width"]), int(s["height"])
    _try([lambda: setattr(cal, "type", typ)])
    for k in ("f", "cx", "cy", "b1", "b2", "k1", "k2", "k3", "k4", "p1", "p2", "p3", "p4"):
        if k in s:
            setattr(cal, k, float(s[k]))
    sen.user_calib = cal
    _try([lambda: setattr(sen, "calibration", cal)])
    _try([lambda: setattr(sen, "fixed_calibration", True), lambda: setattr(sen, "fixed", True)])
    return sen


def set_mask(chunk, cam, path):
    def by_mask():
        m = M.Mask()
        m.load(path)
        cam.mask = m

    def by_generate():
        chunk.generateMasks(path=path, masking_mode=M.MaskingModeFile, mask_operation=M.MaskOperationReplacement,
                            cameras=[cam])

    def by_import():
        chunk.importMasks(path=path, source=M.MaskSourceFile, operation=M.MaskOperationReplacement, cameras=[cam])
    _, err = _try([by_mask, by_generate, by_import])
    return err is None


def read_bundler_points(path):
    """The 3-D points of a Bundler v0.3 file, in its order (Metashape's track ids follow it)."""
    with open(path, encoding="ascii") as f:
        f.readline()
        nc, npt = (int(v) for v in f.readline().split())
        for _ in range(5 * nc):
            f.readline()
        pts = []
        for _ in range(npt):
            pts.append([float(v) for v in f.readline().split()])
            f.readline()
            f.readline()
    return pts


def set_tie_coords(chunk, pts):
    """30 Oct (Chal Rocks): the Bundler import triangulates its own points with Bundler's pinhole cameras - 8 m off,
    681 px reprojection error after the real cameras were set - so the points get COLMAP's coordinates (internal
    = MPPP's world).  Returns the number set (0: not writable)."""
    tp = tie_points(chunk)
    if tp is None:
        return 0
    points = tp.points
    n = 0
    for i in range(len(points)):
        t = points[i].track_id
        if 0 <= t < len(pts):
            x, y, z = pts[t]
            points[i].coord = M.Vector([x, y, z, 1.0])
            n += 1
    if n:                                                  # read back: an API that ignores the write
        t = tp.points[0].track_id
        c = tp.points[0].coord
        if 0 <= t < len(pts) and abs(c.x / c.w - pts[t][0]) > 1e-3:
            return 0
    return n


def ypr_of(R_world_from_cam):
    """Metashape's yaw, pitch, roll of a camera (world from camera, 3 x 3 rows; the chunk's local frame is the
    identity in a local coordinate system) - the recipe of Metashape's Reference pane."""
    rows = []
    for r in R_world_from_cam:
        v = [float(r[0]), -float(r[1]), -float(r[2])]           # x diag(1, -1, -1): Metashape's photo axes
        n = math.sqrt(sum(x * x for x in v)) or 1.0
        rows.append([x / n for x in v])
    y, p, r = M.Utils.mat2ypr(M.Matrix(rows))
    return [y, p, r]


def set_reference(chunk, cam, ref):
    """The camera's reference: the label prior's position (+ sigma) and attitude (yaw, pitch, roll by Metashape's
    own conversion of the prior rotation; MPPP's formula when that fails)."""
    rr = cam.reference
    rr.location = M.Vector(ref["location"])
    _try([lambda: setattr(rr, "location_accuracy", M.Vector(ref["location_accuracy_m"]))])
    ypr = None
    if ref.get("R_w2c"):
        R = ref["R_w2c"]
        ypr, _ = _try([lambda: ypr_of([[R[0][i], R[1][i], R[2][i]] for i in range(3)])])
    ypr = ypr or ref.get("ypr_deg")
    if ypr:
        rr.rotation = M.Vector(ypr)
        a = float(ref.get("rotation_accuracy_deg") or 2.0)
        _try([lambda: setattr(rr, "rotation_accuracy", M.Vector([a, a, a]))])
    _try([lambda: setattr(rr, "enabled", True)])
    _try([lambda: setattr(rr, "location_enabled", True)])
    _try([lambda: setattr(rr, "rotation_enabled", bool(ypr))])
    return bool(ypr)


def angle_diff(a, b):
    return (a - b + 180.0) % 360.0 - 180.0


def check(chunk, job, cams_by_name):
    """COLMAP's observations reprojected by Metashape, and the imported tie points' reprojection errors."""
    du, dv, miss = [], [], 0
    for name, X, Y, Z, u, v in job.get("check") or []:
        cam = cams_by_name.get(name)
        p = cam.project(M.Vector([X, Y, Z])) if cam is not None else None
        if p is None:
            miss += 1
            continue
        du.append(p.x - u)
        dv.append(p.y - v)
    out = {"colmap_observations": dict(stats(du, dv), not_projected=miss)}
    d = []                                                 # MPPP's yaw/pitch/roll formula against Metashape's
    for im in job["images"]:
        ref = im.get("reference") or {}
        cam = cams_by_name.get(im["name"])
        if not ref.get("ypr_refined_deg") or cam is None or cam.transform is None:
            continue
        T = cam.transform
        got, _ = _try([lambda: ypr_of([list(T.row(i))[:3] for i in range(3)])])
        if got:
            d.append(max(abs(angle_diff(a, b)) for a, b in zip(got, ref["ypr_refined_deg"])))
    out["ypr_formula_vs_metashape_deg"] = {"n": len(d), "max": max(d) if d else None, "median": median(d)}
    tp = tie_points(chunk)
    if tp is not None and len(tp.points):
        coord = {}
        for pt in tp.points:
            c = pt.coord
            coord[pt.track_id] = M.Vector([c.x / c.w, c.y / c.w, c.z / c.w]) if abs(c.w) > 0 else None
        du2, dv2 = [], []
        per_cam = max(20, 20000 // max(1, len(chunk.cameras)))
        for cam in chunk.cameras:
            prj, _ = _try([lambda: tp.projections[cam]])
            for k, pr in enumerate(prj or []):
                if k >= per_cam:
                    break
                X = coord.get(pr.track_id)
                q = cam.project(X) if X is not None else None
                if q is not None:
                    du2.append(q.x - pr.coord.x)
                    dv2.append(q.y - pr.coord.y)
        out["tie_points"] = dict(stats(du2, dv2), points=len(tp.points))
    else:
        out["tie_points"] = {"n": 0, "points": 0}
    return out


def build(job_path, quit_after=False):
    t0 = time.time()
    job = json.load(open(job_path, encoding="utf-8"))
    psx = job["psx"]
    log(f"block {job['block']}: {len(job['images'])} images, {len(job['sensors'])} sensors, "
        f"{job['tie_points']['points']} tie points -> {psx}")
    missing = [im["path"] for im in job["images"] if not os.path.isfile(im["path"])]
    if missing:
        raise RuntimeError(f"{len(missing)} image files missing, e.g. {missing[0]}")
    doc = getattr(M.app, "document", None) or M.Document()
    chunk = doc.addChunk()
    chunk.label = job["block"]
    chunk.crs = M.CoordinateSystem(job["crs_wkt"])
    chunk.transform.matrix = M.Matrix.Diag((1, 1, 1, 1))
    # 1. the photos, linked
    add_photos(chunk, [im["path"] for im in job["images"]])
    by_path = {_norm(c.photo.path): c for c in chunk.cameras}
    cams_by_name = {}
    for im in job["images"]:
        c = by_path.get(_norm(im["path"]))
        if c is None:
            raise RuntimeError(f"no Metashape camera for {im['path']}")
        cams_by_name[im["name"]] = c
    # 2. the tie points (Bundler)
    err = import_bundler(chunk, job)
    tp = tie_points(chunk)
    n_tp = len(tp.points) if tp is not None else 0
    log(f"Bundler import: {n_tp} tie points" + (f" (error: {err})" if err else ""))
    # 3. sensors and poses from the job
    sensors = {s["id"]: make_sensor(chunk, s) for s in job["sensors"]}
    for im in job["images"]:
        c = cams_by_name[im["name"]]
        c.sensor = sensors[im["sensor_id"]]
        c.transform = M.Matrix(im["world_from_camera"])
        _try([lambda: setattr(c.reference, "enabled", False)])
    chunk.euler_angles = getattr(M, "EulerAnglesYPR", chunk.euler_angles)
    n_ref = sum(1 for im in job["images"] if im.get("reference")
                and set_reference(chunk, cams_by_name[im["name"]], im["reference"]) is not None)
    log(f"references: {n_ref} cameras (the PDS label priors MPPP started from)")
    used = set(sensors.values())
    stale = [s for s in chunk.sensors if s not in used]
    if stale:
        _try([lambda: chunk.remove(stale)])
    tie_coords = "none"
    if n_tp:
        # 30 Oct (Chal Rocks): the import triangulates with Bundler's pinhole cameras AND renumbers its tracks (12 of
        # 383,851 dropped, the ids shifted by 0-12), so COLMAP's coordinates by track id landed on the wrong tracks
        # for part of the points (median 0.24 px, rms 480 px).  The points are triangulated again from COLMAP's
        # observations with the job's (exact) cameras; COLMAP's coordinates by id only if that fails.
        _, e1 = _try([lambda: chunk.triangulateTiePoints(max_error=10, min_image=2),
                      lambda: chunk.triangulateTiePoints()])
        if e1 is None:
            tie_coords = "triangulated by Metashape from COLMAP's observations with the job's cameras"
        else:
            n_set, e2 = _try([lambda: set_tie_coords(chunk, read_bundler_points(job["bundler"]))])
            tie_coords = (f"COLMAP by track id ({n_set} points; triangulation failed: {e1})" if n_set
                          else f"as imported (triangulation: {e1}; coordinates: {e2})")
        tp = tie_points(chunk)
        n_tp = len(tp.points) if tp is not None else 0
        log(f"tie point coordinates: {tie_coords}; {n_tp} points")
    if n_tp == 0:                                            # no tracks came in: Metashape's own, poses kept
        log("no tie points from the Bundler file: matching with the poses fixed and triangulating instead")
        _try([lambda: chunk.matchPhotos(downscale=1, generic_preselection=False, reference_preselection=True,
                                        reference_preselection_mode=M.ReferencePreselectionEstimated,
                                        filter_mask=True, mask_tiepoints=True, keypoint_limit=40000,
                                        tiepoint_limit=10000, reset_matches=True)])
        _try([lambda: chunk.triangulateTiePoints()])
        tp = tie_points(chunk)
        n_tp = len(tp.points) if tp is not None else 0
    # 4. masks
    n_mask = sum(1 for im in job["images"] if im.get("mask") and set_mask(chunk, cams_by_name[im["name"]], im["mask"]))
    log(f"masks: {n_mask} of {sum(1 for im in job['images'] if im.get('mask'))}")
    # 5. the region (internal coordinates), then the chunk transform (the block offset)
    rg = job.get("region")
    if rg:
        reg = chunk.region
        reg.center = M.Vector(rg["center"])
        reg.size = M.Vector(rg["size"])
        reg.rot = M.Matrix.Diag((1, 1, 1))
        chunk.region = reg
    off = job.get("offset") or [0, 0, 0]
    chunk.transform.matrix = M.Matrix.Translation(M.Vector(off))
    # 7. saved before the check (30 Oct: a check that fails must not cost the project; the count of tie points had
    # the check's key "tie_points", and the log line then stopped the script before the save)
    doc.save(psx)
    log(f"saved {psx} ({time.time() - t0:.0f} s)")
    # 6. the check
    try:
        chk = check(chunk, job, cams_by_name)
    except Exception as e:                                 # noqa: BLE001
        chk = {"error": f"{type(e).__name__}: {e}"}
    chk.update(block=job["block"], images=len(job["images"]), sensors=len(sensors), tie_points_imported=n_tp,
               tie_point_coordinates=tie_coords, references=n_ref,
               masks=n_mask,
               conversion_max_error_px=job.get("conversion_max_error_px"), seconds=round(time.time() - t0, 1),
               metashape=getattr(M.app, "version", "?"))
    out = os.path.join(os.path.dirname(psx), "metashape_check.json")
    json.dump(chk, open(out, "w", encoding="utf-8"), indent=1)
    co, ti = chk.get("colmap_observations") or {}, chk.get("tie_points") or {}
    if co.get("n"):
        log(f"check: COLMAP observations reprojected by Metashape - median {co['median_px']:.3f} px, rms "
            f"{co['rms_px']:.3f}, mean offset ({co['mean_du_px']:+.3f}, {co['mean_dv_px']:+.3f}) px over {co['n']}")
    if ti.get("n"):
        log(f"check: imported tie points - median {ti['median_px']:.3f} px, rms {ti['rms_px']:.3f}, mean offset "
            f"({ti['mean_du_px']:+.3f}, {ti['mean_dv_px']:+.3f}) px over {ti['n']}")
    log(f"check in {out}")
    if quit_after:
        _try([lambda: M.app.quit()])
    return chk


def main(argv=None):
    a = list(sys.argv[1:] if argv is None else argv)
    jobs = [x for x in a if x.lower().endswith(".json")]
    if not jobs:
        here = os.path.dirname(os.path.abspath(__file__))
        jobs = [os.path.join(here, "metashape_job.json")]
    return build(jobs[0], quit_after="--quit" in a)


if __name__ == "__main__" and M is not None:
    main()

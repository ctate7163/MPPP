"""
Where do the reprojection and alignment errors come from (v0p31; working notes §13)?  Per scape: the share of the
summed squared residual by image radius, track length, depth, resolution, eye, LMST, camera temperature, station and
image; the mean radial / tangential residual by radius (a lens-model signature); the pose offsets from the priors.

Usage:
  python scripts/error_sources.py OUT.json SCAPE_DIR [SCAPE_DIR ...] [--samples label_temps.json]

SCAPE_DIR is a notebook-03 work folder (containing colmap/ with sparse/cahv_ba and error_input/poses.csv).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import pycolmap
from mppp.sfm.project import SfmProject
from mppp.sfm.reconstruction import native_residuals
from mppp.sfm.thermal import image_temperatures
from temperature_bins_experiment import align_project_to_model

SAMPLES = None


def groups(key, vals, r2, edges=None, labels=None):
    out = []
    tot = r2.sum()
    if edges is not None:
        idx = np.digitize(vals, edges)
        keys = range(len(edges) + 1)
    else:
        idx = vals
        keys = sorted(set(vals.tolist()))
    for k in keys:
        m = idx == k
        if not m.any():
            continue
        lab = labels[k] if labels else k
        out.append({key: lab, "n": int(m.sum()), "share_obs": float(m.mean()), "share_sq": float(r2[m].sum() / tot),
                    "rms": float(np.sqrt(r2[m].mean())), "median": float(np.median(np.sqrt(r2[m])))})
    return out


def run(root):
    root = Path(root) / "colmap"
    proj = SfmProject.load(root)
    rec = pycolmap.Reconstruction(str(root / "sparse" / "cahv_ba"))
    align_project_to_model(proj, rec)
    nr = native_residuals(rec, proj)
    r = nr["residual_native_px"]
    ok = np.isfinite(r)
    iid, pid, idx, r = nr["image_id"][ok], nr["point3D_id"][ok], nr["point2D_idx"][ok], r[ok]
    r2 = r ** 2
    byid = {x["image_id"]: x for x in proj.images}
    temps = image_temperatures(proj, samples=SAMPLES)
    # geometry per observation
    rad = np.empty(r.size); depth = np.empty(r.size); tl = np.empty(r.size); rad_c = np.empty(r.size); tan_c = np.empty(r.size)
    order = np.argsort(iid, kind="stable")
    bounds = np.flatnonzero(np.diff(iid[order])) + 1
    for sel in np.split(order, bounds):
        im = rec.images[int(iid[sel[0]])]
        cam = rec.cameras[im.camera_id]
        cx, cy = cam.params[2], cam.params[3]
        T = im.cam_from_world()
        X = np.array([rec.points3D[int(p)].xyz for p in pid[sel]]) @ T.rotation.matrix().T + np.asarray(T.translation)
        kp = np.array([im.points2D[int(k)].xy for k in idx[sel]], float)
        uv = cam.img_from_cam(X)
        d = uv - kp
        v = kp - [cx, cy]
        rn = np.linalg.norm(v, axis=1)
        u = v / np.maximum(rn[:, None], 1e-9)
        rad[sel] = rn / np.hypot(max(cx, cam.width - cx), max(cy, cam.height - cy))
        rad_c[sel] = (d * u).sum(1)
        tan_c[sel] = d[:, 0] * -u[:, 1] + d[:, 1] * u[:, 0]
        depth[sel] = X[:, 2]
    tlen = {p: rec.points3D[int(p)].track.length() for p in np.unique(pid)}
    tl = np.array([tlen[p] for p in pid])
    img_rec = [byid.get(int(i), {}) for i in iid]
    station = np.array([x.get("station", "?") for x in img_rec])
    eye = np.array([str(x.get("instrument", "?"))[:2] for x in img_rec])
    lmst = np.array([float(x.get("lmst") or np.nan) if not isinstance(x.get("lmst"), str) else
                     (lambda s: int(s.split(":")[0]) + int(s.split(":")[1]) / 60 if ":" in s else np.nan)(x["lmst"].split("M")[-1])
                     for x in img_rec])
    Tm = np.array([temps.get(x.get("name"), {}).get("T", np.nan) for x in img_rec])
    scale = np.array([x.get("downsample_scale", 1.0) for x in img_rec])
    out = {"scape": root.parent.name, "observations": int(r.size), "rms": float(np.sqrt(r2.mean())),
           "median": float(np.median(r)), "p95": float(np.percentile(r, 95)),
           "share_sq_top1pct_obs": float(np.sort(r2)[::-1][: max(1, r.size // 100)].sum() / r2.sum()),
           "share_sq_gt_1px": float(r2[r > 1].sum() / r2.sum()), "frac_gt_1px": float((r > 1).mean())}
    out["radius"] = groups("radius", rad, r2, [0.25, 0.5, 0.7, 0.85, 1.0], ["<.25", ".25-.5", ".5-.7", ".7-.85", ".85-1", ">1"])
    out["track"] = groups("track", tl, r2, [2.5, 4.5, 8.5, 16.5], ["2", "3-4", "5-8", "9-16", ">16"])
    out["depth"] = groups("depth_m", depth, r2, [3, 6, 12, 25, 50], ["<3", "3-6", "6-12", "12-25", "25-50", ">50"])
    out["scale"] = groups("downsample", scale, r2)
    out["eye"] = groups("eye", eye, r2)
    out["lmst"] = groups("lmst_h", np.floor(np.nan_to_num(lmst, nan=-1)).astype(int), r2)
    out["temperature"] = groups("T", np.floor(np.nan_to_num(Tm, nan=99) / 10).astype(int) * 10, r2)
    st = groups("station", station, r2)
    out["station"] = sorted(st, key=lambda x: -x["share_sq"])[:6]
    im = groups("image", iid, r2)
    for x in im:
        x["name"] = byid.get(int(x["image"]), {}).get("name", "?")[:40]
    ims = sorted(im, key=lambda x: -x["rms"])
    out["n_images"] = len(im)
    out["worst_images"] = ims[:6]
    out["share_sq_worst5pct_images"] = float(sum(x["share_sq"] for x in sorted(im, key=lambda x: -x["share_sq"])[: max(1, len(im) // 20)]))
    # radial / tangential bias by radius (lens-model signature), full-frame px
    rb = []
    e = np.array([0, .25, .5, .7, .85, 1.0, 1.3])
    for a, b in zip(e[:-1], e[1:]):
        m = (rad >= a) & (rad < b)
        if m.sum() > 100:
            rb.append({"radius": f"{a:.2f}-{b:.2f}", "mean_radial_px": float(np.mean(rad_c[m])),
                       "mean_tangential_px": float(np.mean(tan_c[m])), "n": int(m.sum())})
    out["radial_bias"] = rb
    # alignment: pose offsets from the prior
    import csv
    rows = list(csv.DictReader(open(root / "error_input" / "poses.csv")))
    dC = np.array([float(x["dC_m"]) for x in rows]); dA = np.array([float(x["dAttitude_deg"]) for x in rows])
    bys = {}
    for x in rows:
        bys.setdefault(x["station"], []).append(float(x["dC_m"]))
    out["poses"] = {"dC_median_m": float(np.median(dC)), "dC_p95_m": float(np.percentile(dC, 95)),
                    "dAtt_median_deg": float(np.median(dA)), "dAtt_p95_deg": float(np.percentile(dA, 95)),
                    "worst_stations": sorted(((k, float(np.median(v)), len(v)) for k, v in bys.items()), key=lambda t: -t[1])[:4]}
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("scapes", nargs="+")
    ap.add_argument("--samples", help="label temperatures {stem: {NL, NR}} for images without camera_temperature_degC")
    a = ap.parse_args()
    if a.samples:
        SAMPLES = json.loads(Path(a.samples).read_text())
    res = {}
    for d in a.scapes:
        try:
            res[Path(d).name] = run(d)
            print("done", d, flush=True)
        except Exception as ex:                   # noqa: BLE001
            print("FAILED", d, repr(ex), flush=True)
        Path(a.out).write_text(json.dumps(res, indent=1, default=float))

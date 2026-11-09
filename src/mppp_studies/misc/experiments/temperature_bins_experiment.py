"""
Navcam focal length against camera temperature inside solved blocks (v0p31; mppp.sfm.thermal).

For each scape: load the refined block (sparse/cahv_ba) and its project, give every Navcam image its camera
temperature, re-adjust once with one camera per eye (fx, fy free, everything else held: the reference) and once
with one camera per eye and temperature bin, and write the bins' focal lengths.  Then fit f = a_scape + b T over
all bins (the within-block thermal coefficient).

Usage:
  python studies/experiments/temperature_bins_experiment.py OUT.json SCAPE_DIR [SCAPE_DIR ...]
         [--samples label_temps.json] [--pds D:/data/m2020] [--bin 10] [--min-images 8] [--free fx,fy]

SCAPE_DIR is a notebook-03 work folder (containing colmap/) or its colmap/ folder.  Temperatures come from the
project (MPPP >= 0.30 manifests), from --samples ({stem: {"NL", "NR"}} label temperatures, interpolated in
spacecraft clock), or from the labels under --pds.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np  # noqa: E402


def align_project_to_model(project, rec):
    """Make the project describe exactly the model's images (by name): image and camera ids from the model; model
    images the project does not know are deregistered.  Returns the number of such frames."""
    from mppp.sfm.reconstruction import exclude_frames
    by_name = {r["name"]: r for r in project.images}
    missing_frames = {im.frame_id for im in rec.images.values() if im.name not in by_name}
    if missing_frames:
        exclude_frames(rec, sorted(missing_frames))
    keep = []
    cams = {}
    for iid, im in rec.images.items():
        r = by_name.get(im.name)
        if r is None:
            continue
        r["image_id"], r["camera_id"], r["frame_id"] = int(iid), int(im.camera_id), int(im.frame_id)
        cams.setdefault(r["instrument"], int(im.camera_id))
        keep.append(r)
    project.images = keep
    db = project.settings.setdefault("database", {})
    db["cameras"] = {k: v for k, v in cams.items()}
    return len(missing_frames)


def run_scape(root, samples, pds, bin_deg, min_images, free, verbose=True):
    import pycolmap
    from mppp.sfm.project import SfmProject
    from mppp.sfm.thermal import image_temperatures, thermal_adjust
    root = Path(root)
    if (root / "colmap").is_dir():
        root = root / "colmap"
    project = SfmProject.load(root)
    rec = pycolmap.Reconstruction(str(root / "sparse" / "cahv_ba"))
    n_drop = align_project_to_model(project, rec)
    temps = image_temperatures(project, samples=samples, pds_dir=pds, verbose=verbose)
    t0 = time.time()
    out = thermal_adjust(rec, project, temps, bin_deg=bin_deg, min_images=min_images, free=free, verbose=verbose)
    Ts = np.array([v["T"] for v in temps.values()])
    res = {"root": str(root), "frames_not_in_project": n_drop, "images": len(project.images),
           "with_temperature": len(temps), "T_range_degC": [float(Ts.min()), float(Ts.max())] if Ts.size else None,
           "reference": out.get("reference"), "bins": out["bins"], "rows": out["rows"],
           "seconds": round(time.time() - t0, 1),
           "temperature_sources": {k: int(v) for k, v in zip(*np.unique([v["source"] for v in temps.values()],
                                                                          return_counts=True))}}
    return res, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("scapes", nargs="+")
    ap.add_argument("--samples")
    ap.add_argument("--pds")
    ap.add_argument("--bin", type=float, default=10.0)
    ap.add_argument("--min-images", type=int, default=8)
    ap.add_argument("--free", default="fx,fy")
    a = ap.parse_args(argv)
    from mppp.sfm.thermal import fit_focal_temperature
    samples = json.loads(Path(a.samples).read_text()) if a.samples else None
    free = tuple(x.strip() for x in a.free.split(",") if x.strip())
    out_path = Path(a.out)
    results = json.loads(out_path.read_text()) if out_path.exists() else {"scapes": {}}
    for s in a.scapes:
        name = Path(s).name.replace("_colmap", "")
        print(f"==== {name}", flush=True)
        try:
            res, _ = run_scape(s, samples, a.pds, a.bin, a.min_images, free)
        except Exception as e:                                   # noqa: BLE001
            import traceback
            traceback.print_exc()
            res = {"error": f"{type(e).__name__}: {e}"}
        results["scapes"][name] = res
        out_path.write_text(json.dumps(results, indent=1, default=float))
    rows = [dict(r, scape=n) for n, v in results["scapes"].items() for r in v.get("rows", [])]
    results["fit"] = {p: fit_focal_temperature(rows, p) for p in ("fx", "fy")}
    out_path.write_text(json.dumps(results, indent=1, default=float))
    for p, f in results["fit"].items():
        for eye, v in f.items():
            print(f"{eye} {p}: {v['px_per_degC']:+.4f} +- {v['sd']:.4f} px/degC ({v['ppm_per_degC']:+.1f} +- "
                  f"{v['ppm_sd']:.1f} ppm/degC) over {v['rows']} bins of {v['scapes']} scapes, residual "
                  f"{v['residual_rms_px']:.2f} px")


if __name__ == "__main__":
    main()

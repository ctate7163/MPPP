"""
The refined figures of a Navcam run (MPPP 0.90), in the formats of the calibration study (project doc "plot
formats", 20 Oct 2026), from the promoted camera models, a consensus study folder and the finished blocks:

    python scripts/paper_figures.py --root D:\\scapes --study D:\\scapes\\camera_analysis\\navcal_v0p86_20261020_pipeline_pass2
                                    [--out "D:\\code\\MPPP\\Claude outputs"] [--label "after the 20 Oct pass"]
                                    [--blocks folder ...] [--terrain DIR --refdata DIR] [--frame image.png ...]

Writes, into ``--out`` (default ``<MPPP>/Claude outputs``):
  navcam_rig_trends_<tag>.png            rig trends 2x2 (mppp.calib.constrain.plot_rig_trends) - needs --study
  navcam_intrinsics_vs_temperature_<tag>.png   (plot_intrinsics_vs_temperature) - needs --study and the blocks
  lmst_histogram_<tag>.png + _by_site.csv      (mppp.waypoints.blocks.plot_lmst_histograms)
  station_maps_<tag>/<block>.png         the blocks' health/station_map.png (mppp.sfm.export.plot_camera_shifts)
  navcam_block_table_<tag>.md/.json      each block against the joint (mppp.calib.fit.navcam_block_table)
  site_map.png                           the traverse map (mppp.waypoints.blocks.plot_site_map, with --terrain)
  undistorted_example.png                with --frame (mppp.colmap.plot_undistorted_example)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent


def finished_blocks(root: Path, group: str = "navcam_consensus"):
    """The block folders of a site group under ``root`` whose alignment finished (``colmap/run_done.json``)."""
    from mppp.waypoints.sites import load_site_table
    tab = load_site_table()
    out = []
    for name in tab["groups"].get(group) or []:
        for f in sorted(glob.glob(str(root / f"mars2020_sol_*_{name}"))):
            if (Path(f) / "colmap" / "run_done.json").is_file():
                out.append(Path(f))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"), help="the scapes root (default MPPP_SCAPES)")
    ap.add_argument("--blocks", nargs="*", default=None, help="block folders instead of the group under --root")
    ap.add_argument("--group", default="navcam_consensus")
    ap.add_argument("--study", default=None, help="a consensus study folder (consensus.json, rig_residuals.json, navcam_joint/)")
    ap.add_argument("--cmods", default=None, help="the camera models the blocks were run with (default: the package's)")
    ap.add_argument("--out", default=str(HERE / "Claude outputs"))
    ap.add_argument("--tag", default=None, help="file name tag (default: the study folder's date or 'run')")
    ap.add_argument("--label", default="", help="figure title label, e.g. 'after the 20 Oct pass'")
    ap.add_argument("--terrain", default=os.environ.get("MPPP_TERRAIN"), help="folder of quantized-mesh .terrain tiles")
    ap.add_argument("--refdata", default=None, help="folder with the site study's refdata.py (Mastcam-Z 360s, sample cores)")
    ap.add_argument("--frame", default=None, help="a processed PNG frame for the undistortion example")
    ap.add_argument("--frame-model", default="NL", help="its camera: NL, NR (promoted model) or a COLMAP cameras JSON")
    ap.add_argument("--frame-mask", default=None)
    ap.add_argument("--frame-T", type=float, default=-20.0)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from mppp.calib import constrain as C, fit as F, models as M
    from mppp.waypoints import blocks as B
    from mppp.waypoints.sites import load_site_table
    from mppp.waypoints.table import snapshot_path
    study = Path(a.study) if a.study else None
    tag = a.tag or (study.name.split("_")[2] if study and len(study.name.split("_")) > 2 else "run")
    inuse = {"NL": M.load_camera("N", "L", a.cmods), "NR": M.load_camera("N", "R", a.cmods)}
    rig_inuse = M.load_rig("N", a.cmods)
    folders = [Path(b) for b in a.blocks] if a.blocks else (finished_blocks(Path(a.root), a.group) if a.root else [])
    res = []
    for f in folders:
        try:
            res.append(F.block_navcam_results(f))
        except Exception as e:                                             # noqa: BLE001
            print(f"{f.name}: skipped ({type(e).__name__}: {e})")
    print(f"{len(res)} finished blocks")
    # ---- rig trends and intrinsics vs temperature (the study's refined formats)
    if study and (study / "navcam_joint" / "consensus.json").is_file():
        cons = json.loads((study / "navcam_joint" / "consensus.json").read_text(encoding="utf-8"))
        rig_new = json.loads((study / "navcam_joint" / "M2020_N_rig.json").read_text(encoding="utf-8"))
        rrf = study / "rig_residuals.json"
        if rrf.is_file():
            rr = json.loads(rrf.read_text(encoding="utf-8"))
            th = {e: C.thermal_terms(inuse[e].get("thermal")) for e in ("NL", "NR")}
            C.plot_rig_trends(rr, cons, rig_new["drift"], out / f"navcam_rig_trends_{tag}.png",
                              cx_inuse_px_per_degC=th["NR"]["cx_px"] - th["NL"]["cx_px"], label=a.label,
                              model_label=f"{tag} drift model")
            tb = study / "thermal_bins.json"                 # 0.91: the tied joint's diagnostic block bins
            bins = json.loads(tb.read_text(encoding="utf-8")) if tb.is_file() else (F.thermal_bin_rows(res) if res else [])
            pu = study / "pp_units.json"                     # 0.91: the pointing pass's per-block principal points
            pp_units = json.loads(pu.read_text(encoding="utf-8")) if pu.is_file() else None
            if bins:
                C.plot_intrinsics_vs_temperature(cons, inuse, bins, rr,
                                                 out / f"navcam_intrinsics_vs_temperature_{tag}.png", label=a.label,
                                                 model_label=tag, pp_units=pp_units)
        else:
            print(f"no rig_residuals.json in {study}: rig trends and intrinsics figures skipped")
    # ---- the block table against the joint the blocks were run with
    if res:
        rows = F.navcam_block_table(res, inuse["NL"], inuse["NR"], rig_inuse)
        (out / f"navcam_block_table_{tag}.md").write_text(F.block_table_markdown(rows) + "\n", encoding="utf-8")
        (out / f"navcam_block_table_{tag}.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
        drift = F.fit_rig_drift(F.rig_rows(res), knot_sol="free")
        (out / f"rig_drift_refit_{tag}.json").write_text(json.dumps(drift, indent=1, default=float), encoding="utf-8")
    # ---- station maps: the blocks' own health figures (plot_camera_shifts), collected
    if folders:
        sm = out / f"station_maps_{tag}"
        sm.mkdir(exist_ok=True)
        for f in folders:
            src = f / "colmap" / "health" / "station_map.png"
            if src.is_file():
                shutil.copyfile(src, sm / f"{f.name}.png")
    # ---- LMST histograms (the analyses script's format)
    if folders:
        roots = sorted({str(f.parent) for f in folders})
        from mppp.config import default_config
        from mppp.sfm.project import NAVCAM_MIN_FRAME_FRACTION
        sites = [f.name.split("_", 3)[-1] for f in folders]
        images, _used = B.lmst_images(roots, sites, NAVCAM_MIN_FRAME_FRACTION or None)
        if images:
            rows = sorted((B.lmst_site_row(k, list(v.values())) for k, v in images.items()), key=lambda r: r["first_sol"])
            window = (default_config().get("selection") or {}).get("lmst_window_h")
            B.plot_lmst_histograms(images, rows, out / f"lmst_histogram_{tag}.png", tuple(window) if window else None)
            B.write_lmst_table(rows, out / f"lmst_histogram_{tag}_by_site.csv")
    # ---- the traverse map
    tab = load_site_table()
    panos, samples = (), ()
    if a.refdata:
        sys.path.insert(0, str(Path(a.refdata)))
        from refdata import PANOS, SAMPLES                                      # the site study's reference lists
        panos, samples = PANOS, SAMPLES
    B.plot_site_map(None, snapshot_path(), out / "site_map.png", terrain_dir=a.terrain, panos=panos, samples=samples,
                    highlight=tab["groups"].get(a.group) or [])
    if a.frame:
        import cv2
        from mppp.calib.constrain import camera_at
        from mppp.colmap import plot_undistorted_example
        im = cv2.cvtColor(cv2.imread(a.frame), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(a.frame_mask, cv2.IMREAD_GRAYSCALE) if a.frame_mask else None
        if a.frame_model in ("NL", "NR"):
            model, p, _ = camera_at(inuse[a.frame_model], T_degC=a.frame_T, sol=1000)
            width, lens = 5120, f"THIN_PRISM_FISHEYE {a.frame_model}, promoted model at {a.frame_T:g} °C"
        else:
            m = json.loads(Path(a.frame_model).read_text())
            model, p, width, lens = m["model"], m["params"], int(m.get("width", 5120)), m.get("source", m["model"])
        plot_undistorted_example(im, model, p, width, out / "undistorted_example.png", mask=mask,
                                 title=Path(a.frame).name, lens=lens)
    print(f"figures in {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

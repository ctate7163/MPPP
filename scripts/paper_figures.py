"""
The standard figures (MPPP 0.91; scripts\\windows\\make_figures.bat), in the formats of the calibration study
(project doc "plot formats", 20 Oct 2026), from the promoted camera models, a Navcam joint folder, a Mastcam-Z
calibration folder and the finished blocks, at 200 dpi by default:

    python scripts/paper_figures.py --root D:\\scapes --study D:\\scapes\\camera_analysis\\navcal_v0p86_20261020_pipeline_pass2
                                    [--out "D:\\code\\MPPP\\Claude outputs"] [--label "after the 20 Oct pass"]
                                    [--zcam D:\\scapes\\camera_analysis\\zcam34_calibration_20261021] [--dpi 200]
                                    [--site-map maps|simple|none] [--blocks folder ...] [--frame image.png ...]

Writes, into ``--out`` (default ``<MPPP>/Claude outputs``):
  navcam_rig_trends_<tag>.png            rig trends 2x2 (mppp.calib.constrain.plot_rig_trends) - needs --study;
                                         _nolabels.png: the same without the block name at each point
  navcam_intrinsics_vs_temperature_<tag>.png   (plot_intrinsics_vs_temperature) - needs --study and the blocks
  lmst_histogram_<tag>.png + _by_site.csv      (mppp.waypoints.blocks.plot_lmst_histograms)
  station_maps_<tag>/<block>.png         the blocks' health/station_map.png (mppp.sfm.export.plot_camera_shifts)
  navcam_block_table_<tag>.md/.json      each block against the joint (mppp.calib.fit.navcam_block_table)
  zcam_rig_trends_<tag>.png              Mastcam-Z rig rotations and baseline vs sol and lens temperature - --zcam
                                         (+ _nolabels.png)
  zcam_focus_lines_vs_temperature_<tag>.png   the blocks' focus lines vs lens temperature - --zcam
  zcam<zoom>_efl_vs_focus_<tag>.png           a joint folder's focal length against focus (law and bins) - --zcam
  zcam<zoom>_efl_vs_focus_<joint>.png         0.95.2: the same for the newest 34, 48 and 63 mm joints (--zcam-joints,
                                              --zcam-zooms; the joint folder's name in the file name)
  hazcam_rig_trends_<F|R>_<fit>.png           0.95.2: the newest Hazcam fit's rig trends (+ _nolabels) - --hazcam
  hazcam_efl_vs_temperature_<fit>.png         0.95.2: its eyes' focal length against temperature (law and block bins)
  site_map_<tag>.png                     the traverse map: figure 2 of ``mppp maps`` (crater | traverse, block table,
                                         1 m DTM where M20_DEMS has it); --site-map simple: plot_site_map
  undistorted_example.png                with --frame (mppp.colmap.plot_undistorted_example)
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Optional, Union

PathLike = Union[str, Path]

HERE = Path(__file__).resolve().parent.parent


def finished_blocks(root: Path, group: str = "ecam_consensus"):
    """The block folders of a site group under ``root`` whose alignment finished (``colmap/run_done.json``)."""
    from mppp.waypoints.sites import load_site_table
    tab = load_site_table()
    out = []
    for name in tab["groups"].get(group) or []:
        for f in sorted(glob.glob(str(root / f"mars2020_sol_*_{name}"))):
            if (Path(f) / "colmap" / "run_done.json").is_file():
                out.append(Path(f))
    return out


def site_map_from_maps(path: Path, dpi: int = 200) -> bool:
    """The site map as figure 2 of the site-map study (``mppp maps figures 2``: crater | traverse with the official
    blocks and their table; the 1 m DTM where M20_DEMS has it), copied to ``path``."""
    import subprocess
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        r = subprocess.run([sys.executable, "-m", "mppp", "maps", "figures", "2", "--out", td, "--background", "dtm",
                            "--tag", "dtm", "--dpi", str(dpi)])
        found = sorted(Path(td).glob("m20_jezero_traverse_clusters_map*.png"))
        if r.returncode or not found:
            print("site map: mppp maps figures 2 failed - see above")
            return False
        shutil.copyfile(found[0], path)
    print(f"site map: {path}")
    return True


def _newest(paths) -> Optional[Path]:
    """The folder whose file (the listed json) was written last."""
    best = None
    for f in paths:
        try:
            t = Path(f).stat().st_mtime
        except OSError:
            continue
        if best is None or t > best[0]:
            best = (t, Path(f).parent)
    return best[1] if best else None


def newest_zcam_joint(root: PathLike, zoom: int) -> Optional[Path]:
    """0.95.2: the newest Mastcam-Z joint folder of a zoom under <root>/camera_analysis: zcam<zoom>_joint_<date>[_N]
    (zcam_joints.bat) or <study>/zcam<zoom>_joint (trial_joints_0p95.bat), with its zcam_joint.json."""
    ca = Path(root) / "camera_analysis"
    return _newest(glob.glob(str(ca / f"zcam{int(zoom)}_joint_*" / "zcam_joint.json"))
                   + glob.glob(str(ca / "*" / f"zcam{int(zoom)}_joint" / "zcam_joint.json")))


def newest_hazcam_fit(root: PathLike) -> Optional[Path]:
    """0.95.2: the newest Hazcam fit folder under <root>/camera_analysis: hazcam_fit_<group>_<date>[_N] or
    <study>/hazcam_fit, with its hazcam_fit.json."""
    ca = Path(root) / "camera_analysis"
    return _newest(glob.glob(str(ca / "hazcam_fit_*" / "hazcam_fit.json"))
                   + glob.glob(str(ca / "*" / "hazcam_fit" / "hazcam_fit.json")))


def folder_tag(folder: PathLike) -> str:
    """A file-name tag for a joint or fit folder: its name, with its study's name for <study>/zcam34_joint."""
    f = Path(folder)
    return f.name if re.search(r"_\d{8}", f.name) else f"{f.parent.name}_{f.name}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"), help="the scapes root (default MPPP_SCAPES)")
    ap.add_argument("--blocks", nargs="*", default=None, help="block folders instead of the group under --root")
    ap.add_argument("--group", default="ecam_consensus")
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
    ap.add_argument("--zcam", default=None, help="a Mastcam-Z calibration folder (zcam_calibration.py calibrate)")
    ap.add_argument("--zcam-joints", nargs="*", default=None,
                    help="Mastcam-Z joint folders (zcam_joint.json) for the efl vs focus figures; default: the newest "
                         "of each of --zcam-zooms under <root>\\camera_analysis; '-' for none")
    ap.add_argument("--zcam-zooms", default="34,48,63", help="the zooms of the default joint folders (0.95.2: 34,48,63)")
    ap.add_argument("--hazcam", default=None, help="a Hazcam fit folder (hazcam_fit.json); default: the newest "
                                                   "under <root>\\camera_analysis; '-' for none")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--site-map", choices=["maps", "simple", "none"], default="maps")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from mppp.calib import constrain as C, fit as F, models as M
    from mppp.waypoints import blocks as B
    from mppp.waypoints.sites import load_site_table
    from mppp.waypoints.table import snapshot_path
    study = Path(a.study) if a.study else None
    tag = a.tag or (study.name if study else "run")
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
            for suffix, labels in (("", True), ("_nolabels", False)):     # 22 Oct: also without the block names
                C.plot_rig_trends(rr, cons, rig_new["drift"], out / f"navcam_rig_trends_{tag}{suffix}.png",
                                  cx_inuse_px_per_degC=th["NR"]["cx_px"] - th["NL"]["cx_px"], label=a.label,
                                  model_label=f"{tag} drift model", dpi=a.dpi, point_labels=labels)
            tb = study / "thermal_bins.json"                 # 0.91: the tied joint's diagnostic block bins
            tj = study / "navcam_joint" / "tied_joint.json"  # 22 Oct: re-cut from the joint (>= 4 images, more cold bins)
            if tj.is_file():
                from mppp.calib.tied import figure_inputs
                bins = figure_inputs(json.loads(tj.read_text(encoding="utf-8")))["bins"]
            else:
                bins = json.loads(tb.read_text(encoding="utf-8")) if tb.is_file() else (F.thermal_bin_rows(res) if res else [])
            pu = study / "pp_units.json"                     # 0.91: the pointing pass's per-block principal points
            pp_units = json.loads(pu.read_text(encoding="utf-8")) if pu.is_file() else None
            if bins:
                C.plot_intrinsics_vs_temperature(cons, inuse, bins, rr,
                                                 out / f"navcam_intrinsics_vs_temperature_{tag}.png", label=a.label,
                                                 model_label=tag, pp_units=pp_units, dpi=a.dpi)
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
            B.plot_lmst_histograms(images, rows, out / f"lmst_histogram_{tag}.png", tuple(window) if window else None,
                                   dpi=a.dpi)
            B.write_lmst_table(rows, out / f"lmst_histogram_{tag}_by_site.csv")
    # ---- Mastcam-Z: the calibration folder's model and block lines
    if a.zcam:
        zd = Path(a.zcam)
        zm = zd / "M2020_ZCAM034_focus_model.json"
        if zm.is_file():
            zmodel = json.loads(zm.read_text(encoding="utf-8"))
            for suffix, labels in (("", True), ("_nolabels", False)):
                C.plot_zcam_rig_trends(zmodel, out / f"zcam_rig_trends_{tag}{suffix}.png", dpi=a.dpi, label=zd.name,
                                       point_labels=labels)
        bl = zd / "block_lines.json"
        if bl.is_file():
            sys.path.insert(0, str(HERE / "studies" / "calibration"))
            import matplotlib
            matplotlib.use("Agg")
            from zcam_lines import plot_block_lines
            plot_block_lines(json.loads(bl.read_text(encoding="utf-8")), out / f"zcam_focus_lines_vs_temperature_{tag}.png",
                             dpi=a.dpi)
        zj = zd / "zcam_joint.json"                         # 0.95.1: the efl against focus of a joint folder
        if zj.is_file():
            from mppp.calib.constrain import plot_zcam_efl_vs_focus
            res_ = json.loads(zj.read_text(encoding="utf-8"))
            plot_zcam_efl_vs_focus(res_, out / f"zcam{int(res_.get('zoom', 34))}_efl_vs_focus_{tag}.png", dpi=a.dpi,
                                   label=zd.name)
        if not zm.is_file() and not bl.is_file():
            print(f"no Mastcam-Z calibration in {zd}: zcam figures skipped")
    # ---- 0.95.2 (Christian, 30 Oct): the efl vs focus of the newest joint of each zoom (34, 48, 63 by default)
    from mppp.calib.constrain import plot_hazcam_efl_vs_temperature, plot_hazcam_rig_trends, plot_zcam_efl_vs_focus
    done_joints = {str(Path(a.zcam).resolve())} if a.zcam and (Path(a.zcam) / "zcam_joint.json").is_file() else set()
    if a.zcam_joints == ["-"]:
        joints = []
    elif a.zcam_joints:
        joints = [Path(j) for j in a.zcam_joints]
    else:
        joints = [newest_zcam_joint(a.root, int(z)) for z in str(a.zcam_zooms).split(",") if z.strip()] if a.root else []
        for z, j in zip([z for z in str(a.zcam_zooms).split(",") if z.strip()], joints):
            if j is None:
                print(f"no Mastcam-Z {z} mm joint folder under {a.root}\\camera_analysis")
    for j in [j for j in joints if j is not None]:
        if str(j.resolve()) in done_joints:
            continue
        done_joints.add(str(j.resolve()))
        zj = j / "zcam_joint.json"
        if not zj.is_file():
            print(f"{j}: no zcam_joint.json - skipped")
            continue
        res_ = json.loads(zj.read_text(encoding="utf-8"))
        f = plot_zcam_efl_vs_focus(res_, out / f"zcam{int(res_.get('zoom', 34))}_efl_vs_focus_{folder_tag(j)}.png",
                                   dpi=a.dpi, label=folder_tag(j))
        print(f"Mastcam-Z {res_.get('zoom', 34)} mm: {j} -> {f.name if f else 'no eyes'}")
    # ---- 0.95.2 (Christian, 30 Oct): the Hazcams - rig trends of each pair, focal length against temperature
    hz = None if a.hazcam == "-" else (Path(a.hazcam) if a.hazcam else (newest_hazcam_fit(a.root) if a.root else None))
    if hz is not None:
        hf = hz / "hazcam_fit.json"
        if hf.is_file():
            hres = json.loads(hf.read_text(encoding="utf-8"))
            htag = folder_tag(hz)
            made = []
            for fam, r in (hres.get("rigs") or {}).items():
                if r.get("skipped"):
                    continue
                for suffix, labels in (("", True), ("_nolabels", False)):
                    f = plot_hazcam_rig_trends(r, out / f"hazcam_rig_trends_{fam}_{htag}{suffix}.png", dpi=a.dpi,
                                               label=f"({htag})", point_labels=labels)
                    if f:
                        made.append(f.name)
            f = plot_hazcam_efl_vs_temperature(hres, out / f"hazcam_efl_vs_temperature_{htag}.png", label=f"({htag})",
                                               dpi=a.dpi)
            if f:
                made.append(f.name)
            print(f"Hazcam: {hz} -> {', '.join(made) or 'nothing to plot'}")
        else:
            print(f"{hz}: no hazcam_fit.json - Hazcam figures skipped")
    # ---- the traverse map
    if a.site_map == "maps":
        site_map_from_maps(out / f"site_map_{tag}.png", a.dpi)
    elif a.site_map == "simple":
        tab = load_site_table()
        panos, samples = (), ()
        if a.refdata:
            sys.path.insert(0, str(Path(a.refdata)))
            from refdata import PANOS, SAMPLES                                      # the site study's reference lists
            panos, samples = PANOS, SAMPLES
        B.plot_site_map(None, snapshot_path(), out / f"site_map_{tag}.png", terrain_dir=a.terrain, panos=panos,
                        samples=samples, highlight=tab["groups"].get(a.group) or [], dpi=a.dpi)
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

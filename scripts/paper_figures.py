"""
The example figures of the JOSS paper and the README (MPPP 0.90), from the promoted camera models in the package,
the finished Navcam blocks of a scapes root and the shipped site table:

    python scripts/paper_figures.py --root D:\\scapes --out paper/figures [--terrain studies/m20_sites/m20_sites/terrain]
                                    [--blocks folder ...] [--frame image.png --frame-model NL --frame-mask mask.png]

Writes navcam_cmod_vs_temperature.png, navcam_rig_vs_sol.png (mppp.calib.constrain), navcam_block_table.md
(mppp.calib.fit), site_map.png and lmst_histograms.png (mppp.waypoints.blocks) and, with ``--frame``,
undistorted_example.png (mppp.colmap).  Each plotting function lives in the module of the thing it plots.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def finished_blocks(root: Path, group: str = "navcam_consensus"):
    """The block folders of a site group under ``root`` whose alignment finished (``colmap/run_done.json``)."""
    from mppp.waypoints.sites import load_site_table
    tab = load_site_table()
    names = tab["groups"].get(group) or []
    out = []
    for name in names:
        for f in sorted(glob.glob(str(root / f"mars2020_sol_*_{name}"))):
            if (Path(f) / "colmap" / "run_done.json").is_file():
                out.append(Path(f))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_SCAPES"), help="the scapes root (default MPPP_SCAPES)")
    ap.add_argument("--blocks", nargs="*", default=None, help="block folders instead of the group under --root")
    ap.add_argument("--group", default="navcam_consensus")
    ap.add_argument("--out", default="paper/figures")
    ap.add_argument("--terrain", default=os.environ.get("MPPP_TERRAIN"), help="folder of quantized-mesh .terrain tiles")
    ap.add_argument("--manifest-version", default=None, help="processing manifest version for the LMST plot (default: newest)")
    ap.add_argument("--frame", default=None, help="a processed PNG frame for the undistortion example")
    ap.add_argument("--frame-model", default="NL", help="its camera: NL, NR (promoted model) or a COLMAP cameras JSON")
    ap.add_argument("--frame-mask", default=None)
    ap.add_argument("--frame-T", type=float, default=-20.0)
    ap.add_argument("--refdata", default=None, help="folder with the site study's refdata.py (Mastcam-Z 360s, sample cores)")
    ap.add_argument("--cmods", default=None, help="a folder of camera model files instead of the package's (a candidate joint)")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from mppp.calib import constrain as C, fit as F, models as M
    from mppp.waypoints import blocks as B
    from mppp.waypoints.sites import load_site_table, site_label
    from mppp.waypoints.table import snapshot_path
    nl, nr, rig = M.load_camera("N", "L", a.cmods), M.load_camera("N", "R", a.cmods), M.load_rig("N", a.cmods)
    folders = [Path(b) for b in a.blocks] if a.blocks else (finished_blocks(Path(a.root), a.group) if a.root else [])
    res = []
    for f in folders:
        try:
            res.append(F.block_navcam_results(f))
        except Exception as e:                                             # noqa: BLE001
            print(f"{f.name}: skipped ({type(e).__name__}: {e})")
    print(f"{len(res)} finished blocks")
    C.plot_camera_vs_temperature(nl, nr, out / "navcam_cmod_vs_temperature.png", blocks=res)
    C.plot_rig_vs_sol(rig, out / "navcam_rig_vs_sol.png", blocks=res)
    if res:
        rows = F.rig_rows(res)
        drift = F.fit_rig_drift(rows, knot_sol="free")
        C.plot_rig_drift(rows, drift, out / "navcam_rig_drift.png", old_drift=rig.get("drift"),
                         labels={"fit": "refit (sol0 = 0, free knot)", "old": "rig file drift", "base": "rig epochs"})
        (out / "rig_drift_refit.json").write_text(json.dumps(drift, indent=1, default=float), encoding="utf-8")
        M.write_model(out / "M2020_N_rig_candidate.json", F.rig_with_drift(rig, drift))
    if res:
        rows = F.navcam_block_table(res, nl, nr, rig)
        (out / "navcam_block_table.md").write_text(F.block_table_markdown(rows) + "\n", encoding="utf-8")
        (out / "navcam_block_table.json").write_text(json.dumps(rows, indent=1), encoding="utf-8")
    tab = load_site_table()
    panos, samples = (), ()
    if a.refdata:
        sys.path.insert(0, str(Path(a.refdata)))
        from refdata import PANOS, SAMPLES                                      # the site study's reference lists
        panos, samples = PANOS, SAMPLES
    B.plot_site_map(None, snapshot_path(), out / "site_map.png", terrain_dir=a.terrain, panos=panos, samples=samples,
                    highlight=tab["groups"].get(a.group) or [])
    data = {}
    for f in folders:
        ms = sorted(glob.glob(str(f / "processed" / f"mppp_manifest_{a.manifest_version or '*'}.json")))
        if ms:
            data[site_label(f.name.split("_", 3)[-1])] = B.manifest_lmst_temperature(ms[-1])
    if data:
        B.plot_lmst_histograms(data, out / "lmst_histograms.png")
    if a.frame:
        import cv2
        from mppp.calib.constrain import camera_at
        from mppp.colmap import plot_undistorted_example
        im = cv2.cvtColor(cv2.imread(a.frame), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(a.frame_mask, cv2.IMREAD_GRAYSCALE) if a.frame_mask else None
        if a.frame_model in ("NL", "NR"):
            model, p, _ = camera_at(M.load_camera("N", a.frame_model[1]), T_degC=a.frame_T, sol=1000)
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

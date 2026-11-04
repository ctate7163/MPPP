"""
The rover deck fiducial check of the solved Navcam blocks (0.95.5, Christian 3 Nov 2026): the surveyed deck targets
measured in every Navcam image that sees them, against each block's own refined cameras and stereo rig
(:mod:`mppp.calib.fiducials`).

    python scripts\\fiducial_check.py                                   (every solved block under MPPP_FREEZE_ROOT / MPPP_SCAPES)
    python scripts\\fiducial_check.py --root D:\\rover_blocks --blocks seitah_huge threeforks_large
    python scripts\\fiducial_check.py --works D:\\rover_blocks\\mars2020_sol_0201_seitah_huge --figures 6

Writes ``<root>/camera_analysis/fiducials_<yyyymmdd>/``:

* ``pairs.csv`` - one row per Navcam image pair (or single image) with targets: the label residual, one-eye
  resections, the rig-held fit (``rig_rms``), the baseline scale against the survey (``scale_ppm`` +- sd),
  the rig translation and rotation the deck sees (``dt_*_mm``, ``dR_*_mdeg``), the triangulated deck's scale and
  shape, the LDIFF-RDIFF separation, the NL pose in the rover frame and the rover pose in the block;
* ``detections.csv`` - every target found (full-resolution px: measured, label, fitted);
* ``<block>_*_fiducials.png`` - annotated images with all six rigid targets found;
* ``summary.json`` and ``report.md`` - robust means over all pairs and per block, scale against temperature,
  rover-pose consistency within stations; ``fiducials_*.png`` plots.

Reads only; nothing in the blocks changes.  Needs the block images (the project's ``images`` folder).
"""
from __future__ import annotations

import argparse
import datetime
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mppp.calib import fiducials as F          # noqa: E402


def find_blocks(root: Path, model: str, names: List[str]) -> List[Path]:
    out = []
    for d in sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("mars2020_sol_")):
        if names and not any(d.name == n or d.name.endswith("_" + n) for n in names):
            continue
        if (d / "colmap" / "sparse" / model / "images.bin").is_file() and (d / "colmap" / "project.json").is_file():
            out.append(d)
    return out


def plots(pairs: List[Dict[str, Any]], out: Path) -> List[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return []
    files = []
    use = [p for p in pairs if F._num(p.get("rig_rms")) is not None]
    if not use:
        return files
    blocks = sorted({p["block"] for p in use})
    cmap = plt.get_cmap("tab20")
    col = {b: cmap(i % 20) for i, b in enumerate(blocks)}

    def short(b):
        return b.split("_", 3)[-1] if b.startswith("mars2020_sol_") else b

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for b in blocks:
        bp = [p for p in use if p["block"] == b and F._num(p.get("scale_ppm")) is not None]
        if not bp:
            continue
        T = [F._num(p.get("T_L")) for p in bp]
        s = [p["scale_ppm"] for p in bp]
        e = [F._num(p.get("scale_sd_ppm")) or 0.0 for p in bp]
        ax[0].errorbar(T, s, yerr=e, fmt="o", ms=4, color=col[b], alpha=0.7, label=short(b))
        ax[1].errorbar([p["sol"] for p in bp], s, yerr=e, fmt="o", ms=4, color=col[b], alpha=0.7)
    for a in ax:
        a.axhline(0, color="k", lw=0.8)
        a.set_ylabel("baseline from the deck / block baseline - 1 (ppm)")
        a.grid(alpha=0.3)
    ax[0].set_xlabel("NL temperature (degC)")
    ax[1].set_xlabel("sol")
    ax[0].legend(fontsize=7, ncol=2)
    fig.suptitle("Navcam baseline against the deck survey, per image pair (rig rotation held)")
    fig.tight_layout()
    f = out / "fiducials_scale.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    files.append(f.name)

    fig, ax = plt.subplots(3, 1, figsize=(11, 9), sharex=True)
    for i, (k, lab) in enumerate((("dR_pitch_mdeg", "pitch"), ("dR_yaw_mdeg", "yaw"), ("dR_roll_mdeg", "roll"))):
        for b in blocks:
            bp = [p for p in use if p["block"] == b and F._num(p.get(k)) is not None]
            if bp:
                ax[i].errorbar([p["sol"] for p in bp], [p[k] for p in bp],
                               yerr=[F._num(p.get(k.replace("_mdeg", "_sd"))) or 0.0 for p in bp],
                               fmt="o", ms=4, color=col[b], alpha=0.7)
        ax[i].axhline(0, color="k", lw=0.8)
        ax[i].set_ylabel(f"{lab}: deck - block rig (mdeg)")
        ax[i].grid(alpha=0.3)
    ax[2].set_xlabel("sol")
    fig.suptitle("Navcam rig rotation seen by the deck, relative to each block's rig")
    fig.tight_layout()
    f = out / "fiducials_rig_rotation.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    files.append(f.name)

    fig, ax = plt.subplots(1, 3, figsize=(14, 4))
    for i, (k, lab) in enumerate((("label_rms_L", "label CAHVORE + label pose"), ("resect_rms_L", "NL alone, pose free"),
                                  ("rig_rms", "pair, block rig held"))):
        v = [F._num(p.get(k)) for p in use]
        v = [x for x in v if x is not None]
        if v:
            ax[i].hist(v, bins=30, color="0.4")
        ax[i].set_title(lab)
        ax[i].set_xlabel("rms (full-resolution px)")
    fig.tight_layout()
    f = out / "fiducials_residuals.png"
    fig.savefig(f, dpi=130)
    plt.close(fig)
    files.append(f.name)
    return files


def report(summary: Dict[str, Any], files: List[str], out: Path, args) -> Path:
    a = summary["all"]

    def line(k, unit):
        r = a.get(k) or {}
        if not r.get("n"):
            return f"| {k} | - | - | - | - |"
        return f"| {k} | {r['n']} | {r['median']:.3g} | {r['mean']:.3g} | {r['se']:.2g} {unit} |"

    L = [f"# Rover deck fiducial check ({datetime.date.today():%d %b %Y})", "",
         f"Blocks: {len(summary['blocks'])}; image pairs with targets: {summary['pairs']}; solved with the block rig held: "
         f"{summary['solved']}.  Model `{args.model}`.  Pixels are full-resolution; scale in ppm of the block's NL-NR baseline.",
         "", "| term | n | median | robust mean | se |", "|---|---|---|---|---|"]
    for k, u in (("label_rms_L", "px"), ("label_rms_R", "px"), ("resect_rms_L", "px"), ("resect_rms_R", "px"),
                 ("rig_rms", "px"), ("scale_ppm", "ppm"), ("tri_scale_ppm", "ppm"), ("tri_shape_rms_mm", "mm"),
                 ("dt_x_mm", "mm"), ("dt_y_mm", "mm"), ("dt_z_mm", "mm"), ("dR_pitch_mdeg", "mdeg"),
                 ("dR_yaw_mdeg", "mdeg"), ("dR_roll_mdeg", "mdeg"), ("bar_ppm", "ppm")):
        L.append(line(k, u))
    sv = summary.get("scale_vs_T")
    if sv:
        L += ["", f"Scale against the NL temperature: {sv['ppm_at_T_mean']:+.0f} ppm at {sv['T_mean']:.1f} degC, "
              f"{sv['ppm_per_degC']:+.1f} +- {sv['sd_ppm_per_degC']:.1f} ppm/degC over {sv['n']} pairs.  "
              f"A deck colder than the room-temperature survey, alone, makes the deck ask for a longer baseline: "
              f"-{sv['deck_cte_ppm_per_degC']:.0f} ppm/degC if the deck follows the camera temperature; a camera bar "
              f"shrinking with the same expansion coefficient cancels it, and a block baseline that does not follow "
              f"temperature then shows as no slope at all."]
    L += ["", "## Per block", "", "| block | pairs | rig rms px | scale ppm | tri scale ppm | roll mdeg | yaw mdeg |",
          "|---|---|---|---|---|---|---|"]
    fmt = lambda v, f: "-" if v is None else format(v, f)                     # noqa: E731
    for b, r in summary["blocks"].items():
        L.append(f"| {b} | {r['solved']} | {fmt(r['rig_rms'], '.2f')} | {fmt(r['scale_ppm'], '+.0f')} | "
                 f"{fmt(r['tri_scale_ppm'], '+.0f')} | {fmt(r['dR_roll_mdeg'], '+.1f')} | {fmt(r['dR_yaw_mdeg'], '+.1f')} |")
    if summary.get("stations"):
        L += ["", "## Rover pose within a station (pairs at the same site/drive)", "",
              "| station | pairs | heading/pitch/roll sd (mdeg) | origin sd x/y/z (mm) |", "|---|---|---|---|"]
        for k, r in summary["stations"].items():
            L.append(f"| {k} | {r['pairs']} | {', '.join(f'{v:.0f}' for v in r['hpr_sd_mdeg'])} | "
                     f"{', '.join(f'{v:.1f}' for v in r['origin_sd_mm'])} |")
    L += ["", "## How to read it", "",
          "* `label_rms`: the label CAHVORE with its own pose (JPL calibration + mast kinematics) against the measurement.",
          "* `resect_rms`: one eye, the block's refined camera, its pose in the rover frame free: "
          "intrinsics + survey + detection.",
          "* `rig_rms`: one NL pose for the pair, NR placed by the block's rig (rotation and translation): "
          "adds the rig at metric scale.",
          "* `scale_ppm`: the rig translation scaled by 1 + k: the baseline the deck asks for, against the block's.",
          "* `dt_*`, `dR_*`: the rig translation, then also its rotation, freed (pitch, yaw, roll about the NR camera x, y, z).",
          "* `tri_scale_ppm`, `tri_shape_rms_mm`: the targets seen by both eyes triangulated with the block's model, "
          "a similarity fit to the survey.",
          "* `bar_ppm`: the LDIFF-RDIFF separation (the differential bar turns; its length does not).",
          "* The survey is taken as a room-temperature one; no deck thermal correction is applied.", ""]
    L += [f"![{f}]({f})" for f in files]
    f = out / "report.md"
    f.write_text("\n".join(L) + "\n", encoding="utf-8")
    return f


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.environ.get("MPPP_FREEZE_ROOT") or os.environ.get("MPPP_SCAPES"),
                    help="the blocks root (default MPPP_FREEZE_ROOT, then MPPP_SCAPES)")
    ap.add_argument("--blocks", nargs="*", default=[], help="block names (full folder names or their ends); default all")
    ap.add_argument("--works", nargs="*", default=[], help="block folders instead of --root/--blocks")
    ap.add_argument("--model", default="cahv_ba", help="the sparse model of each block (default cahv_ba)")
    ap.add_argument("--images-dir", default=None, help="read the images from here instead of each project's images folder")
    ap.add_argument("--out", default=None, help="output folder (default <root>/camera_analysis/fiducials_<yyyymmdd>)")
    ap.add_argument("--figures", type=int, default=3, help="annotated images per block (all six targets found)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.works:
        works = [Path(w) for w in a.works]
        root = works[0].parent
    else:
        if not a.root:
            print("ERROR: give --root (or MPPP_FREEZE_ROOT / MPPP_SCAPES) or --works", file=sys.stderr)
            return 2
        root = Path(a.root)
        works = find_blocks(root, a.model, a.blocks)
    if not works:
        print(f"ERROR: no solved blocks ({a.model}) under {root}", file=sys.stderr)
        return 2
    out = Path(a.out) if a.out else root / "camera_analysis" / f"fiducials_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    pairs, dets = [], []
    t0 = time.time()
    for i, w in enumerate(works, 1):
        print(f"[{i}/{len(works)}] {w.name}", flush=True)
        try:
            r = F.check_block(w, model=a.model, images_dir=a.images_dir, out_dir=out, figures=a.figures,
                              progress=lambda s: None)
        except Exception as exc:                      # noqa: BLE001 - one block never stops the others
            print(f"  FAILED: {exc}", flush=True)
            continue
        solved = [p for p in r["pairs"] if F._num(p.get("rig_rms")) is not None]
        print(f"  {len(r['pairs'])} image pairs with targets, {len(solved)} solved, {len(r['detections'])} detections",
              flush=True)
        pairs += r["pairs"]
        dets += r["detections"]
    F.write_csv(out / "pairs.csv", pairs, ["block"] + [c for c in F.PAIR_COLUMNS if c != "block"])
    F.write_csv(out / "detections.csv", dets, F.DET_COLUMNS)
    summary = F.summarize(pairs)
    summary.update({"root": str(root), "model": a.model, "blocks_checked": [w.name for w in works],
                    "minutes": round((time.time() - t0) / 60, 1)})
    (out / "summary.json").write_text(json.dumps(summary, indent=1, default=float), encoding="utf-8")
    files = plots(pairs, out)
    rep = report(summary, files, out, a)
    s = summary["all"].get("scale_ppm") or {}
    print(f"fiducial_check: {summary['solved']} pairs solved; scale {s.get('mean', float('nan')):+.0f} +- "
          f"{s.get('se', float('nan')):.0f} ppm; report {rep}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

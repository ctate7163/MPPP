"""
Time of day (LMST) of the processed images, per site: one histogram panel per site plus one for all sites, Navcam and
Mastcam-Z 34 mm each normalised to unit area, and a table per site.

    python scripts\\lmst_histogram.py                                   # D:\\scapes\\colmap and D:\\scapes\\colmap_old
    python scripts\\lmst_histogram.py --roots D:\\scapes\\colmap --sites rockytop sid threeforks_south
    python scripts\\lmst_histogram.py --out "D:\\code\\MPPP\\Claude outputs" --name lmst_2026-09-30

Images come from the newest ``processed/mppp_manifest_v*.json`` of every WORK folder (``<site>_colmap``,
``<site>_colmap_nav_zcam34``) under ``--roots``; a folder name found under several roots counts once (its newest
manifest); a site is the union of its folders, each image counted once.  LMST is per image from its label (``LOCAL_MEAN_SOLAR_TIME``).  The dotted lines mark the
processing window ``selection.lmst_window_h`` (images outside it are not processed since v0p30; older manifests
may still hold some).  Navcam tiles below ``--min-frame-fraction`` of the frame (default: the current selection
rule, 1/4) are left out, as processing does since v0p43.3 / v0p44.  Writes ``<out>/<name>.png`` and ``<out>/<name>_by_site.csv``.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

NAV_C, ZC_C = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (validated: CVD dE 24.7, normal 33.6)
FILL_ALPHA = 0.30
XLIM = (5.0, 19.0)
BINS = np.arange(XLIM[0], XLIM[1] + 1e-9, 0.25)


def lmst_hours(s) -> float:
    m = re.search(r"M(\d+):(\d+):([\d.]+)", str(s or ""))
    return int(m[1]) + int(m[2]) / 60 + float(m[3]) / 3600 if m else float("nan")


def _version(p: Path):
    m = re.search(r"mppp_manifest_v(\d+)p(\d+)(?:p(\d+))?", p.name)
    return tuple(int(g or 0) for g in m.groups()) if m else (0, 0, 0)


def newest_manifest(processed: Path):
    fs = sorted(processed.glob("mppp_manifest_v*.json"), key=lambda f: (_version(f), f.stat().st_mtime))
    return fs[-1] if fs else None


def collect(roots, only=None, min_frame_fraction=None):
    """{site: {stem: record}} and [(site, folder, manifest, n images)] from the WORK folders under ``roots``."""
    from mppp.sfm.sites import parse_work_folder
    found = []                                   # (site, folder, manifest file)
    for root in map(Path, roots):
        if not root.is_dir():
            print(f"skipped {root}: not a folder")
            continue
        for d in sorted(p for p in root.iterdir() if p.is_dir()):
            site, _ = parse_work_folder(d)
            if not site or (only and site not in only):
                continue
            m = newest_manifest(d / "processed")
            if m:
                found.append((site, d, m))
    # the same WORK folder under two roots (e.g. colmap and colmap_old): only its newest manifest counts
    best = {}
    for site, d, m in found:
        k = d.name
        if k not in best or (_version(m), m.stat().st_mtime) > (_version(best[k][2]), best[k][2].stat().st_mtime):
            best[k] = (site, d, m)
    found = sorted(best.values(), key=lambda t: (_version(t[2]), t[2].stat().st_mtime))   # newest record wins
    images, used = {}, []
    for site, d, m in found:
        recs = json.loads(m.read_text(encoding="utf-8")).get("images", [])
        n_small = 0
        for r in recs:
            if min_frame_fraction and family(r) == "N" and (frame_fraction(r) or 1.0) < min_frame_fraction:
                n_small += 1                     # a tile the current selection leaves out (manifests before v0p43.3)
                continue
            images.setdefault(site, {})[Path(str(r.get("source_product", ""))).stem] = r
        used.append((site, str(d), m.name, len(recs) - n_small, n_small))
    return images, used


def frame_fraction(r):
    """The fraction of the Navcam frame a manifest record covers (its native size against the full frame at its
    downsampling), as ``select_best_products`` judges it; None if unknown."""
    try:
        s = float((r.get("filename") or {}).get("downsample_scale") or 1.0)
        w, h = r["native_size"]
        return float(w) * float(h) / (5120 * 3840 * s * s)
    except (KeyError, TypeError, ValueError):
        return None


def family(r) -> str:
    fn = r.get("filename") or {}
    return str(fn.get("family") or Path(str(r.get("source_product", ""))).name[:1])


def site_row(name: str, recs) -> dict:
    nav = [r for r in recs if family(r) == "N"]
    zc = [r for r in recs if family(r) == "Z"]
    C = {}
    for r in recs:
        C.setdefault((r.get("site"), r.get("drive")), []).append(r["pose"]["C_enu_m"])
    cent = np.array([np.mean(v, axis=0) for v in C.values()])
    span = float(np.linalg.norm(cent[:, None, :2] - cent[None, :, :2], axis=2).max()) if len(cent) > 1 else 0.0
    lm = np.array([lmst_hours(r.get("LMST")) for r in recs])
    lmN = np.array([lmst_hours(r.get("LMST")) for r in nav])
    sols = sorted({int(r["sol"]) for r in recs if r.get("sol") is not None})
    q = lambda a, p: round(float(np.nanpercentile(a, p)), 2) if np.isfinite(a).any() else None   # noqa: E731
    return {"site": name, "sols": f"{sols[0]}-{sols[-1]}" if sols else "", "first_sol": sols[0] if sols else 0,
            "navcam": len(nav), "mastcam_z34": len(zc), "stations": len(C), "span_m": round(span, 1),
            "lmst_min_h": q(lm, 0), "lmst_median_h": q(lm, 50), "lmst_max_h": q(lm, 100),
            "navcam_lmst_iqr_h": round(q(lmN, 75) - q(lmN, 25), 2) if len(nav) else None}


def draw(images, rows, out_png: Path, window=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mppp.sfm.sites import site_label

    def panel(ax, series, title, bold=False):
        for vals, color in series:
            v = np.asarray([x for x in vals if np.isfinite(x)])
            if v.size:
                ax.hist(v, bins=BINS, density=True, color=color, alpha=FILL_ALPHA)
                ax.hist(v, bins=BINS, density=True, histtype="step", color=color, lw=1.2)
        ax.axvline(12, color="0.35", lw=0.9, ls="--")
        for w in (window or ()):
            ax.axvline(w, color="0.55", lw=0.8, ls=":")
        ax.set_xlim(*XLIM)
        ax.set_yticks([])
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(labelsize=7)
        ax.set_title(title, loc="left", fontsize=7.5, pad=2, fontweight="bold" if bold else "normal")

    def outside(vals):
        v = np.asarray([x for x in vals if np.isfinite(x)])
        return int(np.sum((v < XLIM[0]) | (v > XLIM[1])))

    names = [r["site"] for r in rows]
    ncol = 2 if len(names) > 6 else 1
    nrow = -(-len(names) // ncol)
    fig = plt.figure(figsize=(7.5 * ncol / 1.35 if ncol > 1 else 7.5, 1.05 * nrow + 2.4))
    gs = fig.add_gridspec(nrow + 2, ncol, height_ratios=[1] * nrow + [0.05, 1.7], hspace=0.95, wspace=0.08)
    allN, allZ = [], []
    for k, name in enumerate(names):
        ax = fig.add_subplot(gs[k % nrow if ncol > 1 else k, k // nrow if ncol > 1 else 0])
        recs = list(images[name].values())
        lmN = [lmst_hours(r.get("LMST")) for r in recs if family(r) == "N"]
        lmZ = [lmst_hours(r.get("LMST")) for r in recs if family(r) == "Z"]
        allN += lmN
        allZ += lmZ
        r = rows[k]
        off = outside(lmN) + outside(lmZ)
        panel(ax, [(lmN, NAV_C), (lmZ, ZC_C)],
              f"{site_label(name)}  sols {r['sols']}  {r['navcam']} Nav" + (f" + {r['mastcam_z34']} Z34" if r["mastcam_z34"] else "")
              + (f"  ({off} outside 5-19 h)" if off else ""))
        if (ncol == 1 and k < len(names) - 1) or (ncol > 1 and (k % nrow) < nrow - 1 and k + 1 < len(names)):
            ax.set_xticklabels([])
    ax = fig.add_subplot(gs[nrow + 1, :])
    off = outside(allN) + outside(allZ)
    panel(ax, [(allN, NAV_C), (allZ, ZC_C)],
          f"All {len(names)} sites: {len(allN)} Navcam + {len(allZ)} Mastcam-Z 34 images"
          + (f" ({off} outside 5-19 h not shown)" if off else ""), bold=True)
    ax.set_facecolor("#f6f6f4")
    ax.set_xticks(range(5, 20))
    ax.set_xlabel("local mean solar time of the image [h]", fontsize=8)
    handles = [Patch(facecolor=NAV_C, alpha=FILL_ALPHA, edgecolor=NAV_C, lw=1.2, label="Navcam"),
               Patch(facecolor=ZC_C, alpha=FILL_ALPHA, edgecolor=ZC_C, lw=1.2, label="Mastcam-Z 34"),
               Line2D([], [], color="0.35", lw=0.9, ls="--", label="local noon")]
    if window:
        handles.append(Line2D([], [], color="0.55", lw=0.8, ls=":", label=f"processing window {window[0]:g}-{window[1]:g} h"))
    fig.legend(handles=handles, loc="upper right", fontsize=7.5, frameon=False, ncol=len(handles),
               bbox_to_anchor=(0.99, 0.995))
    fig.suptitle("Time of day of the processed images, per site (each camera normalised to unit area)",
                 fontsize=9.5, x=0.01, ha="left", y=0.995)
    fig.subplots_adjust(left=0.03, right=0.99, top=1 - 0.55 / fig.get_figheight(), bottom=0.5 / fig.get_figheight())
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)
    return len(allN), len(allZ)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roots", nargs="+", default=["D:/scapes/colmap", "D:/scapes/colmap_old"],
                    help="folders holding the WORK folders")
    ap.add_argument("--sites", nargs="+", default=None, help="only these sites")
    ap.add_argument("--out", default=str(ROOT / "Claude outputs"), help="output folder")
    ap.add_argument("--name", default="lmst_histogram", help="file name stem")
    ap.add_argument("--min-frame-fraction", type=float, default=None,
                    help="leave out Navcam tiles below this fraction of the frame, as the current selection does "
                         "(default: select_best_products' NAVCAM_MIN_FRAME_FRACTION; 0: keep all)")
    a = ap.parse_args(argv)
    from mppp.config import default_config
    from mppp.sfm.project import NAVCAM_MIN_FRAME_FRACTION
    mff = NAVCAM_MIN_FRAME_FRACTION if a.min_frame_fraction is None else a.min_frame_fraction
    images, used = collect(a.roots, set(a.sites) if a.sites else None, mff or None)
    if not images:
        print("no processed images found under", a.roots)
        return 1
    rows = sorted((site_row(k, list(v.values())) for k, v in images.items()), key=lambda r: r["first_sol"])
    window = (default_config().get("selection") or {}).get("lmst_window_h")
    out = Path(a.out)
    nN, nZ = draw(images, rows, out / f"{a.name}.png", tuple(window) if window else None)
    with (out / f"{a.name}_by_site.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, [k for k in rows[0] if k != "first_sol"], extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    for site, d, m, n, n_small in used:
        print(f"  {site:22s} {n:5d} images  {m}  {d}" + (f"  ({n_small} tiles below {mff:g} of the frame left out)"
                                                       if n_small else ""))
    print(f"wrote {out / (a.name + '.png')} and {a.name}_by_site.csv: {len(rows)} sites, {nN} Navcam + {nZ} Mastcam-Z")
    return 0


if __name__ == "__main__":
    sys.exit(main())

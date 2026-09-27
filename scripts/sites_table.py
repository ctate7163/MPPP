"""Summary table of the scapes on disk (from their MPPP manifests) and an LMST-of-day histogram per scape."""
import json, glob, re, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

SCAPES = [  # label, folder, manifest, which run
    ("Three Forks", "/home/claude/scapes/threeforks_colmap", "v0p15", "Navcam"),
    ("Three Forks + Z34", "/home/claude/scapes22/threeforks_colmap_nav_zcam34", "v0p22", "Navcam + Mastcam-Z 34"),
    ("Rockytop", "/home/claude/scapes/rockytop_colmap", "v0p15", "Navcam"),
    ("Rockytop + Z34", "/home/claude/scapes/rockytop_colmap_nav_zcam34", "v0p15", "Navcam + Mastcam-Z 34"),
    ("Belva", "/home/claude/scapes/belva_colmap", "v0p15", "Navcam"),
    ("Airey Hill + Z34", "/home/claude/scapes22/aireyhill_colmap_nav_zcam34", "v0p22", "Navcam + Mastcam-Z 34"),
    ("Bell Island", "/home/claude/scapes/bellisland_colmap", "v0p15", "Navcam"),
    ("Bell Island + Z34", "/home/claude/scapes22/bellisland_colmap_nav_zcam34", "v0p22", "Navcam + Mastcam-Z 34"),
    ("Taylor Fjellet", "/home/claude/scapes/taylorfjellet_colmap", "v0p15", "Navcam"),
]

def lmst_hours(s):
    m = re.search(r"M(\d+):(\d+):([\d.]+)", s or "")
    return int(m[1]) + int(m[2]) / 60 + float(m[3]) / 3600 if m else np.nan

rows, hist = [], {}
for label, root, ver, what in SCAPES:
    f = f"{root}/processed/mppp_manifest_{ver}.json"
    d = json.load(open(f))
    im = d["images"]
    fam = np.array([r["filename"]["family"] for r in im])
    nav = [r for r in im if r["filename"]["family"] == "N"]
    zc = [r for r in im if r["filename"]["family"] == "Z"]
    sols = sorted({r["sol"] for r in im})
    stations = sorted({(r["site"], r["drive"]) for r in im})
    sites = sorted({r["site"] for r in im})
    C = {}
    for r in im:
        C.setdefault((r["site"], r["drive"]), []).append(r["pose"]["C_enu_m"])
    cent = np.array([np.mean(v, axis=0) for v in C.values()])
    span = 0.0
    if len(cent) > 1:
        dd = np.linalg.norm(cent[:, None, :2] - cent[None, :, :2], axis=2)
        span = float(dd.max())
    lm = np.array([lmst_hours(r.get("LMST")) for r in im])
    el = np.array([r.get("solar_elevation_deg", np.nan) for r in im], float)
    scale = sorted({r["filename"]["downsample_scale"] for r in nav})
    hist[label] = lm[np.isfinite(lm)]
    rows.append({
        "scape": label, "folder": Path(root).name, "manifest": ver, "cameras": what,
        "site": "-".join(str(s) for s in sites) if len(sites) > 1 else str(sites[0]),
        "sols": f"{sols[0]}–{sols[-1]}" if sols[0] != sols[-1] else str(sols[0]), "n_sols": len(sols),
        "images": len(im), "navcam": len(nav), "navcam_left": sum(r["filename"]["eye"] == "L" for r in nav),
        "mastcam_z": len(zc), "navcam_scales": "/".join(f"{s:g}" for s in scale),
        "stations": len(stations), "span_m": round(span, 1),
        "lmst_min_h": round(float(np.nanmin(lm)), 1), "lmst_max_h": round(float(np.nanmax(lm)), 1),
        "lmst_spread_h": round(float(np.nanmax(lm) - np.nanmin(lm)), 1),
        "lmst_iqr_h": round(float(np.nanpercentile(lm, 75) - np.nanpercentile(lm, 25)), 1),
        "sun_el_min_deg": round(float(np.nanmin(el)), 0), "sun_el_max_deg": round(float(np.nanmax(el)), 0),
    })

out = Path("/home/claude/MPPP_repo/docs/results")
cols = list(rows[0].keys())
with open(out / "sites.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(rows)
# markdown
hdr = ["scape", "cameras", "manifest", "site", "sols", "n_sols", "images", "navcam (left)", "Mastcam-Z", "Navcam scales", "stations",
       "span m", "LMST h", "LMST spread h", "sun el. deg"]
md = ["# Scapes used\n", "Made by `scripts/sites_table.py` from the MPPP manifests of the scape folders on disk (the v0p15 manifests are the "
      "earlier processing of the same image sets; the v0p22 manifests are the runs analysed in notebook 04). Span is the largest "
      "horizontal distance between station centres; stations are (site, drive) pairs. LMST is per image, from the label "
      "(`LOCAL_MEAN_SOLAR_TIME`). Histogram: `sites_lmst.png`.\n",
      "| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
for r in rows:
    md.append("| " + " | ".join([r["scape"], r["cameras"], r["manifest"], r["site"], r["sols"], str(r["n_sols"]), str(r["images"]),
                                 f"{r['navcam']} ({r['navcam_left']})", str(r["mastcam_z"]), r["navcam_scales"], str(r["stations"]),
                                 f"{r['span_m']:g}", f"{r['lmst_min_h']:.1f}–{r['lmst_max_h']:.1f}", f"{r['lmst_spread_h']:.1f}",
                                 f"{r['sun_el_min_deg']:.0f}–{r['sun_el_max_deg']:.0f}"]) + " |")
(out / "sites.md").write_text("\n".join(md) + "\n", encoding="utf-8")
print("\n".join(md))

# histogram: small multiples, one row per scape, LMST 6-20 h in 15-min bins; Navcam vs Mastcam-Z stacked
fig, axes = plt.subplots(len(SCAPES), 1, figsize=(7.5, 1.5 * len(SCAPES)), sharex=True)
bins = np.arange(5, 22.01, 0.25)
for ax, (label, root, ver, what) in zip(axes, SCAPES):
    d = json.load(open(f"{root}/processed/mppp_manifest_{ver}.json"))["images"]
    lmN = [lmst_hours(r.get("LMST")) for r in d if r["filename"]["family"] == "N"]
    lmZ = [lmst_hours(r.get("LMST")) for r in d if r["filename"]["family"] == "Z"]
    ax.hist([lmN, lmZ], bins=bins, stacked=True, color=["#3b6ea5", "#d98b3a"], label=["Navcam", "Mastcam-Z 34"])
    ax.axvline(12, color="0.5", lw=0.6, ls=":")
    r = next(x for x in rows if x["scape"] == label)
    ax.set_title(f"{label}: sols {r['sols']}, {r['stations']} stations / {r['span_m']:g} m, {r['images']} images",
                 loc="left", fontsize=8, pad=2)
    ax.set_ylabel("images", fontsize=8); ax.tick_params(labelsize=8)
axes[0].legend(fontsize=7, loc="upper right")
axes[-1].set_xlabel("local mean solar time of the image [h]")
axes[-1].set_xticks(range(6, 23, 2))
fig.suptitle("When the images were taken: LMST per image, per scape", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.98))
fig.savefig(out / "sites_lmst.png", dpi=160)
print("wrote", out / "sites_lmst.png")

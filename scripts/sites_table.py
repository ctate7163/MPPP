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
      "earlier processing of the same image sets; the v0p22 manifests are the runs analysed in notebook 04, camera models). Span is the largest "
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

# ---------------------------------------------------------------------------------------- per site
# A site = the union of its scapes (Navcam-only and Navcam + Mastcam-Z runs of the same place), each image once.
SITE_OF = {"Three Forks": "Three Forks", "Three Forks + Z34": "Three Forks", "Rockytop": "Rockytop",
           "Rockytop + Z34": "Rockytop", "Belva": "Belva", "Airey Hill + Z34": "Airey Hill", "Bell Island": "Bell Island",
           "Bell Island + Z34": "Bell Island", "Taylor Fjellet": "Taylor Fjellet"}
SITES = ["Three Forks", "Rockytop", "Belva", "Airey Hill", "Bell Island", "Taylor Fjellet"]
site_imgs = {k: {} for k in SITES}
for label, root, ver, what in SCAPES:
    for r in json.load(open(f"{root}/processed/mppp_manifest_{ver}.json"))["images"]:
        site_imgs[SITE_OF[label]][r["filename"]["stem"]] = r
site_rows = []
for name in SITES:
    im = list(site_imgs[name].values())
    nav = [r for r in im if r["filename"]["family"] == "N"]
    zc = [r for r in im if r["filename"]["family"] == "Z"]
    C = {}
    for r in im:
        C.setdefault((r["site"], r["drive"]), []).append(r["pose"]["C_enu_m"])
    cent = np.array([np.mean(v, axis=0) for v in C.values()])
    span = float(np.linalg.norm(cent[:, None, :2] - cent[None, :, :2], axis=2).max()) if len(cent) > 1 else 0.0
    lm = np.array([lmst_hours(r.get("LMST")) for r in im]); el = np.array([r.get("solar_elevation_deg", np.nan) for r in im], float)
    lmN = np.array([lmst_hours(r.get("LMST")) for r in nav])
    sols = sorted({r["sol"] for r in im}); sites = sorted({r["site"] for r in im})
    site_rows.append({"site": name, "rover_sites": "-".join(map(str, sites)), "sols": f"{sols[0]}–{sols[-1]}", "n_sols": len(sols),
                      "scapes": "; ".join(l for l, *_ in SCAPES if SITE_OF[l] == name),
                      "navcam": len(nav), "navcam_left": sum(r["filename"]["eye"] == "L" for r in nav), "mastcam_z34": len(zc),
                      "stations": len(C), "span_m": round(span, 1),
                      "lmst_min_h": round(float(np.nanmin(lm)), 1), "lmst_max_h": round(float(np.nanmax(lm)), 1),
                      "navcam_lmst_iqr_h": round(float(np.nanpercentile(lmN, 75) - np.nanpercentile(lmN, 25)), 1),
                      "sun_el_min_deg": round(float(np.nanmin(el))), "sun_el_max_deg": round(float(np.nanmax(el)))})
with open(out / "sites_by_site.csv", "w", newline="") as f:
    w = csv.DictWriter(f, list(site_rows[0])); w.writeheader(); w.writerows(site_rows)
md2 = ["", "## By site", "", "Each site is the union of its scapes, each image counted once.", "",
       "| site | rover site | sols | sols with images | Navcam (left) | Mastcam-Z 34 | stations | span m | LMST h | Navcam LMST IQR h | sun el. deg |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for r in site_rows:
    md2.append(f"| {r['site']} | {r['rover_sites']} | {r['sols']} | {r['n_sols']} | {r['navcam']} ({r['navcam_left']}) | {r['mastcam_z34']} | "
               f"{r['stations']} | {r['span_m']:g} | {r['lmst_min_h']:.1f}–{r['lmst_max_h']:.1f} | {r['navcam_lmst_iqr_h']:.1f} | "
               f"{r['sun_el_min_deg']}–{r['sun_el_max_deg']} |")
with open(out / "sites.md", "a", encoding="utf-8") as f:
    f.write("\n".join(md2) + "\n")
print("\n".join(md2))

# ---------------------------------------------------------------------------------------- LMST figure
# one panel per site + one for all sites; Navcam and Mastcam-Z each normalised to unit area (densities), overlaid as
# translucent fills with a solid outline, so the two time-of-day distributions compare by shape whatever their counts
NAV_C, ZC_C = "#2a78d6", "#eb6834"          # categorical slots 1 and 2 (dataviz reference palette)
FILL_ALPHA = 0.30
XLIM = (5, 19)
bins = np.arange(XLIM[0], XLIM[1] + 1e-9, 0.25)


def overlaid(ax, series):
    for vals, color, lab in series:
        v = np.asarray([x for x in vals if np.isfinite(x)])
        if v.size:
            ax.hist(v, bins=bins, density=True, color=color, alpha=FILL_ALPHA, label=lab)
            ax.hist(v, bins=bins, density=True, histtype="step", color=color, lw=1.2)
    ax.axvline(12, color="0.35", lw=0.9, ls="--")
    ax.set_xlim(*XLIM)
    ax.grid(axis="y", color="0.9", lw=0.6); ax.set_axisbelow(True)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("density", fontsize=8); ax.tick_params(labelsize=8)
    ax.set_yticks([]); ax.grid(False)          # densities: the shapes compare, the values do not matter


def outside(vals):
    v = np.asarray([x for x in vals if np.isfinite(x)])
    return int(np.sum((v < XLIM[0]) | (v > XLIM[1])))


n = len(SITES) + 1
fig, axes = plt.subplots(n, 1, figsize=(7.5, 1.55 * n + 0.6), sharex=True,
                         gridspec_kw={"height_ratios": [1] * len(SITES) + [1.5]})
allN, allZ = [], []
for ax, name in zip(axes, SITES):
    im = site_imgs[name].values()
    lmN = [lmst_hours(r.get("LMST")) for r in im if r["filename"]["family"] == "N"]
    lmZ = [lmst_hours(r.get("LMST")) for r in im if r["filename"]["family"] == "Z"]
    allN += lmN; allZ += lmZ
    overlaid(ax, [(lmN, NAV_C, "Navcam"), (lmZ, ZC_C, "Mastcam-Z 34")])
    r = next(x for x in site_rows if x["site"] == name)
    off = outside(lmN) + outside(lmZ)
    ax.set_title(f"{name}: sols {r['sols']}, {r['stations']} stations / {r['span_m']:g} m, "
                 f"{r['navcam']} Navcam + {r['mastcam_z34']} Mastcam-Z" + (f" ({off} after 19 h not shown)" if off else ""),
                 loc="left", fontsize=8, pad=2)
ax = axes[-1]
overlaid(ax, [(allN, NAV_C, "Navcam"), (allZ, ZC_C, "Mastcam-Z 34")])
off = outside(allN) + outside(allZ)
ax.set_title(f"All sites: {len(allN)} Navcam + {len(allZ)} Mastcam-Z images" + (f" ({off} after 19 h not shown)" if off else ""),
             loc="left", fontsize=8.5, pad=2, fontweight="bold")
ax.set_facecolor("#f6f6f4")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
fig.legend(handles=[Patch(facecolor=NAV_C, alpha=FILL_ALPHA, edgecolor=NAV_C, lw=1.2, label="Navcam"),
                    Patch(facecolor=ZC_C, alpha=FILL_ALPHA, edgecolor=ZC_C, lw=1.2, label="Mastcam-Z 34"),
                    Line2D([], [], color="0.35", lw=0.9, ls="--", label="local noon")],
           loc="upper right", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.99, 0.997))
axes[-1].set_xlabel("local mean solar time of the image [h]")
axes[-1].set_xticks(range(5, 20, 1))
fig.suptitle("Time of day of the images, per site (each camera normalised to unit area)", fontsize=10, x=0.02, ha="left")
fig.tight_layout(rect=(0, 0, 1, 0.985))
fig.savefig(out / "sites_lmst.png", dpi=160)
print("wrote", out / "sites_lmst.png", "| all sites:", len(allN), "Navcam,", len(allZ), "Mastcam-Z")

"""Tables, figures and report.md of study 3A."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

AXES = ("x", "y", "z")
COOL, WARM = "C0", "C3"                           # left eye / front pair cool, right eye / rear pair warm


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    keys: List[str] = []
    for r in rows:
        keys += [k for k in r if k not in keys and not isinstance(r[k], (dict, list))]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items() if k in keys})


def _fmt(s: Dict[str, Any], key="median", spread="nmad") -> str:
    if not s or not s.get("n"):
        return "-"
    return f"{s[key]:+.1f} ± {s[spread]:.1f}"


def markdown(res: Dict[str, Any], blocks: Dict[str, str]) -> str:
    L = ["# Rover frame (study 3A)", "",
         "Hazcam frames give the rover body pose of their sequence (the Hazcams are fixed to the body); the mast "
         "cameras at the same body pose (same station and RMC POSE count) give the mast-head pose.  Attitudes in "
         "mdeg, positions in mm; median ± NMAD.  Head frame: the label's RSM_HEAD_FRAME; rover frame: x forward, "
         "y right, z down.", "",
         f"Blocks: {len(blocks)}.  Images: {res['n_images']}, used {res['n_used']}" +
         (" (left out: " + ", ".join(f"{v} {k}" for k, v in res["left_out"].items()) + ")" if res["left_out"] else "") + ".", ""]
    st = res["stations"]
    mast = sum(s["mast_frames"] for s in st)
    withb = sum(s["mast_with_body"] for s in st)
    prop = sum(s.get("mast_propagated", 0) for s in st)
    L += [f"Stations: {len(st)}; Hazcam body poses: {len(res['groups'])} "
          f"({sum(g['family'] == 'F' for g in res['groups'])} front, {sum(g['family'] == 'R' for g in res['groups'])} rear); "
          f"mast frames with a Hazcam body pose of the same rover pose (RMC POSE): {withb} of {mast}, with one of "
          f"another pose of the station carried over by the telemetry change: {prop}.", "",
          "Body sources: " + ", ".join(f"{k} {v}" for k, v in (res.get("body_sources") or {}).items()) + ".", ""]
    L += ["## Mounts: the constant part and its scatter (images with a Hazcam body pose)", "",
          "| group | frame | images | blocks | att x | att y | att z | pos x | pos y | pos z |", "|---|---|---|---|---|---|---|---|---|---|"]
    for g, m in res["mounts"].items():
        L.append(f"| {g} | {m['frame']} | {m['images']} | {m['blocks']} | " +
                 " | ".join(_fmt(m["attitude_mdeg"][a]) for a in AXES) + " | " +
                 " | ".join(_fmt(m["position_mm"][a]) for a in AXES) + " |")
    for key, title in (("mounts_propagated", "With the Hazcam body of another rover pose of the station carried over "
                                               "by the telemetry change too:"),
                       ("mounts_telemetry", "With the telemetry attitude where there is no Hazcam pose (the block frame "
                                            "and the IMU errors included):")):
        L += ["", title, "", "| group | images | att x | att y | att z |", "|---|---|---|---|---|"]
        for g, m in (res.get(key) or {}).items():
            L.append(f"| {g} | {m['images']} | " + " | ".join(_fmt(m["attitude_mdeg"][a]) for a in AXES) + " |")
    L += ["", "## Body attitude against the telemetry (per Hazcam pose, rover frame)", "",
          "| block | poses | front x | front y | front z | rear x | rear y | rear z |", "|---|---|---|---|---|---|---|---|"]
    for b in sorted({g["block"] for g in res["groups"]}):
        cells = []
        for fam in "FR":
            gs = [g for g in res["groups"] if g["block"] == b and g["family"] == fam]
            for a in AXES:
                v = [g["omega_tel_mdeg"][a] for g in gs]
                cells.append(f"{np.median(v):+.0f} ({len(v)})" if v else "-")
        L.append(f"| {b} | {sum(g['block'] == b for g in res['groups'])} | " + " | ".join(cells) + " |")
    hr = res.get("heading_by_site_role") or []
    if hr:
        L += ["", "## Heading error and site index changes (front Hazcam body vs telemetry, about the block median)", "",
              "Christian 24 Oct: the last waypoint before a site index change is the more likely to have a rotation "
              "error. Site start = DRIVE 0 (at the previous site's last position).", "",
              "| site role | stations | median |heading| mdeg | p90 | > 1 deg | median |tilt| mdeg |", "|---|---|---|---|---|---|"]
        for r in hr:
            L.append(f"| {r['role']} | {r['stations']} | {r['heading_median_mdeg']:.0f} | {r['heading_p90_mdeg']:.0f} | "
                     f"{r['heading_over_1deg']} | {r['tilt_median_mdeg']:.0f} |")
    ch = res["changes"]
    L += ["", "## Body pose changes between the sequences of one station", ""]
    if ch:
        for lab, sel in (("arm not moved", [c for c in ch if not c["arm_moved"]]), ("arm moved", [c for c in ch if c["arm_moved"]])):
            if not sel:
                continue
            ang = np.median([c["angle_ba_mdeg"] for c in sel])
            miss = np.median([c["missed_mdeg"] for c in sel])
            mv = np.median([c["move_mm"] for c in sel])
            L.append(f"- {lab}: {len(sel)} changes; median adjusted rotation {ang:.0f} mdeg, telemetry missed {miss:.0f} mdeg, "
                     f"origin moved {mv:.1f} mm (median)")
        big = sorted(ch, key=lambda c: -c["missed_mdeg"])[:10]
        L += ["", "Largest changes the telemetry missed:", "",
              "| block | station | pair | POSE | ARM | dt s | adjusted mdeg | telemetry mdeg | missed mdeg | moved mm | bogie L/R mdeg |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for c in big:
            bg = "/".join("-" if c.get(k) is None else f"{c[k]:+.0f}" for k in ("d_left_bogie_mdeg", "d_right_bogie_mdeg"))
            L.append(f"| {c['block']} | {c['station']} | {c['family']} | {c['POSE']} | {c['ARM']} | {c['dt_s']:.0f} | "
                     f"{c['angle_ba_mdeg']:.0f} | {c['angle_tel_mdeg']:.0f} | {c['missed_mdeg']:.0f} | {c['move_mm']:.1f} | {bg} |")
    else:
        L.append("No station with two Hazcam poses.")
    L += ["", "## Pointing law (head-frame attitude residual against the RSM angles and the temperature)", "",
          "Terms: a constant (the mount), cos / sin el (droop, elevation-axis tilt), cos / sin az (bearing "
          "eccentricity, azimuth-axis tilt), T from -20 degC.  Encoder angles; rms before -> after the fit, mdeg.", "",
          "Bodies: the Hazcam pose of the same rover pose or carried over from another pose of the station; "
          "\"measured\": the same rover pose only; \"telemetry\": every frame, the telemetry attitude where there is no "
          "Hazcam pose, one constant per block.", "",
          "| group | images | blocks | x | y | z | measured: x | y | z | per-block constants: x | y | z | resolver angles: x | y | z | telemetry: x | y | z |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for g, f in res["fits"].items():
        p = f["pooled"]
        cell = lambda q, a: f"{q['axes'][a]['rms_before_mdeg']:.1f} -> {q['axes'][a]['rms_after_mdeg']:.1f}" if q.get("axes") else "-"   # noqa: E731
        L.append(f"| {g} | {p['n']} | {p.get('blocks', '-')} | " + " | ".join(
            " | ".join(cell(f[k], a) for a in AXES) for k in ("pooled", "measured", "per_block", "resolver", "telemetry")) + " |")
    for g, f in res["fits"].items():
        p = f["pooled"]
        if not p.get("axes"):
            continue
        L += ["", f"### {g}: coefficients (mdeg ± se; condition number {p['cond']:.0f})", "",
              "| axis | " + " | ".join(p["terms"]) + " |", "|---|" + "---|" * len(p["terms"])]
        for a in AXES:
            q = p["axes"][a]
            L.append(f"| {a} | " + " | ".join(f"{q['coef_mdeg'][t]:+.1f} ± {q['se_mdeg'][t]:.1f}" for t in p["terms"]) + " |")
    L += ["", "Files: `images.csv` (one row per image), `body_poses.csv`, `body_changes.csv`, `stations.csv`, "
          "`rover_frame.json`, figures `*.png`.", ""]
    return "\n".join(L)


def figures(res: Dict[str, Any], out: Path) -> List[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    made = []
    gs = res["groups"]
    if gs:
        fig, axs = plt.subplots(3, 1, figsize=(10, 7), sharex=True)
        order = sorted(range(len(gs)), key=lambda i: (gs[i]["block"], gs[i]["sclk"]))
        for k, a in enumerate(AXES):
            for fam, col, lab in (("F", COOL, "front Hazcams"), ("R", WARM, "rear Hazcams")):
                idx = [j for j, i in enumerate(order) if gs[i]["family"] == fam]
                axs[k].plot(idx, [gs[order[j]]["omega_tel_mdeg"][a] for j in idx], "o", ms=3, color=col, label=lab)
            axs[k].set_ylabel(f"{('roll', 'pitch', 'yaw')[k]} ({a}) mdeg")
            axs[k].axhline(0, color="0.6", lw=0.6)
        prev = None
        for j, i in enumerate(order):
            if gs[i]["block"] != prev:
                for ax in axs:
                    ax.axvline(j - 0.5, color="0.85", lw=0.6)
                axs[0].text(j, axs[0].get_ylim()[1], gs[i]["block"], fontsize=6, rotation=90, va="top")
                prev = gs[i]["block"]
        axs[0].legend(loc="lower right", fontsize=7)
        axs[-1].set_xlabel("Hazcam body pose (blocks in order, poses in time)")
        fig.suptitle("Rover body attitude from the Hazcams minus the telemetry attitude (rover frame)")
        fig.tight_layout()
        p = out / "body_attitude.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(p)
    rows = [r for r in res["rows"] if r["family"] in "NZ" and r["body"] != "telemetry" and "e_head_x_mdeg" in r]
    tel = not rows                                 # no Hazcam body anywhere: the telemetry attitude
    if tel:
        rows = [r for r in res["rows"] if r["family"] in "NZ" and "e_head_x_mdeg" in r]
    fams = sorted({r["group"][2:] if r["family"] == "Z" else "" for r in rows})
    for zoom in fams:
        gl, gr = (f"ZL{zoom}", f"ZR{zoom}") if zoom else ("NL", "NR")
        sel = {g: [r for r in rows if r["group"] == g and r.get("el_encoder") is not None] for g in (gl, gr)}
        if sum(map(len, sel.values())) < 12:
            continue
        fig, axs = plt.subplots(3, 4, figsize=(12, 7.5), sharey="row")
        for k, a in enumerate(AXES):
            for c, (g, ang, col) in enumerate(((gl, "el", COOL), (gr, "el", WARM), (gl, "az", COOL), (gr, "az", WARM))):
                rs = sel[g]
                if not rs:
                    axs[k, c].set_axis_off()                    # no frames of that eye
                    if k == 0:
                        axs[k, c].set_title(f"{g}: no frames")
                    continue
                x = np.degrees([r[f"{ang}_encoder"] for r in rs])
                axs[k, c].plot(x, [r[f"e_head_{a}_mdeg"] for r in rs], ".", ms=3, color=col)
                axs[k, c].axhline(0, color="0.6", lw=0.6)
                if k == 0:
                    axs[k, c].set_title(f"{g} vs {'elevation' if ang == 'el' else 'azimuth'}")
                if k == 2:
                    axs[k, c].set_xlabel(f"RSM {'elevation' if ang == 'el' else 'azimuth'} (deg, encoder)")
            axs[k, 0].set_ylabel(f"head {a} (mdeg)")
        fig.suptitle(f"{'Mastcam-Z ' + zoom + ' mm' if zoom else 'Navcam'}: attitude relative to the "
                     f"{'telemetry' if tel else 'Hazcam'} body, head frame (label kinematics removed)")
        fig.tight_layout()
        p = out / f"pointing_{'zcam' + zoom if zoom else 'navcam'}.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(p)
    ch = res["changes"]
    if ch:
        fig, axs = plt.subplots(1, 3, figsize=(11, 3.8))
        for k, a in enumerate(AXES):
            for moved, col, lab in ((False, COOL, "arm not moved"), (True, WARM, "arm moved")):
                cs = [c for c in ch if c["arm_moved"] == moved]
                axs[k].plot([c[f"tel_{a}_mdeg"] for c in cs], [c[f"ba_{a}_mdeg"] for c in cs], "o", ms=3, color=col, label=lab)
            lim = max([abs(c[f"tel_{a}_mdeg"]) for c in ch] + [abs(c[f"ba_{a}_mdeg"]) for c in ch] + [1.0])
            axs[k].plot([-lim, lim], [-lim, lim], "--", color="0.6", lw=0.8)
            axs[k].set_xlabel(f"telemetry change {a} (mdeg)")
            axs[k].set_ylabel(f"adjusted change {a} (mdeg)")
        axs[0].legend(fontsize=7)
        fig.suptitle("Body pose changes between the Hazcam sequences of one station")
        fig.tight_layout()
        p = out / "body_changes.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(p)
    rel = [r for r in res["rows"] if r["body"] != "telemetry" and not r["body"].endswith("_propagated")]
    groups = sorted({r["group"] for r in rel})
    if groups:
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
        for k, a in enumerate(AXES):
            data, labs, cols = [], [], []
            for g in groups:
                key = f"d_head_{a}_mm" if g[0] in "NZ" else f"d_rnav_{a}_mm"
                v = [r[key] for r in rel if r["group"] == g and r.get(key) is not None]
                if v:
                    data.append(np.asarray(v) - np.median(v))
                    labs.append(g)
                    cols.append(COOL if g[1] == "L" else WARM)
            if data:
                bp = axs[k].boxplot(data, vert=False, showfliers=False, patch_artist=True)
                for patch, c in zip(bp["boxes"], cols):
                    patch.set_facecolor(c)
                    patch.set_alpha(0.5)
                axs[k].set_yticks(range(1, len(labs) + 1), labs)
            axs[k].set_xlabel(f"{a} minus the median (mm)")
        fig.suptitle("Camera positions in the body frame (Hazcams) and the head frame (mast cameras): the scatter")
        fig.tight_layout()
        p = out / "position_scatter.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        made.append(p)
    return made


def write_all(res: Dict[str, Any], out: Path, blocks: Dict[str, str], plots: bool = True) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "images.csv", res["rows"])
    write_csv(out / "body_poses.csv", [dict(g, **{f"omega_tel_{a}_mdeg": g["omega_tel_mdeg"][a] for a in AXES})
                                       for g in res["groups"]])
    write_csv(out / "body_changes.csv", res["changes"])
    write_csv(out / "stations.csv", res["stations"])
    js = {k: v for k, v in res.items() if k not in ("rows",) and not k.startswith("_")}
    js["blocks"] = blocks
    (out / "rover_frame.json").write_text(json.dumps(js, indent=1, default=str), encoding="utf-8")
    made = figures(res, out) if plots else []
    (out / "report.md").write_text(markdown(res, blocks), encoding="utf-8")
    return {"report": str(out / "report.md"), "figures": [str(p) for p in made]}

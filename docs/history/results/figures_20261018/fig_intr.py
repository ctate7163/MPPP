"""Navcam intrinsic temperature dependence after pass 2 (18 Oct 2026). Columns NL (cool) / NR (warm); rows focal length
and principal point, as the change from T0 = -20 degC; left/right share y range and ticks. Points: the per-temperature-
bin cameras of the 23 v0.82 block runs (made with the cameras in use, v0p73 form: dotted lines), solid: pass-2 model."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
P2 = "/mnt/user-data/uploads/colmap/camera_analysis/navcal_v0p82_20261018_pipeline_pass2/navcam_joint/"
cons = json.load(open(P2 + "consensus.json")); rr = json.load(open("/home/claude/rr/rig_residuals_pass2.json"))
inuse = {e: json.load(open(f"/home/claude/MPPP_repo/src/mppp/data/cmods/M2020_{e}_fisheye_tangential.json")) for e in ("NL", "NR")}
bins = json.load(open("/home/claude/rr/thermal_bins_pass2.json"))
sols = {}
for b in rr["blocks"]:
    sols.setdefault(b["site"], []).append(b["sol_median"])
early_blk = {k for k, v in sols.items() if min(v) < 380}
T0 = -20.0; IFOV_MDEG = 18.9
ax_ppm = cons["thermal"]["axis_ppm"]; pp = cons["thermal"]["pp_slopes"]
EYE = {"NL": "C0", "NR": "C3"}
yawT = rr["fits"]["yaw"]["vs_T"]
CX2 = pp["NR"][0]; CX_ALL = CX2 + yawT["slope"] / IFOV_MDEG; CX_ALL_SE = yawT["se"] / IFOV_MDEG
B = rr["blocks"]; Tblk = np.array([b["T_median_degC"] for b in B]); yaw = np.array([b["yaw_mdeg"] for b in B])
tt = np.linspace(-60, 0, 121)
fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6), sharex=True)
for j, eye in enumerate(("NL", "NR")):
    col = EYE[eye]; f2 = cons[eye]; iu = inuse[eye]; th_iu = iu["thermal"]; p_iu = iu["params"]
    rows = [r for r in bins if r["eye"] == eye]
    ax = axs[0][j]
    for k, (p, ls, mk, alpha) in enumerate((("fx", "-", "o", 0.8), ("fy", "--", "s", 0.35))):
        ppm2 = ax_ppm[eye][k]; ppm_iu = th_iu[f"{p}_ppm_per_degC"]
        ax.plot(tt, f2[p] * 1e-6 * ppm2 * (tt - T0), ls, color=col, lw=1.4, label=f"{p} pass 2: {ppm2:+.1f} ppm/°C ({f2[p] * 1e-6 * ppm2:+.4f} px/°C)")
        if k == 0:
            ax.plot(tt, p_iu[k] * 1e-6 * ppm_iu * (tt - T0), ":", color="0.4", lw=1.1, label=f"fx in use for the block runs: {ppm_iu:+.1f} ppm/°C")
        for early, mfc in ((False, None), (True, "white")):
            rs = [r for r in rows if (r["block"] in early_blk) == early]
            if not rs:
                continue
            Tb = np.array([r["T"] for r in rs]); nb = np.array([r["n"] for r in rs])
            v = np.array([r[p] for r in rs]) - p_iu[k]
            ax.scatter(Tb, v, s=6 + 1.5 * np.sqrt(nb * 10), marker=mk, facecolors=mfc or col, edgecolors=col, alpha=alpha, zorder=3,
                       label=(f"{p} per temperature bin of the block runs (size ~ images)" if not early else f"{p}, blocks before sol 380 (early-mission offset not removed)") if p == "fx" else None)
    ax.set_title(f"{eye}: focal length", fontsize=10)
    ax.set_ylabel(f"Δf from {T0:.0f} °C (full-frame px)", fontsize=8)
    ax.secondary_yaxis("right", functions=(lambda d, f=f2["fx"]: 1e6 * d / f, lambda q, f=f2["fx"]: q * f / 1e6)).set_ylabel("ppm of f", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper left")
    ax = axs[1][j]
    if eye == "NL":
        ax.plot(tt, 0 * tt, "-", color=col, lw=1.3, label="cx, cy: 0 (NL reference; pass 3 profiles NL cx)")
    else:
        ax.plot(tt, -0.0714 * (tt - T0), ":", color="0.4", lw=1.2, label="cx in use for the block runs: -0.0714 px/°C")
        ax.plot(tt, CX2 * (tt - T0), "--", color=col, lw=1.2, label=f"cx pass 2: {CX2:+.4f} px/°C")
        ax.fill_between(tt, (CX_ALL - CX_ALL_SE) * (tt - T0), (CX_ALL + CX_ALL_SE) * (tt - T0), color=col, alpha=0.12, lw=0)
        ax.plot(tt, CX_ALL * (tt - T0), "-", color=col, lw=1.6, label=f"cx with the yaw residual folded in: {CX_ALL:+.4f} ± {CX_ALL_SE:.4f} px/°C")
        ax.plot(tt, 0 * tt, "-.", color=col, lw=0.8, label="cy: 0")
        y = CX2 * (Tblk - T0) + yaw / IFOV_MDEG
        ax.plot(Tblk, y, "o", color=col, ms=5, mec="white", mew=0.5, zorder=3, label="rig epoch stereo pointing as NR cx (pass-2 cx + yaw residual)")
    ax.set_title(f"{eye}: principal point", fontsize=10)
    ax.set_ylabel(f"Δ principal point from {T0:.0f} °C (full-frame px)", fontsize=8)
    ax.set_xlabel("Navcam camera temperature (°C)", fontsize=9)
    ax.secondary_yaxis("right", functions=(lambda d: d * IFOV_MDEG, lambda m: m / IFOV_MDEG)).set_ylabel("pointing equivalent (mdeg; 1 px = 18.9 mdeg)", fontsize=8)
    ax.legend(fontsize=6.5, loc="upper right" if eye == "NR" else "upper left")
Tb_all = np.array([r["T"] for r in bins])
for r in range(2):
    lo = min(axs[r][0].get_ylim()[0], axs[r][1].get_ylim()[0]); hi = max(axs[r][0].get_ylim()[1], axs[r][1].get_ylim()[1])
    for a in axs[r]:
        a.set_ylim(lo, hi)
    axs[r][1].set_yticks(axs[r][0].get_yticks()); axs[r][1].set_ylim(lo, hi)
for a in axs.flat:
    a.axvline(T0, color="0.4", ls="--", lw=0.8); a.axhline(0, color="0.6", lw=0.6)
    a.axvspan(Tb_all.min(), Tb_all.max(), color="0.85", alpha=0.35, lw=0, zorder=0); a.grid(alpha=0.3); a.set_xlim(-60, 0)
fig.suptitle(f"Navcam intrinsic temperature dependence after pass 2 (23 blocks, 18 Oct 2026; T0 = {T0:.0f} °C; grey band: temperatures of the bins)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig("/mnt/user-data/outputs/navcam_intrinsics_vs_temperature_pass2.png", dpi=140)
for eye in ("NL", "NR"):
    rs = [r for r in bins if r["eye"] == eye and r["block"] not in early_blk]
    T = np.array([r["T"] for r in rs]); w = np.array([r["n"] for r in rs], float)
    for k, p in enumerate(("fx", "fy")):
        v = np.array([r[p] for r in rs]); c = np.polyfit(T, v, 1, w=np.sqrt(w))
        print(eye, p, "bins slope ppm/degC (late blocks, weighted):", round(1e6 * c[0] / v.mean(), 1), "n bins", len(rs))

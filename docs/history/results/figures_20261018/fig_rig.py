"""Navcam rig trends after pass 2 (18 Oct 2026): pitch (UL), yaw with its temperature slope removed (UR), roll (LL)
against sol, with the pass-2 drift model; NR cx against temperature with the yaw residual folded in (LR).
Rig residuals: each block (rig epochs split as in v0p82) with the pass-2 candidate cameras held, rig rotation free."""
import json, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
P2 = "/mnt/user-data/uploads/colmap/camera_analysis/navcal_v0p82_20261018_pipeline_pass2/navcam_joint/"
rr = json.load(open("/home/claude/rr/rig_residuals_pass2.json"))
cons = json.load(open(P2 + "consensus.json"))
drift = json.load(open(P2 + "M2020_N_rig.json"))["drift"]
B = rr["blocks"]; names = [b["block"] for b in B]
sol = np.array([b["sol_median"] for b in B], float); T = np.array([b["T_median_degC"] for b in B], float)
T0 = -20.0; IFOV_MDEG = 18.9
CX_T = cons["thermal"]["pp_slopes"]["NR"][0]; CX0 = cons["NR"]["cx"]

def off(a, s):
    rate = drift.get(f"{a}_mdeg_per_sol") or 0.0
    v = rate * (np.asarray(s, float) - drift["sol0"])
    if drift.get(f"{a}_early_mdeg_per_sol") is not None and drift.get("knot_sol"):
        v = v + (drift[f"{a}_early_mdeg_per_sol"] - rate) * (np.minimum(s, drift["knot_sol"]) - drift["knot_ref"])
    return v

def annotate(ax, x, y, nm):
    for xx, yy, n in zip(x, y, nm):
        ax.annotate(n.replace(" [sols ", " [")[:22], (xx, yy), fontsize=5.5, alpha=0.6, xytext=(3, 3), textcoords="offset points")

fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.2))
ss = np.linspace(20, 2000, 400)
panels = {"pitch": axs[0][0], "yaw": axs[0][1], "roll": axs[1][0]}
yaw_T = rr["fits"]["yaw"]["vs_T"]
for ang, ax in panels.items():
    tot = np.array([b[f"{ang}_mdeg"] for b in B]) + off(ang, sol)
    if ang == "yaw":
        tot = tot - yaw_T["slope"] * (T - T0)
        lab = f"block rig epoch, {yaw_T['slope']:+.3f} ± {yaw_T['se']:.3f} mdeg/°C removed (to {T0:.0f} °C)"
    else:
        lab = "block rig epoch (rig rotation free, cameras held)"
    ax.plot(sol, tot, "o", color="C0", label=lab)
    ax.plot(ss, off(ang, ss), "-", color="C3", label="pass-2 drift model")
    annotate(ax, sol, tot, names)
    r = tot - off(ang, sol)
    rate = 1000 * (drift.get(f"{ang}_mdeg_per_sol") or 0.0)
    f = rr["fits"][ang]["vs_sol"]
    extra = f"; {1000 * drift['pitch_early_mdeg_per_sol']:+.1f} before sol {drift['knot_sol']:.0f}" if ang == "pitch" and drift.get("pitch_early_mdeg_per_sol") is not None else ""
    ax.set_title(f"{ang}: drift {rate:+.2f} mdeg/1000 sol{extra}; residual trend {f['slope']:+.2f} ± {f['se']:.2f} (z {f['z']:+.1f})\n"
                 f"epoch sd about the model {r.std(ddof=1):.2f} mdeg = {r.std(ddof=1) / IFOV_MDEG:.3f} px", fontsize=9)
    ax.axhline(0, color="0.6", lw=0.8); ax.axvline(drift["sol0"], color="0.4", ls="--", lw=0.8)
    if drift.get("knot_sol"):
        ax.axvline(drift["knot_sol"], color="0.4", ls="--", lw=0.8)
    ax.set_xlabel("median sol of the rig epoch", fontsize=8)
    ax.set_ylabel(f"{ang} relative to the joint rig at sol {drift['sol0']:.0f} (mdeg)", fontsize=8)
    ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower right")
lo = min(a.get_ylim()[0] for a in panels.values()); hi = max(a.get_ylim()[1] for a in panels.values())
for a in panels.values():
    a.set_ylim(lo, hi); a.set_xlim(0, 2050)
    a.secondary_yaxis("right", functions=(lambda m: m / IFOV_MDEG, lambda p: p * IFOV_MDEG)).set_ylabel("Navcam IFOV equivalent (px)", fontsize=8)
ax = axs[1][1]
yaw_px = np.array([b["yaw_mdeg"] for b in B]) / IFOV_MDEG
cx_blk = CX0 + CX_T * (T - T0) + yaw_px
corr = CX_T + yaw_T["slope"] / IFOV_MDEG
tt = np.linspace(np.nanmin(T) - 2, np.nanmax(T) + 2, 2)
ax.plot(T, cx_blk, "o", color="C3", label="epoch stereo pointing as NR cx (model + rig-yaw residual)")
ax.plot(tt, CX0 + CX_T * (tt - T0), "--", color="0.4", label=f"pass-2 model: {CX_T:+.4f} px/°C (per image)")
ax.plot(tt, CX0 - 0.0714 * (tt - T0), ":", color="0.6", label="in use for these blocks: -0.0714 px/°C")
ax.fill_between(tt, CX0 + (corr - yaw_T["se"] / IFOV_MDEG) * (tt - T0), CX0 + (corr + yaw_T["se"] / IFOV_MDEG) * (tt - T0), color="C3", alpha=0.12, lw=0)
ax.plot(tt, CX0 + corr * (tt - T0), "-", color="C3", label=f"with the yaw residual folded in: {corr:+.4f} ± {yaw_T['se'] / IFOV_MDEG:.4f} px/°C")
annotate(ax, T, cx_blk, names)
ax.axvline(T0, color="0.4", ls="--", lw=0.8)
ax.set_xlabel("epoch median Navcam temperature (°C)", fontsize=8); ax.set_ylabel("NR cx (px)", fontsize=8)
ax.set_title(f"NR cx against temperature: no yaw temperature term left\n(scatter about the line {np.std(cx_blk - (CX0 + corr * (T - T0)), ddof=2):.3f} px)", fontsize=9)
ax.grid(alpha=0.3); ax.legend(fontsize=7, loc="lower left")
ax.secondary_yaxis("right", functions=(lambda p: (p - CX0) * IFOV_MDEG, lambda m: CX0 + m / IFOV_MDEG)).set_ylabel(f"yaw equivalent about cx at {T0:.0f} °C (mdeg)", fontsize=8)
fig.suptitle(f"Navcam rig trends after pass 2 (23 blocks, {len(B)} rig epochs, 18 Oct 2026; right axes in Navcam IFOV, 0.33 mrad = {IFOV_MDEG} mdeg)", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.97)); fig.savefig("/mnt/user-data/outputs/navcam_rig_trends_pass2.png", dpi=130)
print("yaw vs T", yaw_T, "corr", corr)

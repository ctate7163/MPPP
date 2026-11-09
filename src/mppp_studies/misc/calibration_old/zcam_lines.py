"""
Mastcam-Z focus lines across blocks (MPPP v0p73, notebook 04 section 5c).

Notebook 03 (stage 2, :func:`mppp.sfm.zcam.refit_param_lines`) puts every focus-bin camera of a block on lines in
focus count, one per eye, zoom, focus (backlash) state and parameter: p = a + b (focus - reference) for fx, fy (both
eyes) and cx, cy (ZR - the R-L boresight; ZL is the reference eye).  This module collects those lines from the solved
blocks (:func:`block_lines`), with each block's Mastcam-Z lens temperature (the mean of the optics sensors HEAD_HTR_1
and HEAD_HTR_2 over its Mastcam-Z images, :func:`zcam_block_temperature`, separate from the Navcam temperature), and
asks how well one line describes all blocks (:func:`lines_consensus`): the weighted mean offset and slope, their
scatter between blocks (chi^2 / dof, tau) and their trend with the lens temperature.  :func:`plot_block_lines` draws
offset and slope against temperature.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np

PathLike = Union[str, Path]
PARAMS = ("fx", "fy", "cx", "cy")


def navcam_scale(sol) -> Optional[float]:
    """The block's Navcam focal scale, refined / start sqrt(fx fy) weighted by observations (the Mastcam-Z focal
    lengths inherit it through the shared points); None without Navcam cameras."""
    num = den = 0.0
    for c in sol.cameras.values():
        if c.family != "N" or not c.n_obs:
            continue
        r = float(np.sqrt(c.params[0] * c.params[1]) / np.sqrt(c.initial[0] * c.initial[1]))
        num += c.n_obs * r
        den += c.n_obs
    return num / den if den else None


def zcam_block_temperature(sol, pds_dir: Optional[PathLike] = None) -> Dict[str, Optional[float]]:
    """
    The block's Mastcam-Z temperatures: ``optics_degC`` - the mean over its Mastcam-Z images of the lens temperature
    (HEAD_HTR_1, HEAD_HTR_2: project rows, else the manifest's ``instrument_temperatures``, else the labels under
    ``pds_dir``), ``fpa_degC`` (HEAD_FPA, for comparison), ``navcam_degC`` (the Navcam camera temperature), and the
    number of Mastcam-Z images with an optics temperature.
    """
    from mppp.image import zcam_optics_temperature
    opt, fpa, nav, missing = [], [], [], []
    for nm, r in sol.images.items():
        fam = str(r.get("instrument", ""))[:1]
        if fam == "N" and r.get("camera_temperature_degC") is not None:
            nav.append(float(r["camera_temperature_degC"]))
        if fam != "Z":
            continue
        if r.get("camera_temperature_degC") is not None:
            fpa.append(float(r["camera_temperature_degC"]))
        t = r.get("zcam_optics_temperature_degC")
        if t is None:
            m = sol.manifest.get(Path(nm).stem) or {}
            t = m.get("zcam_optics_temperature_degC") or zcam_optics_temperature(m.get("instrument_temperatures"))
        if t is None:
            missing.append(Path(nm).stem)
        else:
            opt.append(float(t))
    if missing and pds_dir:
        from mppp.select import iter_imgs
        from mppp.labels import instrument_temperatures as _it, read_pds as _rp
        todo = set(missing)
        for fp, fn in iter_imgs(pds_dir):
            if fn.stem in todo:
                todo.discard(fn.stem)
                t = zcam_optics_temperature(_it(_rp(fp, load_image=False)[0]))
                if t is not None:
                    opt.append(float(t))
            if not todo:
                break
    med = lambda v: float(np.mean(v)) if v else None          # noqa: E731
    return {"optics_degC": med(opt), "fpa_degC": med(fpa), "navcam_degC": med(nav), "n_optics": len(opt),
            "optics_sd_degC": float(np.std(opt)) if len(opt) > 1 else None}


def block_lines(sols: Dict[str, Any], pds_dir: Optional[PathLike] = None, normalise_navcam: bool = True) -> List[Dict[str, Any]]:
    """
    One row per block, eye and zoom, focus state and parameter of the solved blocks ``sols`` ({name: Solution}) that
    have v0p73 focus lines (``project.settings["zcam_focus_lines"][group]["lines"]``): a, b (per focus count, about
    the group's reference focus), their sd, the number of bins, and the block's sol, temperatures and Navcam scale.
    ``normalise_navcam``: fx, fy (offset and slope) are divided by the block's Navcam focal scale (audit item 2).
    """
    rows = []
    for name, s in sols.items():
        lines = ((s.project.get("settings") or {}).get("zcam_focus_lines") or {})
        if not any(isinstance(v, dict) and v.get("lines") for v in lines.values()):
            continue
        T = zcam_block_temperature(s, pds_dir)
        ns = navcam_scale(s) if normalise_navcam else None
        sols_z = [float(r["sol"]) for r in s.images.values() if str(r.get("instrument", ""))[:1] == "Z" and r.get("sol") is not None]
        for g, r in lines.items():
            for st, ln in (r.get("lines") or {}).items():
                if not isinstance(ln, dict) or "skipped" in ln:
                    continue
                for pn, e in ln.items():
                    k = (ns or 1.0) if pn in ("fx", "fy") else 1.0
                    rows.append({"block": name, "group": g, "eye": g[:2], "zoom": int(g[2:5]) if g[2:5].isdigit() else None,
                                 "state": st, "param": pn, "reference_focus": r.get("reference_focus"),
                                 "a": float(e["a"]) / k, "b": float(e["b"]) / k,
                                 "sd_a": (float(e["sd_a"]) / k) if e.get("sd_a") is not None else None,
                                 "sd_b": (float(e["sd_b"]) / k) if e.get("sd_b") is not None else None,
                                 "n_bins": e.get("n_bins"), "slope_source": e.get("slope_source"), "rms": e.get("rms"),
                                 "focus_range": e.get("focus_range"), "navcam_scale": ns,
                                 "sol": float(np.median(sols_z)) if sols_z else None,
                                 "T_optics_degC": T["optics_degC"], "T_fpa_degC": T["fpa_degC"],
                                 "T_navcam_degC": T["navcam_degC"]})
    return rows


def _meta(y: np.ndarray, sd: np.ndarray, x: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """Weighted mean (or line in x) with the between-block scatter tau (DerSimonian-Laird) and chi^2 / dof."""
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nanmedian(sd[np.isfinite(sd) & (sd > 0)]) if np.any(np.isfinite(sd) & (sd > 0)) else 1.0)
    w = 1.0 / sd ** 2
    X = np.ones((len(y), 1)) if x is None else np.c_[np.ones(len(y)), x]
    p = X.shape[1]
    beta = np.linalg.lstsq(X * np.sqrt(w)[:, None], y * np.sqrt(w), rcond=None)[0]
    r = y - X @ beta
    Q = float(np.sum(w * r ** 2))
    dof = max(len(y) - p, 1)
    c = float(np.sum(w) - (np.sum(w ** 2) / np.sum(w) if p == 1 else 0.0))
    tau2 = max(0.0, (Q - dof) / c) if (p == 1 and c > 0) else max(0.0, (Q / dof - 1.0) * float(np.mean(sd ** 2)))
    w2 = 1.0 / (sd ** 2 + tau2)
    beta2 = np.linalg.lstsq(X * np.sqrt(w2)[:, None], y * np.sqrt(w2), rcond=None)[0]
    try:
        cov = np.linalg.inv(X.T @ (X * w2[:, None]))
    except np.linalg.LinAlgError:
        cov = np.full((p, p), np.nan)
    out = {"n": int(len(y)), "chi2_dof": Q / dof, "tau": float(np.sqrt(tau2)), "value": float(beta2[0]),
           "se": float(np.sqrt(max(cov[0, 0], 0.0)))}
    if p == 2:
        out.update({"slope": float(beta2[1]), "slope_se": float(np.sqrt(max(cov[1, 1], 0.0)))})
        out["slope_z"] = out["slope"] / out["slope_se"] if out["slope_se"] > 0 else None
    return out


def lines_consensus(rows: Sequence[Dict[str, Any]], blocks: Optional[Sequence[str]] = None,
                    temperature: str = "T_optics_degC") -> List[Dict[str, Any]]:
    """
    How well one line describes the blocks (``blocks``: e.g. two or three robust sites; None = all): per eye and
    zoom, state and parameter, the offset a and the slope b over the blocks - weighted mean, its se, the scatter
    between blocks (``tau``, ``chi2_dof``: 1 = the blocks agree within their sd) - and each against the block lens
    temperature (``temperature``: ``T_optics_degC`` (default), ``T_fpa_degC`` or ``T_navcam_degC``): slope per degC
    with its se and z.  The offsets are at the group's reference focus (the same in every block for one focus model).
    """
    sel = [r for r in rows if blocks is None or r["block"] in set(blocks)]
    keys = sorted({(r["group"], r["state"], r["param"]) for r in sel})
    out = []
    for g, st, pn in keys:
        rr = [r for r in sel if (r["group"], r["state"], r["param"]) == (g, st, pn)]
        ent: Dict[str, Any] = {"group": g, "state": st, "param": pn, "blocks": [r["block"] for r in rr], "n": len(rr)}
        for k, sdk in (("a", "sd_a"), ("b", "sd_b")):
            y = np.array([r[k] for r in rr], float)
            sd = np.array([np.nan if r.get(sdk) is None else r[sdk] for r in rr], float)
            if len(rr) >= 2:
                ent[k] = _meta(y, sd)
                xT = np.array([np.nan if r.get(temperature) is None else r[temperature] for r in rr], float)
                ok = np.isfinite(xT)
                if ok.sum() >= 3 and np.ptp(xT[ok]) > 0:
                    m = _meta(y[ok], sd[ok], xT[ok] - float(np.mean(xT[ok])))
                    ent[f"{k}_vs_T"] = {"slope_per_degC": m["slope"], "se": m["slope_se"], "z": m.get("slope_z"),
                                        "tau": m["tau"], "chi2_dof": m["chi2_dof"], "T_mean": float(np.mean(xT[ok])),
                                        "temperature": temperature}
            else:
                ent[k] = {"n": len(rr), "value": float(y[0]) if len(y) else None}
        out.append(ent)
    return out


def consensus_lines(cons: Sequence[Dict[str, Any]]) -> List[str]:
    """Printable summary of :func:`lines_consensus`."""
    out = []
    for e in cons:
        a, b = e.get("a") or {}, e.get("b") or {}
        s = f"{e['group']} {e['state']:8s} {e['param']}: n {e['n']}"
        if "se" in a:
            s += (f"; a {a['value']:.2f} +- {a['se']:.2f} (tau {a['tau']:.2f}, chi2/dof {a['chi2_dof']:.1f})")
        if "se" in b:
            s += f"; b {b['value']:+.5f} +- {b['se']:.5f}/count (tau {b['tau']:.5f})"
        t = e.get("a_vs_T")
        if t:
            s += f"; a vs T {t['slope_per_degC']:+.3f} +- {t['se']:.3f} px/degC (z {t['z']:+.1f})" if t.get("z") is not None else ""
        out.append(s)
    return out


def plot_block_lines(rows: Sequence[Dict[str, Any]], out: Optional[PathLike] = None, temperature: str = "T_optics_degC",
                     figsize=(13, 9), dpi: int = 200, point_labels: bool = True):
    """Offset a (top) and slope b (bottom) of every focus line against the block lens temperature, one column per
    parameter, one colour per eye and zoom, filled for the backlash state, open for the regular state.  Panels
    lettered A, B ...; ``point_labels=False`` leaves out the block name at every point."""
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, len(PARAMS), figsize=figsize, squeeze=False)
    groups = sorted({r["group"] for r in rows})
    colours = {g: f"C{i}" for i, g in enumerate(groups)}
    for j, pn in enumerate(PARAMS):
        for i, k in enumerate(("a", "b")):
            ax = axs[i][j]
            for r in rows:
                if r["param"] != pn or r.get(temperature) is None:
                    continue
                sd = r.get("sd_a" if k == "a" else "sd_b")
                fill = r["state"] == "backlash"
                ax.errorbar(r[temperature], r[k], yerr=sd if sd else None, fmt="o",
                            mfc=colours[r["group"]] if fill else "none", mec=colours[r["group"]], ecolor=colours[r["group"]],
                            ms=6, alpha=0.9)
                if point_labels:
                    ax.annotate(r["block"][:10], (r[temperature], r[k]), fontsize=6, alpha=0.6, xytext=(3, 3),
                                textcoords="offset points")
            ax.set_title(f"{pn}: {'offset at the reference focus (px)' if k == 'a' else 'slope (px / count)'}", fontsize=9)
            ax.set_xlabel({"T_optics_degC": "block Mastcam-Z lens temperature (HEAD_HTR) degC",
                           "T_fpa_degC": "block focal-plane temperature degC",
                           "T_navcam_degC": "block Navcam temperature degC"}.get(temperature, temperature), fontsize=8)
            ax.grid(alpha=0.3)
    handles = [plt.Line2D([], [], marker="o", ls="", color=colours[g], label=g) for g in groups]
    handles += [plt.Line2D([], [], marker="o", ls="", mfc="k", mec="k", label="backlash state"),
                plt.Line2D([], [], marker="o", ls="", mfc="none", mec="k", label="regular state")]
    for n, ax in enumerate(axs.ravel()):           # panel letters A, B ... (Christian, 22 Oct)
        ax.text(-0.02, 1.0, chr(ord("A") + n), transform=ax.transAxes, ha="right", va="bottom", fontsize=12, fontweight="bold")
    fig.legend(handles=handles, loc="upper center", ncol=min(len(handles), 6), fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    if out:
        fig.savefig(out, dpi=dpi)
    return fig

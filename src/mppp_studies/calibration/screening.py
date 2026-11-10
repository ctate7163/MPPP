"""
mppp_studies.calibration.screening - re-judging finished blocks for the joints (0.96.0: moved out of
``mppp.sfm.health``, which keeps the block health report the pipeline writes): a report re-stationed to the frames
with enough observations (``restation``), restricted to one Mastcam-Z zoom (``restrict_to_zoom``), re-judged with
other thresholds (``rejudge``), and the screening of many blocks (``screen_blocks``).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np

import re
from typing import Sequence, Tuple

from mppp.sfm.health import (DEFAULT_THRESHOLDS, _RANK, _status, beyond_ground, corner_check, log,  # noqa: F401
                             navcam_aspect_changes, zcam_uncalibrated_groups)
from mppp.sfm.project import SfmProject

PathLike = Union[str, Path]


# ----------------------------------------------------------------------------------------------------- block screening
def restation(report: Dict[str, Any], colmap_dir: PathLike, min_frame_observations: int = 30) -> List[str]:
    """
    Recompute the station shift checks of a saved health report from the label priors (0.94.4, in place).
    Before 0.94.4 a Hazcam image's shift was measured from its prior as moved by its station's Navcam shift (stage
    2, ``hazcam_station_priors``), so a station that moved made its Hazcam frames look that far from its Navcam
    frames: ``within_station_shift_spread_m`` FAIL at 1.06 m in Van Zyl Overlook Huge (Navcam shift 2.1 m).  Uses
    ``<colmap>/error_input/poses.csv`` (the refined centres) and ``project.json`` (``prior_C_label``).  Returns the
    checks changed; nothing when the block has no moved prior.
    """
    import csv
    import json as _json
    root = Path(colmap_dir)
    try:
        imgs = _json.loads((root / "project.json").read_text(encoding="utf-8")).get("images") or []
    except (OSError, ValueError):
        return []
    label = {r["name"]: r["prior_C_label"] for r in imgs if r.get("prior_C_label")}
    pc = root / "error_input" / "poses.csv"
    if not label or not pc.is_file():
        return []
    prior = {r["name"]: label.get(r["name"]) or r.get("prior_C") for r in imgs}
    by_st: Dict[str, List[np.ndarray]] = {}
    with pc.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            p0 = prior.get(row.get("name"))
            if p0 is None or float(row.get("observations") or 0) < min_frame_observations:
                continue
            d = np.array([float(row["E_m"]), float(row["N_m"]), float(row["U_m"])]) - np.asarray(p0, float)
            by_st.setdefault(row["station"], []).append(d)
    st_tab = report.get("stations") or {}
    for st, v in by_st.items():
        d = np.array(v)
        e = st_tab.setdefault(st, {"label": st})
        e.update(images=len(v), shift_median_m=float(np.median(np.linalg.norm(d, axis=1))),
                 shift_spread_m=float(np.sqrt(np.mean(np.sum((d - np.median(d, 0)) ** 2, axis=1)))))
    report["stations"] = st_tab
    changed = []
    loc = {k: v for k, v in st_tab.items() if v.get("position_prior", True) and k in by_st}
    for c in report.get("checks") or []:
        if c.get("check") == "station_shift_median_m" and loc:
            k, v = max(loc.items(), key=lambda kv: kv[1]["shift_median_m"])
            new, note = v["shift_median_m"], f"largest station median |refined - label prior| ({v.get('label', k)}; 0.94.4 recomputed)"
        elif c.get("check") == "within_station_shift_spread_m" and by_st:
            k, v = max(((k, st_tab[k]) for k in by_st), key=lambda kv: kv[1]["shift_spread_m"])
            new, note = v["shift_spread_m"], f"largest RMS spread of the shifts within one station ({v.get('label', k)}; 0.94.4 recomputed)"
        else:
            continue
        if abs(float(c.get("value") or 0) - new) > 1e-9:
            changed.append(f"{c['check']}: {c.get('value'):.3f} -> {new:.3f} m (label priors)")
            c["value"], c["note"] = new, note
    return changed


# 0.94.4 (24 Oct): checks of the labels' pointing knowledge, not of the block - Tuxedo Park (block rotation 1.62 deg:
# eight stations' headings 1.6-1.9 deg off the solution) and Witch Hazel Hill (one site's heading 3.3 deg off) are
# good blocks (median residual 0.12-0.15 px); the calibration joints take them (a block keeps its own frame there)
LABEL_POINTING_CHECKS = ("block_rotation_deg", "attitude_residual_p95_deg")


_ZOOM = re.compile(r"ZCAM\d{5}_(\d{3})")
_ZCAM_CAMERA = re.compile(r"Z[LR](\d{3})(?:_|$)")        # a Mastcam-Z bin camera (ZR048_F02070, ZL034_F00180_reg)
_ZCAM_NAMED = re.compile(r"\bZ[LR](\d{3})\b")             # Mastcam-Z camera groups named in a note (ZL063: 0.252)
_ZCAM_NAMED_FULL = re.compile(r"\bZ[LR]\d{3}\b")


def _group_zoom(grp: str) -> Optional[int]:
    """The zoom of a Mastcam-Z camera group (``ZL034``, ``ZR079_F03216``), else None."""
    m = re.match(r"Z[LR](\d{3})", str(grp))
    return int(m.group(1)) if m else None


def restrict_to_zoom(report: Dict[str, Any], colmap_dir: PathLike, zoom: int) -> List[str]:
    """0.94.4 (24 Oct, Chal Rocks Mastcam-Z: 30.2 % excluded, 180 of the 194 frames at 79 mm): for a joint of one
    zoom, the excluded fraction counted over the Navcam, Hazcam and that zoom's images only (the other zooms' frames
    are not the joint's).  Changes the report's ``excluded_fraction`` value in place; returns notes.  0.94.4 (27 Oct):
    the camera-model checks of the other zooms' Mastcam-Z cameras are taken out of the report too."""
    root = Path(colmap_dir)
    out = []
    # 0.94.4 (Christian, 27 Oct: Taylorfjellet failed on its one 48 mm pair, ZR048_F02070 principal point +90 px):
    # a camera-model check of another zoom's Mastcam-Z camera is not this joint's - it no longer fails the block
    others = [c for c in report.get("checks") or [] if c.get("section") == "camera model"
              and (m := _ZCAM_CAMERA.match(str(c.get("note") or ""))) and int(m.group(1)) != int(zoom)]
    # 27 Oct (Seitah Overlook: the eye ratio of its 63 mm frames failed the block for the Z34 joint): a residual check
    # that names only Mastcam-Z cameras of other zooms (the worse / better eye ratio "ZL063: 0.252, ZR063: 0.646")
    for c in report.get("checks") or []:
        if c.get("check") == "residual_eye_ratio" and not any(c is o for o in others):
            zs = {int(z) for z in _ZCAM_NAMED.findall(str(c.get("note") or ""))}
            if zs and int(zoom) not in zs:
                others.append(c)
    if others:
        report["checks"] = [c for c in report["checks"] if not any(c is o for o in others)]
        bad = [o for o in others if o.get("status") in ("warn", "fail")]
        def _name(o):
            nm = str(o.get("note") or "")
            return nm if _ZCAM_CAMERA.match(nm) else f"{o.get('check')} {'/'.join(sorted(set(_ZCAM_NAMED_FULL.findall(nm))))}"
        out.append(f"checks of other zooms' Mastcam-Z cameras set aside: {len(others)}"
                   + (f" ({', '.join(sorted({_name(o) + ' ' + str(o['status']) for o in bad}))})" if bad else ""))
    try:
        pj = json.loads((root / "project.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return out

    def keep(name: str) -> bool:
        m = _ZOOM.search(str(name))
        return m is None or int(m.group(1)) == int(zoom)

    ex = ((pj.get("settings") or {}).get("reconstruction") or {}).get("excluded") or []
    unc = {n for e in ex if e.get("kind", "outlier" if e.get("observations", 1) else "unconstrained") == "unconstrained"
           for n in e.get("images", [])}
    excl = {n for e in ex for n in e.get("images", [])} - unc
    names = [r["name"] for r in pj.get("images") or [] if keep(r["name"])]
    k_ex = [n for n in excl if keep(n)]
    for c in report.get("checks") or []:
        if c.get("check") == "excluded_fraction":
            old = c.get("value")
            c["value"] = len(k_ex) / max(1, len(names))
            out.append(f"excluded_fraction at {int(zoom)} mm: {old:.3f} -> {c['value']:.3f} "
                       f"({len(k_ex)} of {len(names)} Navcam, Hazcam and {int(zoom)} mm images)")
    report["zoom_restricted"] = int(zoom)
    return out


def rejudge(report: Dict[str, Any], thresholds: Optional[Dict[str, tuple]] = None,
            warn_only: Sequence[str] = ()) -> Tuple[str, List[str]]:
    """The verdict of a saved health report (``health/health.json``) under the current thresholds and rules (0.93.3):
    every check's value judged again, the corner check recomputed from ``coverage_by_radius``.  ``warn_only``: checks
    that cannot fail here (a FAIL counts as WARN; the calibration joints pass :data:`LABEL_POINTING_CHECKS`).
    Returns (verdict, the checks whose status changed)."""
    thr = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    changed, statuses = [], []
    for c in report.get("checks") or []:
        name, st0, v = c.get("check"), c.get("status"), c.get("value")
        if name == "rig_rotation_change_deg" and str(c.get("note", ""))[:2] in ("F:", "R:"):
            name = "hazcam_rig_rotation_change_deg"   # 0.93.6: a Hazcam rig (reports before 0.93.6)
        if name == "residual_eye_ratio" and any(f"{e}: " in str(c.get("note", "")) for e in ("FL", "RL")):
            name = "hazcam_residual_eye_ratio"        # 0.94.4: a Hazcam pair (reports before 0.94.4)
        if name == "corner_triangulated_ratio" and report.get("coverage_by_radius"):
            cams = report["coverage_by_radius"].get("cameras") or {}
            skip = set(zcam_uncalibrated_groups(cams))
            if report.get("zoom_restricted") is not None:  # 0.94.4: a joint of one zoom judges that zoom's corners
                skip |= {g for g in cams if _group_zoom(g) not in (None, int(report["zoom_restricted"]))}
            wc, st_c = corner_check(report["coverage_by_radius"], thr, skip_groups=sorted(skip))
            v = wc[1] if wc else None
            st = st_c if wc else _status(name, None, thr)     # no camera to judge: "info", as before 0.94.5
        else:
            st = _status(name, v, thr) if name in thr else st0
        if st == "fail" and name in warn_only:
            st = "warn"
        if st != st0:
            changed.append(f"{name}: {st0} -> {st}" + (f" ({c.get('note', '')[:60]})" if c.get("note") else ""))
        statuses.append(st)
    return max(statuses, key=lambda s: _RANK.get(s, 0), default="pass"), changed


def screen_block(work: PathLike, model: str = "cahv_ba", thresholds: Optional[Dict[str, tuple]] = None,
                 sample: Optional[int] = 100000) -> Dict[str, Any]:
    """
    the two checks that found the bad blocks of v0p63-v0p64 (beyond-ground share of the tie points, change of
    the Navcam pixel aspect fy / fx from its start), from a saved alignment ``<work>/colmap/sparse/<model>``, for blocks
    aligned before these checks were part of the health report.  Also records the MPPP version and whether COLMAP
    guided matching was on.  Returns {"verdict": "pass" | "warn" | "fail", "reasons", "beyond_ground_fraction",
    "navcam_aspect_change_pct", "aspect_camera", "mppp", "guided_matching", "zcam"}.
    """
    import json as _json
    import pycolmap
    from mppp.waypoints.sites import parse_work_folder
    thr = dict(DEFAULT_THRESHOLDS, **(thresholds or {}))
    work = Path(work)
    root = work / "colmap" if (work / "colmap").is_dir() else work
    out: Dict[str, Any] = {"work": str(work), "zcam": parse_work_folder(work if root != work else work.parent)[1]}
    sp = root / "sparse" / model
    if not (sp / "points3D.bin").is_file() and not (sp / "points3D.txt").is_file():
        out.update(verdict="fail", reasons=[f"no alignment in {sp}"])
        return out
    # the result is kept in <colmap>/health/screen_<model>.json and reused while the alignment is unchanged
    cache = root / "health" / f"screen_{model}.json"
    key = {"files": {f.name: [f.stat().st_size, f.stat().st_mtime_ns] for f in sorted(sp.glob("*.bin"))},
           "thresholds": {k: list(v) for k, v in thr.items() if k in ("beyond_ground_fraction", "navcam_aspect_change_pct")},
           "sample": sample,
           "meta": {f: [(root / f).stat().st_size, (root / f).stat().st_mtime_ns] for f in ("run_done.json", "database_matches.json")
                    if (root / f).is_file()}}
    try:
        c = _json.loads(cache.read_text(encoding="utf-8"))
        if c.get("key") == key:
            return dict(c["result"], cached=True)
    except (OSError, ValueError):
        pass
    rec = pycolmap.Reconstruction(str(sp))
    project = SfmProject.load(root)
    bg = beyond_ground(rec, sample=sample)
    asp = navcam_aspect_changes(project, rec)
    wk = max(asp, key=lambda k: abs(asp[k])) if asp else None
    out.update(beyond_ground_fraction=bg["fraction"], navcam_aspect_change_pct=(asp[wk] if wk else None),
               aspect_camera=wk)
    for f, k in (("run_done.json", "mppp"), ("database_matches.json", "matching")):
        try:
            d = _json.loads((root / f).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if k == "mppp":
            out["mppp"] = d.get("mppp")
        else:
            out["guided_matching"] = bool((d.get("matching") or {}).get("guided_matching"))
    reasons, worst = [], "pass"
    for name, v in (("beyond_ground_fraction", bg["fraction"]),
                    ("navcam_aspect_change_pct", None if wk is None else abs(asp[wk]))):
        st = _status(name, v, thr)
        if st in ("warn", "fail"):
            reasons.append(f"{name} {v:.4g} ({st}; warn {thr[name][0]:g}, fail {thr[name][1]:g})"
                           + (f" - {wk}" if name.startswith("navcam") else ""))
            worst = max(worst, st, key=lambda s: _RANK[s])
    if out.get("guided_matching"):
        reasons.append("matched with COLMAP guided matching (v0p63-v0p64.0): realign")
        worst = max(worst, "warn", key=lambda s: _RANK[s])
    out.update(verdict=worst, reasons=reasons)
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(_json.dumps({"key": key, "result": out}, indent=1, default=str), encoding="utf-8")
    except OSError:
        pass
    return out


def screen_blocks(blocks: Dict[str, PathLike], verbose: bool = True, **kw) -> Dict[str, Dict[str, Any]]:
    """:func:`screen_block` of every block {name: WORK folder}; prints one line per block."""
    out = {}
    for name, work in blocks.items():
        try:
            r = screen_block(work, **kw)
        except Exception as e:                                   # noqa: BLE001 - a damaged block is reported, not fatal
            r = {"work": str(work), "verdict": "fail", "reasons": [f"{type(e).__name__}: {e}"]}
        out[name] = r
        if verbose:
            bg, a = r.get("beyond_ground_fraction"), r.get("navcam_aspect_change_pct")
            log.info(f"  {r['verdict'].upper():4s} {name:34s} MPPP {str(r.get('mppp')):7s} beyond-ground "
                  f"{'-' if bg is None else f'{100 * bg:5.1f} %'}  aspect {'-' if a is None else f'{a:+.3f} %'} "
                  f"{r.get('aspect_camera') or ''}" + (f"  | {'; '.join(r['reasons'])}" if r.get("reasons") else ""))
    return out

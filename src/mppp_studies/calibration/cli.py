"""
mppp_studies.calibration.cli - the ``calib`` and ``zcam-states`` commands (0.96.0: moved out of ``mppp.cli``, which
forwards them here).

    python -m mppp calib zcam --root D:\\rover_blocks ...          (through the core CLI)
    python -m mppp_studies.calibration calib zcam ...               (the same, the study alone)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional


log = logging.getLogger("mppp.calib")


def add_parsers(sub) -> None:
    """The ``calib`` and ``zcam-states`` sub-parsers on an argparse subparsers object."""
    p = sub.add_parser("calib", help="the camera calibrations (0.91: the Navcam tied joint, the Mastcam-Z joint)")
    p.add_argument("which", choices=["navcam", "zcam", "nz", "hazcam", "lawcheck"],
                   help="navcam: the one-adjustment joint over the finished Navcam blocks; zcam: the Mastcam-Z joint "
                        "over the finished Navcam + Mastcam-Z blocks, the Navcam held at the models in use; nz: the "
                        "combined joint (both laws free) compared with the Navcam-only law - a check, never promoted; "
                        "hazcam: the Hazcam law (Navcam form) from the finished blocks with Hazcams (0.93); "
                        "lawcheck: how well each Navcam law (--law) predicts the blocks' cameras (mppp_studies.calibration.lawcheck)")
    p.add_argument("--variant", default="", help="the blocks' project colmap_<variant> (default colmap; the Hazcam-only "
                   "alignments: hazcam)")
    p.add_argument("--root", default=None, metavar="DIR", help="the scapes root (default: MPPP_SCAPES)")
    p.add_argument("--group", default=None, help="site group(s) of the blocks, comma-separated (default ecam_consensus / "
                   "zcam<zoom>_joint, else zcam<zoom>_consensus); zcam / nz take the sites whose \"zcam\" list has the zoom")
    p.add_argument("--zoom", type=int, default=34, help="zcam: the Mastcam-Z zoom (default 34)")
    p.add_argument("--zcam-blocks", action="store_true",
                   help="hazcam: the sites' Navcam + Mastcam-Z blocks (..._zcam folders, every zoom) - 0.93.6, "
                        "zcam_joints.bat fits the Hazcams of the run_sites_zcam blocks")
    p.add_argument("--navcam-group", default=None,
                   help="nz: also the finished Navcam-only blocks of this group (e.g. ecam_consensus), "
                        "3000 points each, so the combined Navcam law is held by every Navcam block")
    p.add_argument("--blocks", nargs="*", default=None, metavar="NAME=WORK", help="blocks given explicitly instead")
    p.add_argument("--sites-file", default=None, metavar="FILE")
    p.add_argument("--out", default=None, metavar="DIR", help="the study folder (default <root>/camera_analysis/navcal_tied_<date>)")
    p.add_argument("--points", type=int, default=None, help="Navcam points per block (default 8000; navcam: "
                   "--points-per-station instead)")
    p.add_argument("--points-per-station", type=float, default=None, metavar="N",
                   help="navcam (2 Nov 2026): a block's Navcam points = N x its stations, at least 2000 (default 750; "
                        "--points N: N per block)")
    p.add_argument("--zpoints", type=int, default=None, help="zcam: Mastcam-Z-only points per block (default 8000; "
                   "every Navcam-Mastcam-Z point with a third look is kept besides)")
    p.add_argument("--hold", nargs="*", default=None, help="global terms held (default Q0_yaw)")
    p.add_argument("--zero", nargs="*", default=None, metavar="TERM",
                   help="navcam (3 Nov 2026): global terms set to 0 and held, besides --hold: D_roll (one roll for the "
                        "mission), S_NL_cx S_NL_cy S_NR_cy (those principal points constant with temperature) ...")
    p.add_argument("--knot", type=float, default=None, help="the pitch hinge sol (default the rig file's)")
    p.add_argument("--tie-f-slope", action="store_true",
                   help="navcam: NR's focal thermal slope tied to NL's ppm/degC (0.95.0; the comparison run before it becomes the form)")
    p.add_argument("--aniso-f", action="store_true",
                   help="separate fx and fy thermal slopes (default: one ppm/degC per eye, fy tied to fx)")
    p.add_argument("--no-diagnostics", action="store_true", help="skip the per-block diagnostic pass (no gate then)")
    p.add_argument("--iterations", type=int, default=None, metavar="N",
                   help="zcam / nz / navcam (0.95.3, Christian 31 Oct: \"allow more joint iterations for convergence\"): Ceres "
                        "iterations per pass of the adjustment and of its tests (default 100; the Z48 joint of 31 Oct "
                        "stopped at 101 in both passes)")
    p.add_argument("--passes", type=int, default=None, metavar="N",
                   help="zcam / nz: passes of the adjustment (default 2; the covariance on the last)")
    p.add_argument("--wait-h", type=float, default=36.0, metavar="H",
                   help="wait up to this many hours for sites of the groups whose alignment is running (0.95.0; 0: take "
                        "the finished ones now)")
    p.add_argument("--include-failed", action="store_true", help="also blocks whose alignment health is FAIL "
                   "(left out by default)")
    p.add_argument("--no-k-tests", action="store_true", help="zcam: skip the k1/k2 likelihood-ratio tests (2 more "
                   "adjustments)")
    p.add_argument("--eye-slope-tests", action="store_true",
                   help="zcam: the eye-slope tests (0.95.0: the rig translation free, one focus slope for both eyes, "
                        "both - 3 more adjustments)")
    p.add_argument("--zcam-rig-fit", choices=("full", "rotation"), default=None,
                   help="zcam: the Mastcam-Z rig in the joint - full: rotation and translation refined (0.95.1 default, "
                        "1 cm prior), rotation: the translation held at the start rig (the 0.94 joint)")
    p.add_argument("--zcam-early-sol", type=int, default=None, metavar="SOL",
                   help="zcam: ZR's early-mission principal-point offset for its bins before this sol (default 150; 0: none)")
    p.add_argument("--zcam-aspect", type=float, default=None, metavar="RATIO",
                   help="zcam (6 Nov 2026): hold each eye's fy / fx at RATIO (1.0: square pixels) - the test of the "
                        "Navcam-Mastcam-Z vertical-scale tension; default: the aspect is free")
    p.add_argument("--zcam-ratio-by", choices=("block", "sol", "global"), default=None,
                   help="zcam: the low-state groups with their own high / low ratio (default block; global: one per eye)")
    p.add_argument("--zcam-focus-below", action="store_true",
                   help="zcam (0.96): fit the focus-approach offset again (off since 9 Nov: held at 0)")
    p.add_argument("--zcam-pool", choices=("sequence", "block", "none"), default=None,
                   help="zcam (0.96): the diagnostic pass's pooled model, which the per-block aspect gate reads - one pool "
                        "per eye and sequence (default), per eye, block and backlash state, or none (the per-bin pass "
                        "alone, the gate on the per-bin lines as before 0.96)")
    p.add_argument("--law", action="append", default=None, metavar="LABEL=DIR",
                   help="lawcheck: a Navcam law to check, a cameras folder (repeatable; default in_use=the models in use)")
    p.add_argument("--promote-if-ok", action="store_true", help="promote the candidate when every check passes")
    p.add_argument("--no-promote", action="store_true", help="never promote (overrides --promote-if-ok)")
    p.add_argument("--note", default="", help="promotion note")
    p.add_argument("--min-blocks", type=int, default=None,
                   help="zcam / hazcam: the blocks a candidate needs (0.94.1, the tiny joints: 1 or 2)")
    p.add_argument("--hazcam-method", choices=["joint", "blocks"], default="joint",
                   help="hazcam: joint = the law fitted on the image observations of every block, each frame at its "
                        "own temperature (0.94.4, default); blocks = a line through one refined camera per block")
    p.add_argument("--hazcam-slope", choices=["front", "FL", "all", "per_eye"], default="front",
                   help="hazcam joint: front = FL and FR their own ppm/degC, RL and RR the mean of the two (default, "
                        "0.95.1, Christian 29 Oct); FL = every eye takes FL's (24 Oct); all = one slope shared by the "
                        "four eyes; per_eye = a slope per eye")
    p.add_argument("--promote-waive", nargs="*", default=None, metavar="CHECK",
                   help="zcam (0.95.5, Christian 8 Nov: \"update the z63, z79, z110 models that are good enough\"): promote "
                        "with --promote-if-ok when the only checks that failed are these (names as check.json gives "
                        "them, e.g. \"per-block aspect vs the Navcam\"); the waiver is written to promotion_waiver.json "
                        "and into the promotion note")
    p.add_argument("--promote-new-only", action="store_true",
                   help="zcam / hazcam: promote only models of which none is in use (0.94.1: a tiny joint never "
                        "replaces a model of more blocks)")
    p = sub.add_parser("zcam-states", help="0.94.4: the Mastcam-Z backlash state of every image from the blocks' focal "
                       "lengths (the states run: run_zcam_states.bat) - proposed additions to and removals from the "
                       "found list")
    p.add_argument("--root", default=os.environ.get("MPPP_SCAPES"), help="the scapes root (default MPPP_SCAPES)")
    p.add_argument("--variant", default="", help="the blocks' project colmap_<variant> (the states run: states)")
    p.add_argument("--sites", nargs="*", default=None, help="only these sites' _zcam blocks (default: every one with "
                   "a zcam_focus_states.json)")
    p.add_argument("--found", default=None, help="the found list (default <root>/zcam_low_backlash_found.csv)")
    p.add_argument("--out", default=None, help="the folder for report.md, images.csv, sequences.csv, levels.json and "
                   "found_list_proposal.txt (default <root>/camera_analysis/zcam_states_<date>)")
    p.add_argument("--z", type=float, default=3.0, help="z of a state call (default 3)")
    p.add_argument("--apply", action="store_true", help="write the proposal into the found list (a dated backup first)")


def _navcam_blocks(a: argparse.Namespace) -> Dict[str, Path]:
    """{name: WORK} of the finished Navcam-only blocks of the group under the scapes root (run_done.json there)."""
    from mppp.pipeline import run_done
    from mppp.waypoints.sites import load_site_table, work_folder
    if a.blocks:
        out = {}
        for b in a.blocks:
            name, _, work = b.partition("=")
            if not work:
                raise SystemExit(f"--blocks takes NAME=WORK, got {b!r}")
            out[name] = Path(work)
        return out
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or "")
    if not str(root):
        raise SystemExit("--root (or MPPP_SCAPES) is required")
    table = load_site_table(a.sites_file)
    zcam = getattr(a, "which", "navcam") in ("zcam", "nz")
    zfolders = zcam or bool(getattr(a, "zcam_blocks", False))     # 0.93.6: calib hazcam --zcam-blocks
    a.group = a.group or (f"zcam{int(getattr(a, 'zoom', 34))}_joint" if zcam and f"zcam{int(getattr(a, 'zoom', 34))}_joint"
                          in table.get("groups", {}) else f"zcam{int(getattr(a, 'zoom', 34))}_consensus" if zcam
                          else "hazcam_consensus" if getattr(a, "which", "") == "hazcam" else "ecam_consensus")
    names: List[str] = []
    for grp in str(a.group).split(","):                 # 0.93: several groups, comma-separated
        g_names = table.get("groups", {}).get(grp.strip())
        if not g_names:
            raise SystemExit(f"no group {grp.strip()!r} in the site definitions")
        names += list(g_names)
    out, skipped = {}, []
    zoom = int(getattr(a, "zoom", 34) or 34)
    wanted = []
    for n in dict.fromkeys(names):
        site = table["sites"].get(n)
        if not site:
            skipped.append(f"{n} (not a site)")
            continue
        if zcam and site.get("zcam") and zoom not in [int(z) for z in site["zcam"]]:
            skipped.append(f"{n} (no {zoom} mm in its \"zcam\" list)")
            continue
        if zfolders and not zcam and not site.get("zcam"):
            skipped.append(f"{n} (no Mastcam-Z block)")
            continue
        wanted.append((n, work_folder(root, n, zfolders, site["sols"])))
    # 0.95.0 (Christian, 27 Oct): a site of the group whose alignment is running is waited for, not taken half-done
    _wait_for_running(wanted, float(getattr(a, "wait_h", 0) or 0))
    for n, work in wanted:
        done = run_done(work, getattr(a, "variant", "") or "")
        if done is None:
            skipped.append(f"{n} (not finished)")
            continue
        if str(done.get("verdict")) == "fail" and not getattr(a, "include_failed", False):
            # 0.93.3: judged again under the current thresholds and rules (a block run before a rule changed)
            v2, changed = _rejudged(work, getattr(a, "variant", "") or "", zoom=zoom if zcam else None)
            if v2 == "fail" or v2 is None:
                skipped.append(f"{n} (alignment health FAIL; --include-failed takes it)")    # 22 Oct
                continue
            log.info("calib: %s was FAIL under the rules of its run, %s under the current ones (%s)", n, v2.upper(),
                     "; ".join(changed) or "-")
        out[n] = work
    if skipped:
        log.info("calib %s: left out: %s", "zcam" if zcam else "navcam", ", ".join(skipped))
    return out


def _wait_for_running(wanted, wait_h: float, poll_s: float = 600.0) -> List[str]:
    """Wait (up to ``wait_h`` hours, polling every ``poll_s``) while any of the ``wanted`` [(name, WORK)] blocks has
    an alive run status (``mppp_status.json``: a site being aligned); returns the names still running at the end."""
    import time
    from mppp.runner import STATUS_FILE, read_status
    t0 = time.time()
    while True:
        running = [n for n, w in wanted if (read_status(Path(w) / STATUS_FILE) or {}).get("alive")]
        if not running or wait_h <= 0 or time.time() - t0 > wait_h * 3600:
            if running:
                log.warning("calib: %s still running (%s) - taken as they are if finished, else left out",
                            ", ".join(running), "not waited for" if wait_h <= 0 else f"after {wait_h:g} h")
            return running
        log.info("calib: waiting for %d running site(s): %s (checking again in %.0f min)", len(running),
                 ", ".join(running), poll_s / 60)
        time.sleep(poll_s)


def _rejudged(work, variant: str = "", zoom: Optional[int] = None):
    """(verdict, changed checks) of a block's saved health report under the current rules; (None, []) without one.
    ``zoom`` (a Mastcam-Z joint, 0.94.4): the excluded frames and the corners judged on that zoom (and the Navcam,
    Hazcams) only - Chal Rocks' 79 mm frames failed a block the Z34 joint needs."""
    from mppp_studies.calibration.screening import LABEL_POINTING_CHECKS
    from mppp_studies.calibration.screening import rejudge, restation, restrict_to_zoom
    from mppp.sfm.workdir import project_dir
    try:
        h = json.loads((project_dir(work, variant) / "health" / "health.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, []
    fixed = restation(h, project_dir(work, variant))     # 0.94.4: shifts from the label priors (moved Hazcam priors)
    if zoom is not None:
        fixed += restrict_to_zoom(h, project_dir(work, variant), int(zoom))
    v, changed = rejudge(h, warn_only=LABEL_POINTING_CHECKS)     # the labels' heading errors are not the block's
    return v, fixed + changed


def cmd_calib(a: argparse.Namespace) -> int:
    """``calib navcam``: the one-adjustment Navcam joint (:mod:`mppp_studies.calibration.tied`) over the finished blocks, the
    candidate folder, the promotion check and, with ``--promote-if-ok``, the promotion."""
    import datetime
    from . import tied as TJ
    from .fit import check_lines, check_navcam, promote_navcam
    if a.which == "zcam":
        return _calib_zcam(a)
    if a.which == "nz":
        return _calib_nz(a)
    if a.which == "hazcam":
        return _calib_hazcam(a)
    if a.which == "lawcheck":
        return _calib_lawcheck(a)
    blocks = _navcam_blocks(a)
    if len(blocks) < 2:
        print(f"ERROR: {len(blocks)} finished block(s); the joint needs several", file=sys.stderr)
        return 1
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or next(iter(blocks.values())).parent)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"navcal_tied_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("calib navcam: %d blocks -> %s", len(blocks), out)
    pts = a.points or {"per_station": float(getattr(a, "points_per_station", None) or TJ.POINTS_PER_STATION),
                       "min": TJ.POINTS_MIN_PER_BLOCK}
    res = TJ.fit_tied(blocks, points=pts, hold=tuple(a.hold) if a.hold is not None else TJ.HOLD_DEFAULT,
                      knot_sol=a.knot, diagnostics=not a.no_diagnostics, iso_f=not a.aniso_f,
                      tie_f=bool(getattr(a, "tie_f_slope", False)) or None, zero=tuple(getattr(a, "zero", None) or ()),
                      **({"iterations": int(a.iterations)} if getattr(a, "iterations", None) else {}))   # 2 Nov 2026
    cand = TJ.write_tied(res, out / "navcam_joint", note=a.note)
    rep = check_navcam(cand)
    (out / "promotion_check.json").write_text(json.dumps(rep, indent=1, default=str), encoding="utf-8")
    print("\n".join(check_lines(rep)))
    if a.promote_if_ok and not a.no_promote and rep["ok"]:
        pr = promote_navcam(cand, a.note or f"tied joint of {len(blocks)} blocks", check=False)
        print(f"promoted into {pr['folder']} ({pr['stamp']})")
    elif a.promote_if_ok:
        print("not promoted: a check failed (the models in use stay)")
    return 0


def _calib_zcam(a: argparse.Namespace) -> int:
    """``calib zcam``: the Mastcam-Z joint (:mod:`mppp_studies.calibration.ztied`) over the finished Navcam + Mastcam-Z blocks, the
    Navcam held at the models in use; the candidate focus model, its check and, with ``--promote-if-ok``, the
    promotion."""
    import datetime
    from mppp.calib import models as M
    from . import ztied as Z
    blocks = _navcam_blocks(a)
    if not blocks:
        print(f"no finished Mastcam-Z {a.zoom} mm block: no joint", file=sys.stderr)
        return 1
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or next(iter(blocks.values())).parent)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"zcam{a.zoom}_joint_{datetime.date.today():%Y%m%d}"
    out.mkdir(parents=True, exist_ok=True)
    log.info("calib zcam: %d blocks -> %s", len(blocks), out)
    res = Z.fit_zjoint(blocks, zoom=a.zoom, nav_points=a.points or Z.NAV_POINTS_PER_BLOCK,
                       iterations=int(getattr(a, "iterations", None) or 100),
                       passes=int(getattr(a, "passes", None) or 2),
                       z_points=a.zpoints or Z.Z_POINTS_PER_BLOCK, diagnostics=not a.no_diagnostics,
                       k_tests=not a.no_k_tests, ratio_by=getattr(a, "zcam_ratio_by", None),
                       slope_tests=bool(getattr(a, "eye_slope_tests", False)),
                       aspect_hold=getattr(a, "zcam_aspect", None),
                       rig_fit=getattr(a, "zcam_rig_fit", None) or Z.ZJOINT_RIG_FIT,
                       early_sol=(Z.ZCAM_EARLY_SOL if getattr(a, "zcam_early_sol", None) is None
                                  else (a.zcam_early_sol or None)),
                       pool=(Z.DIAG_POOL if getattr(a, "zcam_pool", None) is None
                             else (None if a.zcam_pool == "none" else a.zcam_pool)),
                       focus_below=bool(getattr(a, "zcam_focus_below", False)) or Z.FOCUS_BELOW_FIT)
    from mppp.calib import models_fingerprint
    from mppp.pipeline import run_done
    res["models_fingerprint"] = models_fingerprint()
    res["block_runs"] = {n: run_done(w, "") for n, w in blocks.items()}
    # 0.93: the joint holds the Navcam: a block must have been aligned with the Navcam models in use (a block aligned
    # with an earlier Mastcam-Z model is fine - the joint refits the Mastcam-Z law from its bins)
    from mppp.calib import navcam_fingerprint
    res["navcam_fingerprint"] = navcam_fingerprint(None)
    res["block_navcam_fingerprints"] = {}
    for n, w in blocks.items():
        try:
            pjf = Path(w) / "colmap" / "project.json"
            if not pjf.is_file():                         # 0.94.1: a block given as its project folder (colmap_<variant>)
                pjf = Path(w) / "project.json"
            pj = json.loads(pjf.read_text(encoding="utf-8"))
            res["block_navcam_fingerprints"][n] = ((pj.get("settings") or {}).get("navcam_cameras") or {}).get("fingerprint")
        except (OSError, ValueError):
            res["block_navcam_fingerprints"][n] = None
    lim = {"min_blocks": a.min_blocks, "min_blocks_start": a.min_blocks} if getattr(a, "min_blocks", None) else None
    w = Z.write_zjoint(res, out, limits=lim)
    for c in w["check"]["checks"]:
        print(f"  {Z.check_label(c):4s} {c['check']}: {c['note']}")
    warned = sum(1 for c in w["check"]["checks"] if c["ok"] and c.get("warning"))
    print(f"=> {'every check passes' if w['check']['ok'] else 'a check failed'}"
          + (f" ({warned} warning{'s' if warned != 1 else ''})" if warned else "") + f"; {out / 'report.md'}")
    promote = a.promote_if_ok and not a.no_promote
    if promote and getattr(a, "promote_new_only", False) and (M.zcam_files() or {}).get(int(a.zoom)):
        print(f"not promoted: a {a.zoom} mm model is in use (--promote-new-only)")
        promote = False
    failed = [c["check"] for c in w["check"]["checks"] if not c["ok"]]
    waive = [str(x) for x in (getattr(a, "promote_waive", None) or [])]
    waived = bool(failed) and all(f in waive for f in failed)
    if promote and (w["check"]["ok"] or waived):
        note = a.note or f"Mastcam-Z {a.zoom} mm joint of {len(blocks)} blocks"
        if waived:                                     # 0.95.5: a hand decision, recorded with the model
            rec = {"zoom": int(a.zoom), "waived": failed, "notes": {c["check"]: c["note"] for c in w["check"]["checks"]
                                                                     if not c["ok"]},
                   "reason": "--promote-waive: the only failed checks were waived (good enough for the freeze)"}
            (out / "promotion_waiver.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
            note += "; waived: " + "; ".join(failed)
            print(f"promoting with the waived check(s): {', '.join(failed)}")
        pr = M.promote([Path(w["model"])], note=note)
        print(f"promoted into {pr['folder']} ({pr['stamp']})")
    elif promote:
        print("not promoted: a check failed (the model in use stays)"
              + (f"; not waived: {', '.join(f for f in failed if f not in waive)}" if waive else ""))
    return 0


def _calib_lawcheck(a: argparse.Namespace) -> int:
    """``calib lawcheck`` (0.94.4; the 25 Oct study as a command): each Navcam law (``--law LABEL=DIR``, default the
    models in use) against the finished Navcam blocks - the law held, the units free, the pointing pass - into
    ``law_validation.json`` and ``report.md``; ``mppp_studies.calibration.plot_lawcheck`` draws the figures."""
    import datetime
    from . import lawcheck as L
    blocks = _navcam_blocks(a)
    if len(blocks) < 2:
        print(f"ERROR: {len(blocks)} finished block(s); the law check needs several", file=sys.stderr)
        return 1
    laws: Dict[str, Optional[str]] = {}
    for item in a.law or ["in_use="]:
        label, _, folder = str(item).partition("=")
        if not label.strip():
            print(f"ERROR: --law {item!r}: LABEL=DIR", file=sys.stderr)
            return 2
        laws[label.strip()] = folder.strip() or None
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or next(iter(blocks.values())).parent)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"lawcheck_{datetime.date.today():%Y%m%d}"
    log.info("calib lawcheck: %d blocks, laws %s -> %s", len(blocks), ", ".join(laws), out)
    res = L.validate_laws(blocks, laws, points=a.points or 4000)
    L.write_validation(res, out)
    print(f"law check -> {out / 'report.md'}; figures: python -m mppp_studies.calibration.plot_lawcheck \"{out}\"")
    return 0


def _calib_hazcam(a: argparse.Namespace) -> int:
    """``calib hazcam``: the Hazcam law (:mod:`mppp_studies.calibration.hazfit`) from the finished blocks with Hazcams; the
    candidate files of the eyes and pairs that pass their checks, promoted with ``--promote-if-ok``."""
    import datetime
    from . import hazfit as H
    from mppp.calib import models as M
    from mppp.sfm.workdir import project_dir
    blocks = _navcam_blocks(a)
    if not blocks:
        print("no finished block: no Hazcam fit", file=sys.stderr)
        return 1
    blocks = {n: project_dir(w, a.variant or None) for n, w in blocks.items()}
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or ".")
    out = Path(a.out) if a.out else root / "camera_analysis" / f"hazcam_fit_{datetime.date.today():%Y%m%d}"
    log.info("calib hazcam: %d blocks -> %s", len(blocks), out)
    lim = {"min_blocks": a.min_blocks} if getattr(a, "min_blocks", None) else None
    method = getattr(a, "hazcam_method", None) or "joint"
    res = None
    if method == "joint":                 # 0.94.4 (Christian 24 Oct): the law fitted on the observations of every block
        from . import hazjoint as HJ
        try:
            res = HJ.fit_hazcam_joint(blocks, slope_mode=getattr(a, "hazcam_slope", None) or HJ.SLOPE_MODE, limits=lim)
        except Exception as e:                       # noqa: BLE001 - the block fit stands in
            log.warning("hazcam joint failed (%s: %s); the block fit instead", type(e).__name__, e)
    if res is None:
        res = H.fit_hazcam(blocks, limits=lim)
    w = H.write_hazcam(res, out)
    for line in H.describe(res):
        print("  " + line)
    for c in w["check"]["checks"]:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['note']}")
    promote = a.promote_if_ok and not a.no_promote
    if promote and getattr(a, "promote_new_only", False):        # 0.94.1: only models of which none is in use
        from mppp.paths import cmods_dir
        kept = [f for f in w["files"] if not (cmods_dir() / Path(f).name).is_file()]
        if len(kept) < len(w["files"]):
            print("in use, not replaced (--promote-new-only): " + ", ".join(Path(f).name for f in w["files"] if f not in kept))
        w["files"] = kept
    if promote and w["files"]:
        try:
            pr = M.promote([Path(f) for f in w["files"]], note=a.note or f"Hazcam fit of {len(res['blocks'])} blocks "
                           f"({', '.join(res['blocks'])})")
        except OSError as e:                          # 0.93.8: say so instead of a traceback; the candidate stays
            print(f"NOT promoted: {e} - promote later with: mppp cmods promote {w['candidate']}", file=sys.stderr)
            return 1
        print(f"promoted {', '.join(Path(f).name for f in w['files'])} into {pr['folder']} ({pr['stamp']})")
    elif promote:
        print("not promoted: no eye or pair passed its checks")
    print(f"=> {out / 'report.md'}")
    return 0


def _calib_nz(a: argparse.Namespace) -> int:
    """``calib nz``: the combined Navcam + Mastcam-Z joint (:mod:`mppp_studies.calibration.nzjoint`) over the finished Navcam +
    Mastcam-Z blocks - both laws free - and its comparison with the Navcam-only law.  Writes a report; promotes
    nothing.  Exit 0 whether or not they agree (the report says which)."""
    import datetime
    from . import nzjoint as NZ
    from . import ztied as Z
    blocks = _navcam_blocks(a)
    if len(blocks) < 2:
        print(f"ERROR: {len(blocks)} finished Mastcam-Z block(s); the joint needs several", file=sys.stderr)
        return 1
    root = Path(a.root or os.environ.get("MPPP_SCAPES") or next(iter(blocks.values())).parent)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"nz{a.zoom}_joint_{datetime.date.today():%Y%m%d}"
    log.info("calib nz: %d blocks -> %s", len(blocks), out)
    extra = {}
    if getattr(a, "navcam_group", None):
        import copy as _copy
        b = _copy.copy(a)
        b.which, b.group, b.blocks = "navcam", a.navcam_group, None
        extra = _navcam_blocks(b)
    res = NZ.fit_nzjoint(blocks, zoom=a.zoom, nav_points=a.points or Z.NAV_POINTS_PER_BLOCK,
                         iterations=int(getattr(a, "iterations", None) or 100),
                         passes=int(getattr(a, "passes", None) or 2),
                         z_points=a.zpoints or Z.Z_POINTS_PER_BLOCK, navcam_blocks=extra)
    w = NZ.write_nzjoint(res, out)
    for line in NZ.describe(res):
        print("  " + line)
    print(f"=> {'agrees with' if w['agree'] else 'DIFFERS from'} the Navcam-only law; {out / 'report.md'} (nothing promoted)")
    return 0


def cmd_zcam_states(a: argparse.Namespace) -> int:
    """``zcam-states``: :mod:`mppp_studies.calibration.zcam_states` over the ``_zcam`` blocks under ``--root``."""
    import datetime
    from . import zcam_states as ZS
    from mppp.sfm.backlash import default_focus_below_path, default_found_path, load_focus_below, load_low_backlash
    if not a.root:
        print("zcam-states: --root (or MPPP_SCAPES) is needed", file=sys.stderr)
        return 2
    root = Path(a.root)
    found = Path(a.found) if a.found else default_found_path(root)
    blocks = ZS.find_blocks(root, a.variant, a.sites)
    if not blocks:
        print(f"zcam-states: no _zcam block with zcam_focus_states.json under {root} (variant "
              f"{a.variant or '(default)'})", file=sys.stderr)
        return 1
    lists = load_low_backlash(None, found=found)
    focus_list = load_focus_below(found=default_focus_below_path(root))      # 0.94.4: the focus approach
    res = ZS.survey(blocks, a.variant, lists, z_min=a.z, focus_list=focus_list)
    out = Path(a.out) if a.out else root / "camera_analysis" / f"zcam_states_{datetime.date.today():%Y%m%d}"
    files = ZS.write_survey(res, out)
    p = res["proposal"]
    print(f"zcam-states: {len(blocks)} blocks, {len(res['images'])} images; {len(p['add'])} to add, "
          f"{len(p['remove'])} to take off, {len(p['review'])} to review -> {files['report']}")
    if a.apply:
        if not found:
            print("zcam-states --apply: no found list (give --found)", file=sys.stderr)
            return 2
        r = ZS.apply_proposal(p, found)
        print(f"found list {found}: {r['added']} added, {r['removed']} taken off (backup {r['backup']})")
    return 0


COMMANDS = {"calib": cmd_calib, "zcam-states": cmd_zcam_states}


def main(argv: Optional[List[str]] = None) -> int:
    from mppp.cli import setup_logging
    ap = argparse.ArgumentParser(prog="python -m mppp_studies.calibration", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log", default=None, help="also log to this file")
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="cmd")
    add_parsers(sub)
    a = ap.parse_args(argv)
    if not a.cmd:
        ap.print_help()
        return 2
    setup_logging(a)
    return int(COMMANDS[a.cmd](a))


if __name__ == "__main__":
    sys.exit(main())

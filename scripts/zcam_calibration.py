"""
Mastcam-Z 34 mm calibration from the solved Navcam + Mastcam-Z blocks (MPPP v0p74; mppp.sfm.zcam_calib).

    python scripts\\zcam_calibration.py start-update --root D:\\scapes\\colmap --promote
    python scripts\\zcam_calibration.py calibrate    --root D:\\scapes\\colmap --n 3 --promote
    python scripts\\zcam_calibration.py grade        --root D:\\scapes\\colmap
    python scripts\\zcam_calibration.py pointclouds  --root D:\\scapes\\colmap --groups zcam34_consensus navcam_consensus

* ``start-update``: from the blocks as they are (also those solved before v0p74), (1) the focus groups they
  classified low-backlash go into ``<root>\\zcam_low_backlash_found.csv`` (every later run reads it), (2) a start focus
  model - fx, fy lines per eye from the refined bins, each block's Navcam scale divided out, the regular-state ratio -
  checked against the model in use (``zcam_calib.MODEL_LIMITS``) and, with ``--promote``, promoted.
* ``calibrate``: the blocks solved with v0p74 (their fx, fy, cx, cy, k1, k2 focus lines per state and refined rigs)
  are graded; the ``--n`` best passing ones give the calibration (lines averaged with their scatter, a lens-temperature
  term where significant, the stereo rig and baseline), checked and, with ``--promote``, promoted; a report (Markdown,
  JSON, figure) goes to ``<root>\\camera_analysis\\zcam34_calibration_<date>``.
* ``clean-found`` (v0p74.8): remove from the found list the observations whose sequence is backlash as a whole
  (one focus state per sequence; the shipped list holds the known exceptions); a dated backup is kept.
* ``grade``: the grades only.  ``pointclouds``: a PLY per finished block (``<work>\\colmap\\pointcloud``) and an index.

Exit code 0 when the step did what it should (promotion included when asked), 1 when it ran but a check failed
(nothing promoted), 2 on errors.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def blocks_of(root: Path, groups, zcam: bool):
    """{site: WORK folder} of the groups' sites that exist on disk (a site listed twice once)."""
    from mppp.sfm.sites import load_site_table, work_folder, zcam_sites
    table = load_site_table()
    zs = set(zcam_sites(table))
    out = {}
    for g in groups:
        for s in dict.fromkeys(table["groups"].get(g, [])):
            if zcam and s not in zs:
                continue
            w = work_folder(root, s, zcam, table["sites"][s]["sols"])
            if w.is_dir():
                out[s] = w
    return out


def infos_of(blocks, need_finished=True, log=print):
    from mppp.sfm.zcam_calib import block_info
    out = []
    for s, w in blocks.items():
        try:
            i = block_info(w)
        except FileNotFoundError as e:
            log(f"  {s}: {e}")
            continue
        if need_finished and not i.get("run_done"):
            log(f"  {s}: no run_done.json - left out")
            continue
        out.append(i)
    return out


def promote(candidate_dir: Path, note: str) -> dict:
    """promote_cmods.py + a local commit + _transfer/promoted.json (as the Navcam consensus)."""
    from mppp.sfm.navcal_consensus import promote_candidate
    return promote_candidate(candidate_dir, note, ROOT)


def _out_dir(root: Path, name: str) -> Path:
    d = root / "camera_analysis" / f"{name}_{datetime.date.today():%Y%m%d}"
    n = 1
    base = d
    while d.exists() and any(d.iterdir()):
        n += 1
        d = base.with_name(f"{base.name}_{n}")
    d.mkdir(parents=True, exist_ok=True)
    return d


def cmd_start_update(a) -> int:
    from mppp.sfm.backlash import FOUND_FILE, append_found, load_low_backlash
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.sfm.zcam_calib import check_model, low_backlash_groups, report_lines, start_model_from_blocks
    root = Path(a.root)
    blocks = blocks_of(root, a.groups, zcam=True)
    print(f"start update from {len(blocks)} Navcam + Mastcam-Z blocks: {', '.join(blocks)}", flush=True)
    infos = infos_of(blocks)
    infos = [i for i in infos if (((i.get("checks") or {}).get("registered_fraction") or {}).get("value") or 0) >= 0.9]
    if not infos:
        print("no finished block: nothing to do")
        return 1
    found = root / FOUND_FILE
    listed = load_low_backlash(None, found=found)
    new = low_backlash_groups(infos, z_min=a.z_min, listed=listed, shipped=load_low_backlash(None))
    n = sum(append_found(found, [g], site=g["block"]) for g in new) if not a.dry_run else 0
    print(f"low backlash: {len(new)} newly classified focus groups -> {n} observations added to {found}", flush=True)
    for g in new:
        print(f"  {g['block']}: {g['camera_group']} sol {g['sol']} {g['sequence']} focus {g['focus_count']} "
              f"f/label {g['ratio']:.4f} (z {g['z']:+.1f}), {len(g['names'])} images", flush=True)
    mpath = zcam_focus_model_path()
    old = json.loads(mpath.read_text(encoding="utf-8"))
    model = start_model_from_blocks(infos, old)
    chk = check_model(model, old)
    out = _out_dir(root, "zcam34_start")
    (out / "M2020_ZCAM034_focus_model.json").write_text(json.dumps(model, indent=1, default=float), encoding="utf-8")
    (out / "check.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
    (out / "report.md").write_text("\n".join(report_lines(model) + ["## Checks", ""] +
                                             [f"- {'ok' if c['ok'] else 'FAIL'} {c['check']}: {c['note']}" for c in chk["checks"]]),
                                   encoding="utf-8")
    print("\n".join(report_lines(model)), flush=True)
    for c in chk["checks"]:
        print(f"  {'ok  ' if c['ok'] else 'FAIL'} {c['check']}: {c['note']}", flush=True)
    print(f"candidate: {out}", flush=True)
    if not chk["ok"]:
        print("not promoted: a check failed (the focus model in use stays)", flush=True)
        return 1
    if a.promote and not a.dry_run:
        r = promote(out, f"v0p74 Mastcam-Z 34 mm start model from {len(infos)} blocks (start-update)")
        if r.get("returncode"):
            return 2
        print("promoted", flush=True)
    return 0


def cmd_grade(a, quiet=False):
    from mppp.sfm.zcam_calib import select_blocks
    root = Path(a.root)
    infos = infos_of(blocks_of(root, a.groups, zcam=True))
    if a.v0p74_only:
        infos = [i for i in infos if i.get("line_params") and any(
            isinstance(v, dict) and v.get("lines") for v in (i.get("lines") or {}).values())]
    sel, grades = select_blocks(infos, n=a.n)
    if not quiet:
        for g in sorted(grades, key=lambda g: -g["score"]):
            print(f"  {'PASS' if g['ok'] else 'fail'} {g['score']:6.2f}  {g['name']}  " + "; ".join(g["reasons"]), flush=True)
        print("selected: " + (", ".join(i["name"] for i in sel) or "none"), flush=True)
    return infos, sel, grades


def cmd_calibrate(a) -> int:
    from mppp.sfm.project import zcam_focus_model_path
    from mppp.sfm.zcam_calib import calibrate, check_model, report_lines
    from mppp.sfm.zcam_lines import plot_block_lines
    root = Path(a.root)
    a.v0p74_only = True
    infos, sel, grades = cmd_grade(a)
    out = _out_dir(root, "zcam34_calibration")
    (out / "grades.json").write_text(json.dumps(grades, indent=1, default=float), encoding="utf-8")
    if len(sel) < 2:
        print(f"only {len(sel)} passing v0p74 blocks: no calibration (see {out / 'grades.json'})", flush=True)
        return 1
    mpath = zcam_focus_model_path()
    old = json.loads(mpath.read_text(encoding="utf-8"))
    model = calibrate(sel, old)
    chk = check_model(model, old)
    (out / "M2020_ZCAM034_focus_model.json").write_text(json.dumps(model, indent=1, default=float), encoding="utf-8")
    (out / "check.json").write_text(json.dumps(chk, indent=1), encoding="utf-8")
    rep = report_lines(model, grades) + ["## Checks", ""] + \
        [f"- {'ok' if c['ok'] else 'FAIL'} {c['check']}: {c['note']}" for c in chk["checks"]]
    # every block's lines (not only the selected), for the figure: offsets and slopes against the lens temperature
    rows = []
    for i in infos:
        for g, L in (i.get("lines") or {}).items():
            for st, ln in ((L.get("lines") or {}) if isinstance(L, dict) else {}).items():
                if not isinstance(ln, dict) or "skipped" in ln:
                    continue
                for pn, e in ln.items():
                    if isinstance(e, dict) and "a" in e:
                        k = (i["navcam_scale"] or 1.0) if pn in ("fx", "fy") else 1.0
                        rows.append({"block": i["name"].replace("mars2020_", ""), "group": g, "state": st, "param": pn,
                                     "a": e["a"] / k, "b": e["b"] / k, "sd_a": (e.get("sd_a") or 0) / k,
                                     "sd_b": (e.get("sd_b") or 0) / k, "T_optics_degC": i.get("T_optics_degC")})
    (out / "block_lines.json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        plot_block_lines(rows, out / "zcam_focus_lines_vs_temperature.png")
        rep += ["", "![focus lines against the lens temperature](zcam_focus_lines_vs_temperature.png)"]
    except Exception as e:                                                 # noqa: BLE001
        rep += ["", f"(figure not made: {e})"]
    (out / "report.md").write_text("\n".join(rep), encoding="utf-8")
    print("\n".join(rep), flush=True)
    print(f"calibration: {out}", flush=True)
    if not chk["ok"]:
        print("not promoted: a check failed", flush=True)
        return 1
    if a.promote and not a.dry_run:
        r = promote(out, f"v0p74 Mastcam-Z 34 mm calibration from {len(sel)} blocks ({', '.join(i['name'] for i in sel)})")
        if r.get("returncode"):
            return 2
        print("promoted", flush=True)
    return 0


def cmd_clean_found(a) -> int:
    """v0p74.8: remove from the found list the observations whose sequence is backlash as a whole (one state per
    sequence; the shipped list holds the known exceptions).  A dated backup of the file is kept next to it."""
    from mppp.sfm.backlash import FOUND_FILE, load_low_backlash
    from mppp.sfm.zcam_calib import found_contradicted
    root = Path(a.root)
    found = root / FOUND_FILE
    if not found.is_file():
        print(f"no found list at {found}")
        return 0
    infos = infos_of(blocks_of(root, a.groups, zcam=True), need_finished=False)
    res = found_contradicted(infos, found, shipped=load_low_backlash(None), z_min=a.z_min)
    seqs = {}
    for s, blk, k, z in res["remove"]:
        seqs.setdefault((blk, k, z), []).append(s)
    for (blk, k, z), ss in sorted(seqs.items()):
        print(f"  remove {len(ss)}: {blk} sol|sequence {k} is backlash (pooled z {(z if z is not None else float('nan')):+.1f})")
    print(f"found list: {len(res['remove'])} to remove, {len(res['keep'])} kept, {len(res['unknown'])} in no block (kept)")
    if not res["remove"] or a.dry_run:
        return 0
    drop = {s.upper() for s, *_ in res["remove"]}
    lines = found.read_text(encoding="utf-8-sig").splitlines()
    bak = found.with_name(found.stem + f"_{datetime.datetime.now():%Y%m%d_%H%M%S}.bak.csv")
    shutil.copy2(found, bak)
    keep = [ln for ln in lines if ln.split("#", 1)[0].strip().upper() not in drop]
    found.write_text("\n".join(keep) + "\n", encoding="utf-8")
    print(f"written {found} (backup {bak})")
    return 0


def cmd_pointclouds(a) -> int:
    from mppp.sfm.zcam_calib import export_point_cloud
    root = Path(a.root)
    index = []
    for zc in (False, True):
        for s, w in blocks_of(root, a.groups, zcam=zc).items():
            if not (w / "colmap" / "run_done.json").is_file():
                continue
            try:
                m = export_point_cloud(w)
                rd = json.loads((w / "colmap" / "run_done.json").read_text(encoding="utf-8"))
                m.update(site=s, block=w.name, verdict=rd.get("verdict"), mppp=rd.get("mppp"))
                index.append(m)
                print(f"  {w.name}: {m['points']} points -> {m['ply']}", flush=True)
            except Exception as e:                                         # noqa: BLE001
                print(f"  {w.name}: no point cloud ({type(e).__name__}: {e})", flush=True)
    d = root / "pointclouds"
    d.mkdir(exist_ok=True)
    (d / "index.json").write_text(json.dumps(index, indent=1, default=str), encoding="utf-8")
    print(f"{len(index)} point clouds; index {d / 'index.json'}", flush=True)
    return 0 if index else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["start-update", "calibrate", "grade", "pointclouds", "clean-found"])
    ap.add_argument("--root", default="D:/scapes/colmap")
    ap.add_argument("--groups", nargs="*", default=["zcam34_consensus"])
    ap.add_argument("--n", type=int, default=3, help="calibrate: the number of best blocks")
    ap.add_argument("--z-min", type=float, default=3.0, help="start-update: |z| of a low-backlash group")
    ap.add_argument("--promote", action="store_true", help="promote the result when every check passes")
    ap.add_argument("--dry-run", action="store_true", help="write the candidate, promote nothing, add nothing")
    a = ap.parse_args(argv)
    a.v0p74_only = False
    try:
        return {"start-update": cmd_start_update, "calibrate": cmd_calibrate,
                "grade": lambda a: (cmd_grade(a), 0)[1], "pointclouds": cmd_pointclouds,
                "clean-found": cmd_clean_found}[a.cmd](a)
    except Exception as e:                                                 # noqa: BLE001
        import traceback
        traceback.print_exc()
        print(f"ERROR: {type(e).__name__}: {e}", flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())

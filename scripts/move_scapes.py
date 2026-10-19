"""
v0p85 (Christian, 19 Oct 2026): move the block folders from ``D:\\scapes\\colmap`` to ``D:\\scapes`` and drop
``_colmap`` from their names (``mars2020_sol_1408_south_arm_colmap_zcam`` -> ``mars2020_sol_1408_south_arm_zcam``).

    python scripts\\move_scapes.py              # dry run: prints what it would move
    python scripts\\move_scapes.py --apply      # moves (renames on the same drive: instant, nothing is copied)

* Block folders get the v0p85 name.  Where the site's first sol in ``sites.json`` changed (e.g. Bell Island Large
  1435 -> 1431), the folder takes the new first sol, so the next run reuses its processed images - unless a folder
  with that sol already exists (then only ``_colmap`` is dropped).
* Inside each moved block, the absolute paths in ``*/project.json``, ``*/run_done.json`` and ``colmap_gui.ini`` are
  rewritten to the new folder.
* Everything else in the old root (``camera_analysis``, ``mppp_pipeline``, ``pointclouds``, ``error_analysis``,
  ``_jobs``, loose files) moves to the new root; a folder that already exists there is merged entry by entry.
* Nothing is overwritten: a name that exists at the destination is left where it is and listed as a conflict.
  Safe to run again (it only moves what is still in the old root).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from mppp.sfm import sites as S                                  # noqa: E402


def new_block_name(folder: Path, table: Dict, taken: set) -> str:
    site, zcam = S._site_of_folder(folder)
    n = folder.name
    stripped = n.replace(S.OLD_ZCAM_SUFFIX, S.ZCAM_SUFFIX) if n.endswith(S.OLD_ZCAM_SUFFIX) else (
        n[: -len(S.OLD_NAV_SUFFIX)] if n.endswith(S.OLD_NAV_SUFFIX) else n)
    if site in table:
        cur = S.folder_name(site, S.sol_span(table[site])[0], zcam)
        if cur != stripped and cur not in taken:
            return cur
    return stripped


def rewrite_paths(block: Path, old: Path, new: Path, apply: bool) -> int:
    """Old -> new absolute folder path in the block's project files (both slash forms, JSON-escaped too)."""
    pairs = []
    for o, w in ((str(old), str(new)), (old.as_posix(), new.as_posix())):
        pairs += [(o, w), (o.replace("\\", "\\\\"), w.replace("\\", "\\\\"))]
    files = [p for pat in ("*/project.json", "*/run_done.json", "*/colmap_gui.ini", "*/gui_native/colmap_gui.ini")
             for p in block.glob(pat)]
    n = 0
    for f in files:
        try:
            s = f.read_text(encoding="utf-8")
        except Exception:                                          # noqa: BLE001
            continue
        t = s
        for a, b in pairs:
            t = t.replace(a, b)
        if t != s:
            n += 1
            if apply:
                f.write_text(t, encoding="utf-8")
    return n


def plan(old_root: Path, new_root: Path, table: Dict) -> Tuple[List[Tuple[Path, Path, str]], List[str]]:
    moves, conflicts = [], []
    taken = {p.name for p in new_root.iterdir()} if new_root.is_dir() else set()
    entries = sorted(old_root.iterdir()) if old_root.is_dir() else []
    blocks = [p for p in entries if p.is_dir() and S._site_of_folder(p)[0] is not None]
    # folders whose name already carries the current first sol go first, so they keep it
    def keeps_sol(p: Path) -> bool:
        site, z = S._site_of_folder(p)
        return site in table and new_block_name(p, table, set()) == new_block_name(p, {}, set())
    blocks.sort(key=lambda p: (not keeps_sol(p), p.name))
    for p in blocks:
        name = new_block_name(p, table, taken)
        if name in taken:
            conflicts.append(f"{p} -> {new_root / name}: exists, left in place")
            continue
        taken.add(name)
        moves.append((p, new_root / name, "block"))
    for p in entries:
        if p in blocks:
            continue
        dest = new_root / p.name
        if not dest.exists():
            moves.append((p, dest, "dir" if p.is_dir() else "file"))
            taken.add(p.name)
        elif p.is_dir() and dest.is_dir():
            for c in sorted(p.iterdir()):
                if (dest / c.name).exists():
                    conflicts.append(f"{c} -> {dest / c.name}: exists, left in place")
                else:
                    moves.append((c, dest / c.name, "merge"))
        else:
            conflicts.append(f"{p} -> {dest}: exists, left in place")
    return moves, conflicts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--old-root", default=S.OLD_SCAPES_ROOT)
    ap.add_argument("--new-root", default=S.DEFAULT_SCAPES_ROOT)
    ap.add_argument("--apply", action="store_true", help="move (default: dry run)")
    a = ap.parse_args()
    old_root, new_root = Path(a.old_root), Path(a.new_root)
    if not old_root.is_dir():
        print(f"{old_root}: not found - nothing to move")
        return 0
    table = S.load_sites()
    moves, conflicts = plan(old_root, new_root, table)
    if not moves and not conflicts:
        print(f"{old_root}: nothing left to move")
        return 0
    print(("Moving" if a.apply else "Would move (dry run; --apply to move)") + f" {len(moves)} entries "
          f"from {old_root} to {new_root}:")
    log = []
    for src, dst, kind in moves:
        renamed = "" if src.name == dst.name else "   (renamed)"
        print(f"  {kind:5s}  {src.name}  ->  {dst}{renamed}")
        if a.apply:
            src.rename(dst)
            n = rewrite_paths(dst, src, dst, True) if kind == "block" else 0
            log.append({"from": str(src), "to": str(dst), "kind": kind, "files_rewritten": n})
    for c in conflicts:
        print(f"  CONFLICT  {c}")
    if a.apply:
        out = new_root / "mppp_pipeline"
        out.mkdir(parents=True, exist_ok=True)
        (out / "move_scapes_v0p85.json").write_text(json.dumps({"moves": log, "conflicts": conflicts}, indent=1),
                                                    encoding="utf-8")
        print(f"done: {len(log)} moved, {len(conflicts)} conflicts; log {out / 'move_scapes_v0p85.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

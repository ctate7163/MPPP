"""What takes the space in a blocks root, and the removal of what can be made again (0.99.24).

Christian, 18 Nov 2026: "To make space, let's remove the high data volume products from D:\\rover_blocks starting with
the colmap databases and the redundant images."

    python -m rpp_studies.misc.free_space --root D:\\rover_blocks                       (the report, nothing removed)
    python -m rpp_studies.misc.free_space --root D:\\rover_blocks --delete colmap-db --confirm DELETE

The report (``free_space_report.md`` in ``--out``, default the root's ``mppp_pipeline``): bytes per category, per
block, and per folder two levels below each block, and what each category's removal would free.

Categories:

* ``colmap-db`` - the SQLite files of feature extraction and matching (``*.db`` and their ``-wal``, ``-shm``,
  ``-journal``; ``features.db``, ``database.db``, the guided-matching databases, the ``matches`` library).  Made again
  by aligning the block again (hours per large block).  Nothing downstream of the alignment reads them: the camera
  joints, the dense stage and the panoramas read the models (``sparse``) and the images.
* ``dense`` - the folders ``dense*`` under a block's ``colmap`` (undistorted images, depth and normal maps, the fused
  cloud).  Made again by the dense stage (Three Forks South: 133 min of stereo for the block).
* ``images`` - the image files.  Reported only.  A file with several hard links (a block's ``colmap\\images`` are links
  to its ``processed`` frames) takes its space once, and removing one of its names frees nothing: the report counts
  such a file once and gives, per folder, the bytes that the folder's removal alone would free.  ``copies`` are image
  files with the same name and size stored more than once (not as links): candidates for "redundant".

A removal is permanent (no recycle bin).  It needs ``--confirm DELETE``, takes only the categories named, skips files
that are open, and writes the list of what it removed beside the report.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DB_SUFFIXES = (".db", ".db-wal", ".db-shm", ".db-journal")
IMAGE_SUFFIXES = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".img", ".exr", ".npy", ".npz", ".bin")
DELETABLE = ("colmap-db", "dense")
SKIP_TOP = ("_transfer", "mppp_pipeline", "camera_analysis")      # not block folders: never touched by a removal


def category(rel: Tuple[str, ...]) -> str:
    """The category of a file from its path parts below the root."""
    name = rel[-1].lower()
    low = [p.lower() for p in rel[:-1]]
    if name.endswith(DB_SUFFIXES):
        return "colmap-db"
    if "colmap" in low and any(p.startswith("dense") for p in low[low.index("colmap") + 1:]):
        return "dense"
    if name.endswith(IMAGE_SUFFIXES) and not ("sparse" in low):
        return "images"
    return "other"


def scan(root: Path, progress: bool = True) -> List[Dict[str, Any]]:
    """Every file below ``root``: {"rel" (parts), "size", "cat", "key" ((device, inode) or None), "links"}."""
    out: List[Dict[str, Any]] = []
    stack = [(root, ())]
    t0 = time.time()
    while stack:
        d, rel = stack.pop()
        try:
            it = list(os.scandir(d))
        except OSError:
            continue
        for e in it:
            r = rel + (e.name,)
            try:
                if e.is_dir(follow_symlinks=False):
                    stack.append((Path(e.path), r))
                    continue
                if not e.is_file(follow_symlinks=False):
                    continue
                cat = category(r)
                st = os.stat(e.path) if cat in ("images", "dense") else e.stat()     # links: a full stat (Windows)
            except OSError:
                continue
            links = int(getattr(st, "st_nlink", 1) or 1)
            key = (st.st_dev, st.st_ino) if links > 1 and st.st_ino else None
            out.append({"rel": r, "size": int(st.st_size), "cat": cat, "key": key, "links": links})
        if progress and len(out) % 20000 < len(it) and time.time() - t0 > 5:
            print(f"  {len(out):,} files ...", flush=True)
    return out


def freed_by(files: Sequence[Dict[str, Any]], chosen) -> int:
    """Bytes freed by removing the files for which ``chosen(f)``: a linked file counts when all its names go."""
    total, seen = 0, defaultdict(int)
    for f in files:
        if not chosen(f):
            continue
        if f["key"] is None:
            total += f["size"]
        else:
            seen[f["key"]] += 1
            if seen[f["key"]] == f["links"]:
                total += f["size"]
    return total


def on_disk(files: Sequence[Dict[str, Any]]) -> int:
    """Bytes of the files, a linked file once."""
    total, seen = 0, set()
    for f in files:
        if f["key"] is not None:
            if f["key"] in seen:
                continue
            seen.add(f["key"])
        total += f["size"]
    return total


def _gb(n: float) -> str:
    return f"{n / 2 ** 30:,.1f}"


def report(root: Path, files: Sequence[Dict[str, Any]], top: int = 30) -> str:
    L = [f"# Space in {root}", "",
         f"{len(files):,} files, {_gb(on_disk(files))} GB on disk (a hard-linked file counted once; "
         f"{_gb(sum(f['size'] for f in files))} GB by the sum of the names).", "",
         "## By category", "", "| category | files | GB on disk | GB freed by removing the category alone |", "|---|---|---|---|"]
    for cat in ("colmap-db", "dense", "images", "other"):
        mine = [f for f in files if f["cat"] == cat]
        L.append(f"| {cat} | {len(mine):,} | {_gb(on_disk(mine))} | {_gb(freed_by(files, lambda f, c=cat: f['cat'] == c))} |")
    blocks: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in files:
        blocks[f["rel"][0] if len(f["rel"]) > 1 else "(files in the root)"].append(f)
    L += ["", f"## The {top} largest blocks", "", "| block | GB on disk | colmap-db | dense | images | other |", "|---|---|---|---|---|---|"]
    for b, fs in sorted(blocks.items(), key=lambda kv: -on_disk(kv[1]))[:top]:
        L.append(f"| {b} | {_gb(on_disk(fs))} | " + " | ".join(_gb(on_disk([f for f in fs if f["cat"] == c]))
                                                             for c in ("colmap-db", "dense", "images", "other")) + " |")
    folders: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for f in files:
        if len(f["rel"]) > 2:
            folders["\\".join(("*",) + f["rel"][1:min(3, len(f["rel"]) - 1)])].append(f)
    L += ["", "## By folder inside the blocks (all blocks together)", "",
          "| folder | files | GB by names | GB freed by removing it alone | hard-linked files |", "|---|---|---|---|---|"]
    for k, fs in sorted(folders.items(), key=lambda kv: -sum(f["size"] for f in kv[1]))[:top]:
        keys = {id(f) for f in fs}
        L.append(f"| {k} | {len(fs):,} | {_gb(sum(f['size'] for f in fs))} | {_gb(freed_by(files, lambda f: id(f) in keys))} | "
                 f"{sum(1 for f in fs if f['key'] is not None):,} |")
    names: Dict[Tuple[str, int], List[Dict[str, Any]]] = defaultdict(list)
    for f in files:
        if f["cat"] == "images" and f["size"] > 2 ** 20:
            names[(f["rel"][-1].lower(), f["size"])].append(f)
    cop: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for (_, size), fs in names.items():
        inodes = {f["key"] or id(f) for f in fs}
        if len(inodes) < 2:
            continue
        for f in fs:
            k = "\\".join(("*",) + f["rel"][1:min(3, len(f["rel"]) - 1)])
            cop[k][0] += 1
            cop[k][1] += size
    L += ["", "## Copies: image files over 1 MB with the same name and size stored more than once (not as links)", ""]
    if cop:
        L += ["| folder | files that have a copy elsewhere | GB |", "|---|---|---|"]
        L += [f"| {k} | {v[0]:,} | {_gb(v[1])} |" for k, v in sorted(cop.items(), key=lambda kv: -kv[1][1])[:top]]
    else:
        L.append("none")
    return "\n".join(L) + "\n"


def delete(root: Path, files: Sequence[Dict[str, Any]], cats: Sequence[str], listing: Path,
           keep: Sequence[str] = ()) -> Dict[str, Any]:
    """Remove the files of the categories ``cats``; not below the folders ``SKIP_TOP`` nor below a top folder whose
    name starts with one of ``keep``."""
    done, skipped, freed = [], [], 0
    left: Dict[Any, int] = defaultdict(int)
    for f in files:
        if f["cat"] not in cats or f["rel"][0] in SKIP_TOP or f["rel"][0].lower().startswith(tuple(k.lower() for k in keep) or ("\0",)):
            continue
        p = root.joinpath(*f["rel"])
        try:
            os.remove(p)
        except OSError as e:
            skipped.append(f"{p}  ({type(e).__name__})")
            continue
        done.append(str(p))
        if f["key"] is None:
            freed += f["size"]
        else:
            left[f["key"]] += 1
            if left[f["key"]] == f["links"]:
                freed += f["size"]
    listing.write_text("\n".join([f"removed from {root} on {time.strftime('%Y-%m-%d %H:%M')}: categories {', '.join(cats)}; "
                                  f"{len(done):,} files, {_gb(freed)} GB freed; {len(skipped)} skipped (open or protected)"]
                                 + done + ["", "skipped:"] + skipped) + "\n", encoding="utf-8")
    return {"removed": len(done), "skipped": len(skipped), "freed": freed}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default=None, help="folder of the report (default ROOT\\mppp_pipeline)")
    ap.add_argument("--delete", nargs="*", default=[], choices=DELETABLE, metavar="CATEGORY",
                    help=f"remove these categories ({', '.join(DELETABLE)}); permanent")
    ap.add_argument("--confirm", default="", help="DELETE: needed with --delete")
    ap.add_argument("--keep", nargs="*", default=[], metavar="PREFIX",
                    help="--delete: not below the top folders whose names start so (msl_: the MSL block's stages read its database)")
    ap.add_argument("--top", type=int, default=30)
    a = ap.parse_args(argv)
    root = Path(a.root)
    if not root.is_dir():
        print(f"{root} not found")
        return 1
    out = Path(a.out) if a.out else root / "mppp_pipeline"
    out.mkdir(parents=True, exist_ok=True)
    print(f"scanning {root} ...", flush=True)
    files = scan(root)
    txt = report(root, files, a.top)
    (out / "free_space_report.md").write_text(txt, encoding="utf-8")
    print(txt.split("## The")[0])
    print(f"report: {out / 'free_space_report.md'}")
    if not a.delete:
        return 0
    if a.confirm != "DELETE":
        print("nothing removed: --delete needs --confirm DELETE (the removal is permanent)")
        return 2
    res = delete(root, files, a.delete, out / f"free_space_removed_{time.strftime('%Y%m%d_%H%M%S')}.txt", keep=a.keep)
    print(f"removed {res['removed']:,} files ({', '.join(a.delete)}), {_gb(res['freed'])} GB freed; {res['skipped']} skipped")
    return 0


if __name__ == "__main__":
    sys.exit(main())

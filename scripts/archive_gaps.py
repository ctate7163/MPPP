"""
Stereo exposures with one eye missing from the local PDS archive (0.94.4, the weekend run of 23 Oct 2026): the
Navcam and Hazcam RAD products whose partner (the other eye, same SCLK) is not on disk, and optionally the
download of that partner from the PDS Imaging Node.

    python scripts\\archive_gaps.py                         # MPPP_PDS; writes <pds>\\archive_gaps.csv
    python scripts\\archive_gaps.py --sols 113 130          # a sol range
    python scripts\\archive_gaps.py --download              # and fetch the missing eyes (highest version)

Seitah Overlook (sols 122-124) and Artuby Ridge (sol 173) had right Hazcam images (J01) whose left image was not
in the archive, while the PDS has both eyes of those exposures (J02).  MPPP aligns such an image in a one-camera rig
of its own; with both eyes it is a stereo pair.  The pipeline logs these exposures too ("stereo exposures with one
eye only").  Downloads go next to the product that is there (same folder), never replacing a file.
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

BASE = "https://planetarydata.jpl.nasa.gov/img/data/mars2020"
COLLECTION = {"N": ("mars2020_navcam_ops_calibrated", "ncam"), "F": ("mars2020_hazcam_ops_calibrated", "fcam"),
              "R": ("mars2020_hazcam_ops_calibrated", "rcam")}
NAME = re.compile(r"^([NFR])([LR])F_(\d{4})_(\d{10}_\d{3})RAD_", re.I)


def rad_products(pds: Path, sols: Optional[Tuple[int, int]] = None) -> Iterable[Path]:
    """The Navcam and Hazcam RAD ``.IMG`` products below ``pds`` (full frames and sub-frames alike)."""
    for dirpath, _, files in os.walk(pds):
        for f in files:
            m = NAME.match(f)
            if not m or not f.upper().endswith(".IMG"):
                continue
            if sols and not sols[0] <= int(m.group(3)) <= sols[1]:
                continue
            yield Path(dirpath) / f


def gaps(paths: Iterable[Path]) -> List[Dict[str, str]]:
    """One row per exposure (family, sol, SCLK) that has products of one eye only: the eye present, its files, the
    missing eye and the name prefix its products start with."""
    by: Dict[Tuple[str, str, str], Dict[str, List[Path]]] = {}
    for p in paths:
        m = NAME.match(p.name)
        fam, eye, sol, sclk = m.group(1).upper(), m.group(2).upper(), m.group(3), m.group(4)
        by.setdefault((fam, sol, sclk), {}).setdefault(eye, []).append(p)
    rows = []
    for (fam, sol, sclk), eyes in sorted(by.items()):
        if len(eyes) != 1:
            continue
        have = next(iter(eyes))
        miss = "L" if have == "R" else "R"
        rows.append({"family": fam, "sol": int(sol), "sclk": sclk, "have": f"{fam}{have}", "missing": f"{fam}{miss}",
                     "prefix": f"{fam}{miss}F_{sol}_{sclk}RAD_", "folder": str(eyes[have][0].parent),
                     "files_present": ";".join(sorted(p.name for p in eyes[have]))})
    return rows


def listing_url(fam: str, sol: int) -> str:
    coll, sub = COLLECTION[fam]
    return f"{BASE}/{coll}/data/sol/{sol:05d}/ids/rdr/{sub}/"


def names_in_listing(html: str, prefix: str) -> List[str]:
    """The ``.IMG`` names in a directory listing that start with ``prefix``, highest version first."""
    found = sorted(set(re.findall(r'href="(' + re.escape(prefix) + r'[A-Za-z0-9_]*\.IMG)"', html, re.I)))
    return sorted(found, key=lambda n: n[-7:-4], reverse=True)      # ..J02.IMG before ..J01.IMG


def _get(url: str, timeout: float = 120.0) -> bytes:
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "MPPP archive_gaps"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def download(rows: Sequence[Dict[str, str]], get=_get, log=print, all_versions: bool = False) -> int:
    """Fetch the missing eye of every row (its highest version; ``all_versions``: every product of that exposure)
    into the folder of the eye that is there.  Returns the number of files written."""
    listings: Dict[str, str] = {}
    n = 0
    for r in rows:
        url = listing_url(r["family"], int(r["sol"]))
        if url not in listings:
            try:
                listings[url] = get(url).decode("utf-8", "replace")
            except Exception as e:                       # noqa: BLE001 - logged, the next sol is tried
                log(f"  {url}: {type(e).__name__}: {e}")
                listings[url] = ""
        names = names_in_listing(listings[url], r["prefix"])
        if not names:
            r["downloaded"] = "not in the PDS listing"
            continue
        if not all_versions:
            # per product kind of the eye that is there (full frame, sub-frame, ...: the name between "RAD_" and
            # the version), the highest version of the other eye's; else the highest version of any kind
            kinds = {f.split("RAD_", 1)[1][:-7] for f in r["files_present"].split(";") if "RAD_" in f}
            pick = []
            for k in sorted(kinds):
                same = [x for x in names if x.split("RAD_", 1)[1][:-7] == k]
                if same:
                    pick.append(same[0])
            names = pick or names[:1]
        got = []
        for name in names:
            dst = Path(r["folder"]) / name
            if dst.exists():
                got.append(f"{name} (there)")
                continue
            tmp = dst.with_suffix(".part")
            try:
                tmp.write_bytes(get(url + name))
                tmp.replace(dst)
                n += 1
                got.append(name)
                log(f"  {dst}")
            except Exception as e:                       # noqa: BLE001
                log(f"  {name}: {type(e).__name__}: {e}")
                tmp.unlink(missing_ok=True)
            time.sleep(0.2)
        r["downloaded"] = ";".join(got)
    return n


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pds", default=os.environ.get("MPPP_PDS"), help="the PDS archive (default MPPP_PDS)")
    ap.add_argument("--sols", type=int, nargs=2, default=None, metavar=("FIRST", "LAST"))
    ap.add_argument("--download", action="store_true", help="fetch the missing eyes from the PDS Imaging Node")
    ap.add_argument("--all-versions", action="store_true", help="with --download: every version, not the highest")
    ap.add_argument("--out", default=None, help="the CSV (default <pds>/archive_gaps.csv)")
    a = ap.parse_args(argv)
    if not a.pds or not Path(a.pds).is_dir():
        print("ERROR: --pds (or MPPP_PDS) must be the PDS archive folder", file=sys.stderr)
        return 2
    rows = gaps(rad_products(Path(a.pds), tuple(a.sols) if a.sols else None))
    by_fam: Dict[str, int] = {}
    for r in rows:
        by_fam[r["missing"]] = by_fam.get(r["missing"], 0) + 1
    print(f"{len(rows)} stereo exposures with one eye only: " + (", ".join(f"{k} missing {v}" for k, v in sorted(by_fam.items())) or "none"))
    if a.download and rows:
        n = download(rows, all_versions=a.all_versions)
        print(f"{n} files downloaded")
    out = Path(a.out) if a.out else Path(a.pds) / "archive_gaps.csv"
    if rows:
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            w.writerows(rows)
        print(f"list: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

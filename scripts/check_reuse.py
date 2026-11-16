"""
0.99.0: does the rename MPPP -> RPP change what would run?  A dry run of every site (Navcam, and with ``--zcam`` the
Mastcam-Z blocks) on a scapes root with this tree and with a v0p98 tree (``--old``, default ``D:\\code\\MPPP``), then the
two lists side by side.  Nothing is processed or aligned and nothing in the WORK folders is written (``run_sites.py
--dry-run`` reads each block's ``run_done.json``; it appends to ``<root>/run_sites_log.txt``).

    python scripts/check_reuse.py [--root D:\\rover_blocks] [--old D:\\code\\MPPP]

Exit 0 when every site gets the same answer from both ("skipped - finished with these settings" from both, or "would
run" from both), 1 otherwise (the differences are listed).
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "src"))
from rpp import env as rpp_env  # noqa: E402
LINE = re.compile(r"^(?:\S*\d\d:\d\d:\d\d\S*\s+)?(?P<site>[A-Za-z0-9_]+): (?P<what>skipped - .*|would run \(.*?\))(?: -> .*)?$")


def parse(text: str) -> Dict[str, str]:
    """{site: "skipped - <why>" | "would run (<why>)"} from the output of ``run_sites.py --dry-run``."""
    out: Dict[str, str] = {}
    for line in text.splitlines():
        m = LINE.match(line.strip())
        if m:
            out[m.group("site")] = m.group("what").strip()
    return out


def dry_run(tree: Path, root: str, zcam: bool, old: bool) -> Tuple[Dict[str, str], str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(tree / "src")
    env["PYTHONUNBUFFERED"] = "1"
    if old:          # the v0p98 tree reads MPPP_*: give it the RPP_* values this prompt has
        for k, v in list(os.environ.items()):
            if k.startswith("RPP_") and "MPPP_" + k[4:] not in env and k != "RPP_HOME":
                env["MPPP_" + k[4:]] = v
    cmd = [sys.executable, str(tree / "scripts" / "run_sites.py"), "--all", "--dry-run", "--root", root]
    if zcam:
        cmd.append("--zcam")
    r = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8", errors="replace")
    text = r.stdout + r.stderr
    return parse(text), text


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=rpp_env.get("RPP_SCAPES") or r"D:\rover_blocks")
    ap.add_argument("--old", default=r"D:\code\MPPP", help="the v0p98 tree")
    ap.add_argument("--log", default=None, help="write both outputs here (default: %%TEMP%%\\rpp_check_reuse.txt)")
    a = ap.parse_args(argv)
    new_tree, old_tree = HERE.parent, Path(a.old)
    log = Path(a.log or Path(os.environ.get("TEMP") or "/tmp") / "rpp_check_reuse.txt")
    bad, n, raw = [], 0, []
    for zcam in (False, True):
        new, tn = dry_run(new_tree, a.root, zcam, old=False)
        old, to = dry_run(old_tree, a.root, zcam, old=True)
        raw += [f"==== {'zcam' if zcam else 'navcam'} - {new_tree}", tn, f"==== {'zcam' if zcam else 'navcam'} - {old_tree}", to]
        kind = "Mastcam-Z" if zcam else "Navcam"
        if not new or not old:
            bad.append(f"{kind}: no site lines from {'this tree' if not new else old_tree} (see {log})")
            continue
        for s in sorted(set(new) | set(old)):
            n += 1
            a_, b_ = new.get(s, "(not listed)"), old.get(s, "(not listed)")
            same = a_.split(" ", 1)[0] == b_.split(" ", 1)[0]
            print(f"{'  ' if same else '!!'} {kind:9s} {s:34s} v0p99: {a_[:60]:60s} v0p98: {b_[:60]}")
            if not same:
                bad.append(f"{kind} {s}: v0p99 {a_} / v0p98 {b_}")
    log.write_text("\n".join(raw), encoding="utf-8")
    print(f"\n{n} site lines compared; both outputs in {log}")
    if bad:
        print("DIFFERENT:\n  " + "\n  ".join(bad))
        return 1
    print("the same: the rename changes nothing that would run")
    return 0


if __name__ == "__main__":
    sys.exit(main())

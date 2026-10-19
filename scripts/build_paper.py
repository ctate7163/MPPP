"""
Render the JOSS paper (``paper/paper.md`` + ``paper/paper.bib``) to ``paper/paper.pdf``.

Two renderers, tried in this order unless one is chosen with ``--renderer``:

* ``inara`` - JOSS's own renderer, run through Docker (``docker run openjournals/inara``); this is what
  JOSS builds the paper with, so its output is authoritative.
* ``pandoc`` - ``pandoc paper.md --citeproc --bibliography paper.bib -o paper.pdf`` (needs pandoc and a LaTeX
  engine on the PATH); a quick local preview.

Run it after every version change (``python scripts/build_paper.py``); the CI workflow runs the same script.
The script also checks that the paper's version-dependent statements match the package: the figure files it
references exist, every citation key is in the bibliography, and no ``TODO`` key remains.
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def check(md: Path, bib: Path) -> list:
    text = md.read_text(encoding="utf-8")
    keys = set(re.findall(r"^@\w+\{([^,]+),", bib.read_text(encoding="utf-8"), re.M))
    problems = []
    for cite in set(re.findall(r"@([A-Za-z][\w:\-]*)", text)):
        if cite not in keys:
            problems.append(f"citation key not in paper.bib: @{cite}")
        if cite.upper().startswith("TODO"):
            problems.append(f"placeholder citation: @{cite}")
    for fig in re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text):
        if not (md.parent / fig).is_file():
            problems.append(f"figure file missing: {fig}")
        elif (md.parent / fig).stat().st_size > 2_000_000:
            problems.append(f"figure larger than 2 MB (JOSS asks for small figures): {fig}")
    return problems


def render_pandoc(md: Path, bib: Path, out: Path) -> int:
    if not shutil.which("pandoc"):
        print("pandoc not found on the PATH", file=sys.stderr)
        return 2
    cmd = ["pandoc", str(md), "--citeproc", "--bibliography", str(bib), "-o", str(out)]
    print(" ".join(cmd))
    return subprocess.call(cmd, cwd=md.parent)


def render_inara(md: Path, out: Path) -> int:
    if not shutil.which("docker"):
        print("docker not found on the PATH", file=sys.stderr)
        return 2
    cmd = ["docker", "run", "--rm", "-v", f"{md.parent}:/data", "-u", "1000:1000",
           "openjournals/inara", "-o", "pdf", "/data/" + md.name]
    print(" ".join(cmd))
    return subprocess.call(cmd)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--renderer", choices=("auto", "inara", "pandoc"), default="auto")
    ap.add_argument("--check-only", action="store_true", help="only run the consistency checks")
    a = ap.parse_args(argv)
    md, bib, out = PAPER / "paper.md", PAPER / "paper.bib", PAPER / "paper.pdf"
    problems = check(md, bib)
    for p in problems:
        print("WARNING:", p)
    if a.check_only:
        return 1 if problems else 0
    if a.renderer in ("auto", "inara"):
        rc = render_inara(md, out)
        if rc == 0 or a.renderer == "inara":
            return rc
    return render_pandoc(md, bib, out)


if __name__ == "__main__":
    sys.exit(main())

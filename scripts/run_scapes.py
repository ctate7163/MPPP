"""
Run the MPPP notebooks for several sites in one go (v0p22).  Since v0p43 ``scripts/run_sites.py`` does this with
the stable site definitions, per-site logs and status files, and skips finished sites; this script is kept for
its old command lines.

For every site, notebook 03 (COLMAP alignment) runs with its settings cell
(tagged ``parameters``) overridden; then notebook 04 (camera models) and 05
(error analysis) run once over all sites that finished (v0p31: the camera models come first, so that the
cameras are constrained before the reconstruction error is analysed).  Each executed notebook
is saved next to its results, and progress goes to ``<root>/batch_log.txt``
(one line per notebook cell), so a long run can be followed from anywhere.

Example (Windows, the Python environment that runs your notebooks)::

    python scripts\\run_scapes.py --sites taylorfjellet rockytop belva threeforks_south landing ^
        --zcam --reprocess --root D:\\scapes\\v0p22 ^
        --gpu-py C:\\Users\\<you>\\AppData\\Local\\miniconda3\\envs\\mppp_gpu\\python.exe

``--set NAME=VALUE`` overrides any other notebook-03 setting (VALUE is Python,
e.g. ``--set "MATCH=dict(max_ratio=0.85, guided_matching=True)"``).  A site
that fails is logged and skipped; the others continue.  ``--only 04 05`` reruns
just the analyses on finished sites.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks"
sys.path.insert(0, str(ROOT / "src"))
from mppp.runner import Log, inject, notebook, run_analyses, run_notebook  # noqa: E402  (v0p43: shared runner)


class _Labels(dict):
    """v0p40: labels from the site names (mppp.sfm.sites.site_label)."""
    def get(self, k, default=None):
        from mppp.sfm.sites import site_label
        return site_label(k)


SITE_LABEL = _Labels()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sites", nargs="+", required=True, help="keys of SITES in notebook 03")
    ap.add_argument("--root", default="D:/scapes/v0p22", help="SCAPES_ROOT: one folder per site below it")
    ap.add_argument("--zcam", action="store_true", help="include the Mastcam-Z 34 mm frames")
    ap.add_argument("--reprocess", action="store_true", help="process every image again")
    ap.add_argument("--gpu-py", default=None, help="python.exe of the CUDA pycolmap environment")
    ap.add_argument("--pds", default=None, help="PDS archive (default: notebook 03's PDS_DIR)")
    ap.add_argument("--set", action="append", default=[], metavar="NAME=VALUE", help="another notebook-03 setting")
    ap.add_argument("--only", nargs="+", default=["03", "04", "05"], help="which notebooks: 03 04 05")
    ap.add_argument("--mask-off", action="append", default=[], metavar="SITE=S032D1184",
                    help="station without mask inference for a site (repeatable)")
    a = ap.parse_args(argv)

    root = Path(a.root)
    log = Log(root / "batch_log.txt")
    log(f"MPPP batch: sites {a.sites}; root {root}; Mastcam-Z {a.zcam}; reprocess {a.reprocess}; notebooks {a.only}")
    import mppp
    log(f"MPPP {mppp.__version__} from {Path(mppp.__file__).parent}; python {sys.executable}")
    mask_off = {}
    for m in a.mask_off:
        k, v = m.split("=", 1)
        mask_off.setdefault(k, []).append(v)
    extra = {}
    for s in a.set:
        k, v = s.split("=", 1)
        extra[k.strip()] = v.strip()

    done = []
    for site in a.sites:
        work = root / (f"{site}_colmap_nav_zcam34" if a.zcam else f"{site}_colmap")
        if "03" in a.only:
            ov = {"SITE": repr(site), "SCAPES_ROOT": f"Path(r{str(root)!r})", "INCLUDE_ZCAM34": repr(bool(a.zcam)),
                  "REPROCESS_ALL": repr(bool(a.reprocess)), "STORE_MASK_IN_ALPHA": "True",
                  "NO_MASK_INFERENCE_AT": repr(mask_off.get(site, []))}
            if a.gpu_py:
                ov["GPU_PY"] = f"r{a.gpu_py!r}"
            if a.pds:
                ov["PDS_DIR"] = f"Path(r{a.pds!r})"
            ov.update(extra)
            log(f"site {site}: notebook 03 -> {work}")
            try:
                ok = run_notebook(notebook("03_colmap_alignment"), work / "03_colmap_alignment_executed.ipynb", ov, log)
            except Exception:                                        # noqa: BLE001
                ok = False
                log(f"site {site}: runner error\n{traceback.format_exc()}")
            if not ok:
                continue
        if (work / "colmap" / "error_input" / "summary.json").is_file():
            done.append((site, work))

    labels = {f"{SITE_LABEL.get(s, s)}{' + Mastcam-Z 34' if a.zcam else ''}": str(w) for s, w in done}
    (root / "batch_sites.json").write_text(json.dumps(labels, indent=1), encoding="utf-8")
    if not labels:
        log("no finished site: notebooks 04 and 05 skipped")
        return 1
    run_analyses(labels, root, a.only, log)
    log("batch finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())

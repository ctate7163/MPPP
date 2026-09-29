"""
Run the MPPP notebooks for several sites in one go (v0p22).

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


def notebook(stem: str):
    """``notebooks/<stem>.ipynb``, or the highest-versioned ``<stem>_v0pXXpY.ipynb`` beside it (v0p22.4: the
    notebooks are also shipped with the MPPP version in their file name)."""
    import re
    plain = NB / f"{stem}.ipynb"
    versioned = sorted(NB.glob(f"{stem}_v0p*.ipynb"),
                       key=lambda p: [int(x) for x in re.findall(r"\d+", p.stem[len(stem):])])
    if plain.is_file():
        return plain
    if versioned:
        return versioned[-1]
    raise FileNotFoundError(f"no {stem}.ipynb or {stem}_v0p*.ipynb in {NB}")
class _Labels(dict):
    """v0p40: labels from the site names (mppp.sfm.sites.site_label; imported once src/ is on the path)."""
    def get(self, k, default=None):
        from mppp.sfm.sites import site_label
        return site_label(k)


SITE_LABEL = _Labels()


class Log:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{datetime.datetime.now().isoformat(timespec='seconds')}  {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def inject(nb, overrides: dict) -> None:
    """Insert a cell setting ``overrides`` (name -> Python source) right after the cell tagged ``parameters``."""
    import nbformat
    idx = next((i for i, c in enumerate(nb.cells) if "parameters" in c.get("metadata", {}).get("tags", [])), None)
    if idx is None:
        raise ValueError("notebook has no cell tagged 'parameters'")
    src = "# injected by scripts/run_scapes.py\n" + "\n".join(f"{k} = {v}" for k, v in overrides.items())
    cell = nbformat.v4.new_code_cell(src)
    cell.metadata["tags"] = ["injected-parameters"]
    nb.cells.insert(idx + 1, cell)


def run_notebook(src: Path, out: Path, overrides: dict, log: Log, timeout_h: float = 48.0) -> bool:
    import nbformat
    from nbclient import NotebookClient
    from nbclient.exceptions import CellExecutionError
    nb = nbformat.read(str(src), as_version=4)
    # the kernel is Jupyter's "python3": log which interpreter that is (it must have pycolmap + pyceres)
    inject(nb, dict(overrides, _kernel_python="__import__('sys').executable; print('kernel python', _kernel_python)"))
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    state = {"t": time.time()}

    def on_cell_executed(cell, cell_index, execute_reply):          # nbclient >= 0.6
        if cell.cell_type != "code":
            return
        first = next((ln for ln in cell.source.splitlines() if ln.strip() and not ln.startswith("#")), "")[:70]
        text = ""
        for o in cell.get("outputs", []):
            if o.get("output_type") == "stream":
                lines = [ln for ln in o.get("text", "").splitlines() if ln.strip() and not re.match(r"^[IW]\d{8} ", ln)]
                text = (lines[-1] if lines else text)[:160]
            elif o.get("output_type") == "error":
                text = f"ERROR {o.get('ename')}: {o.get('evalue')}"[:300]
        log(f"  {src.stem} cell {cell_index} ({time.time() - state['t']:.0f} s): {first} | {text}")
        state["t"] = time.time()
        try:
            nbformat.write(nb, str(out))                             # progress is visible while it runs
        except OSError:
            pass

    client = NotebookClient(nb, timeout=int(timeout_h * 3600), kernel_name="python3",
                            resources={"metadata": {"path": str(NB)}}, on_cell_executed=on_cell_executed)
    ok = True
    try:
        client.execute()
    except CellExecutionError as e:
        ok = False
        log(f"  {src.stem}: FAILED - {str(e).strip().splitlines()[-1][:300]}")
    except Exception as e:                                           # noqa: BLE001
        ok = False
        log(f"  {src.stem}: FAILED - {type(e).__name__}: {e}")
    nbformat.write(nb, str(out))
    log(f"  {src.stem}: {'done' if ok else 'stopped'} in {(time.time() - t0) / 3600:.2f} h -> {out}")
    return ok


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
    sys.path.insert(0, str(ROOT / "src"))
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
    today = datetime.date.today().isoformat()
    if "04" in a.only:
        log(f"notebook 04 (camera models) on {list(labels)}")
        run_notebook(notebook("04_camera_models"), root / "camera_analysis" / today / "04_camera_models_executed.ipynb",
                     {"SCAPES": "{" + ", ".join(f"{k!r}: Path(r{v!r})" for k, v in labels.items()) + "}",
                      "SCAPES_ROOT": f"Path(r{str(root)!r})",
                      "OUT": f"Path(r{str(root / 'camera_analysis' / today)!r})"}, log)
    if "05" in a.only:
        log(f"notebook 05 (error analysis) on {list(labels)}")
        run_notebook(notebook("05_error_analysis"), root / "error_analysis" / today / "05_error_analysis_executed.ipynb",
                     {"ALIGNMENTS": repr({k: v for k, v in labels.items()}), "MIN_TRACK_LENGTH": "3",
                      "OUT": f"Path(r{str(root / 'error_analysis' / today)!r})"}, log)
    log("batch finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())

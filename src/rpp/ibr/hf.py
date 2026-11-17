"""
rpp.ibr.hf - the panoramas on Hugging Face (0.99.3).

Christian, 16 Nov 2026: "v0p99 should upload rpv products to hugging face near to where the mask CNN are", and
"Prepare ibr run to reprocess everything and upload to hugging face".

The pictures go to a public dataset next to the mask models (``ctate7163/rpp-rpv``; ``RPP_HF_DATASET``), one
commit per station; the viewer's page, ``vendor/`` and ``panos.json`` stay on GitHub Pages (``ctate7163/RPV``),
whose ``panos.json`` points at the dataset with addresses pinned to the station's commit
(``https://huggingface.co/datasets/<repo>/resolve/<commit>/<station>/viewer/<file>``), so a link never changes
under a browser's cache.  Layout of the dataset:

    README.md                          the dataset card (credits, licence ``RPP_HF_LICENSE``, layout)
    <station>/viewer/<file>            what the page loads: the 8K stereo JPEGs of every view and variant, the nadir
                                       mark's picture, the rover GLB (the same files web.publish used to put into
                                       panos/ of the GitHub checkout)
    <station>/record/[with_rover/]     the data record: the 16-bit stacked panoramas (_360_3DTD_16.png), the pivot's
                                       16-bit depth (_ods_centre_depth_16.png), ods.json, station.json, provenance.png,
                                       tone_levels.json (the left and right 16-bit eyes are the same pixels as the
                                       stacked file and are left out)

A station whose upload fails keeps the entry it had in ``panos.json`` (or none), so the page never points at a file
that is not there; its pictures stay in the stage folder and go up with the next run.  The token is the one
``huggingface-cli login`` stored on the PC (or ``HF_TOKEN``), the same one the mask release uses.

    rpp single-sites --upload                  (0.99.3: --host hf is the default) run, then upload station by station
    rpp single-sites --publish-only --upload   upload what the output root holds
    rpp hf-check                               the dataset is reachable and serves the page's origin (CORS)
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import env as rpp_env

log = logging.getLogger("rpp.ibr.hf")

DATASET = "ctate7163/rpp-rpv"          # 0.99.12 (Christian, 17 Nov 2026: "there should be a separate rpp-rpv repository")
OLD_DATASETS = ("ctate7163/rpv-panoramas",)    # 0.99.3-0.99.11: moved to DATASET by ensure_dataset (commits kept)
LICENSE = "cc-by-4.0"              # the proposal of 16 Nov (the code is Apache-2.0); RPP_HF_LICENSE to choose another
RESOLVE = "https://huggingface.co/datasets/{repo}/resolve/{rev}/{path}"
PAGE_ORIGIN = "https://ctate7163.github.io"
STAGE_NAME = "_hf_stage"           # <output root>/_hf_stage/panos: the pictures web.publish writes for the upload
RECORD_GLOBS = ("*_360_3DTD_16.png", "*_ods_centre_depth_16.png", "*ods.json", "station.json", "*provenance*.png")
RECORD_TOP = ("tone_levels.json",)
SUBS = ("", "with_rover")


def dataset() -> str:
    return rpp_env.get("RPP_HF_DATASET") or DATASET


def licence() -> str:
    return rpp_env.get("RPP_HF_LICENSE") or LICENSE


def stage_dir(out_root) -> Path:
    return Path(out_root) / STAGE_NAME / "panos"


def resolve_url(repo: str, rev: str, path: str) -> str:
    return RESOLVE.format(repo=repo, rev=rev, path=path)


def refs(obj: Any) -> List[str]:
    """Every ``panos/<file>`` string in a ``panos.json`` entry (its views, ground mark, rover surface), in order."""
    out: List[str] = []
    if isinstance(obj, dict):
        for v in obj.values():
            out += refs(v)
    elif isinstance(obj, list):
        for v in obj:
            out += refs(v)
    elif isinstance(obj, str) and obj.startswith("panos/") and "/" not in obj[6:]:
        out.append(obj)
    return list(dict.fromkeys(out))


def rewrite(obj: Any, mapping: Dict[str, str]) -> Any:
    """The entry with its ``panos/<file>`` strings replaced by ``mapping`` (the dataset's addresses)."""
    if isinstance(obj, dict):
        return {k: rewrite(v, mapping) for k, v in obj.items()}
    if isinstance(obj, list):
        return [rewrite(v, mapping) for v in obj]
    if isinstance(obj, str):
        return mapping.get(obj, obj)
    return obj


def record_files(station_out: Path) -> List[Tuple[Path, str]]:
    """(file, path under ``record/``) of a station's data record (both variants)."""
    out: List[Tuple[Path, str]] = []
    for sub in SUBS:
        d = Path(station_out) / sub
        if not d.is_dir():
            continue
        seen = set()
        for g in RECORD_GLOBS:
            for f in sorted(d.glob(g)):
                if f.is_file() and f.name not in seen:
                    seen.add(f.name)
                    out.append((f, f"{sub + '/' if sub else ''}{f.name}"))
    for name in RECORD_TOP:
        f = Path(station_out) / name
        if f.is_file():
            out.append((f, name))
    return out


def card(repo: str) -> str:
    """The dataset card (README.md)."""
    from .. import __version__
    return f"""---
license: {licence()}
pretty_name: Rover Photogrammetry Viewer panoramas
tags:
- mars
- perseverance
- stereo
- panorama
- omnidirectional-stereo
- photogrammetry
---

# Rover Photogrammetry Viewer (RPV) panoramas

Omnidirectional stereo (ODS) panoramas of Mars 2020 Perseverance rover stations, made with the Rover Photogrammetry
Pipeline (RPP, https://github.com/ctate7163/RPP) by Christian Tate (Malin Space Science Systems) for research purposes.
They are shown by the Rover Photogrammetry Viewer: {PAGE_ORIGIN}/RPV/

**Image credit:** NASA/JPL-Caltech/MSSS (Navcam); NASA/JPL-Caltech/MSSS/ASU (Mastcam-Z).

## Layout

- `<station>/viewer/`: what the viewer loads - the stereo JPEGs (`*_360_3DTD.jpg`, the left eye on top, 8192 px),
  the enhanced, normal-map and distance-map views, the nadir mark and the rover model (`*.glb`).
- `<station>/record/` (and `record/with_rover/`): the 16-bit stacked panoramas (`*_360_3DTD_16.png`, linear),
  the 16-bit depth from the pivot (`*_ods_centre_depth_16.png`), `ods.json` and `station.json` (how each was made),
  the provenance picture and the tone levels.

The station label is `Sol<sol>_S<site>D<drive>`; the `*_rover_*` files show the rover, the others the terrain only.
Uploaded by RPP {__version__}.
"""


def _api(api=None):
    if api is not None:
        return api
    try:
        from huggingface_hub import HfApi
    except ImportError as e:                                     # pragma: no cover - on the PC it is installed
        raise RuntimeError("huggingface_hub is not installed (pip install huggingface_hub)") from e
    return HfApi()


def _exists(api, repo: str) -> bool:
    try:
        return bool(api.repo_exists(repo, repo_type="dataset"))
    except Exception:                                            # noqa: BLE001 - unknown: create_repo decides
        return False


def move_old_dataset(repo: Optional[str] = None, api=None) -> Optional[str]:
    """0.99.12: when ``repo`` (the default ``ctate7163/rpp-rpv``) does not exist and a dataset of an earlier name does
    (``OLD_DATASETS``: ``ctate7163/rpv-panoramas``, 0.99.3-0.99.11), that one is renamed to ``repo``, so what it holds
    and its commits stay (Hugging Face forwards the old addresses).  Returns the old name moved, else None."""
    repo = repo or dataset()
    if repo != DATASET:                                          # another dataset named by RPP_HF_DATASET: as it is
        return None
    api = _api(api)
    if _exists(api, repo):
        return None
    for old in OLD_DATASETS:
        if old != repo and _exists(api, old):
            api.move_repo(from_id=old, to_id=repo, repo_type="dataset")
            log.info("hf: the dataset %s renamed %s", old, repo)
            return old
    return None


def old_to_new(obj: Any, repo: str) -> Any:
    """0.99.12: ``panos.json`` addresses into an earlier dataset (``OLD_DATASETS``) pointed at ``repo`` - the same
    commit and path (a renamed dataset keeps its commits)."""
    if isinstance(obj, dict):
        return {k: old_to_new(v, repo) for k, v in obj.items()}
    if isinstance(obj, list):
        return [old_to_new(v, repo) for v in obj]
    if isinstance(obj, str):
        for old in OLD_DATASETS:
            if old != repo:
                obj = obj.replace(f"/datasets/{old}/", f"/datasets/{repo}/")
        return obj
    return obj


def ensure_dataset(repo: Optional[str] = None, api=None) -> str:
    """The dataset exists (public) and has its card - an earlier dataset renamed to it if there is one
    (:func:`move_old_dataset`).  Returns the repo id."""
    repo = repo or dataset()
    api = _api(api)
    moved = move_old_dataset(repo, api)
    if moved:
        print(f"the Hugging Face dataset {moved} is now {repo} (its files and commits kept)")
    api.create_repo(repo, repo_type="dataset", exist_ok=True, private=False)
    try:
        has = api.file_exists(repo, "README.md", repo_type="dataset")
    except Exception:                                            # noqa: BLE001 - an older hub: upload it once more
        has = False
    if not has:
        api.upload_file(path_or_fileobj=card(repo).encode("utf-8"), path_in_repo="README.md", repo_id=repo,
                        repo_type="dataset", commit_message="The dataset card")
    return repo


def upload_station(label: str, entry: Dict[str, Any], pano_dir: Path, station_out: Optional[Path] = None,
                   repo: Optional[str] = None, api=None, record: bool = True) -> Dict[str, Any]:
    """One station in one commit: its viewer files (the entry's ``panos/`` files, from ``pano_dir``) and its record.
    Returns {"entry": the entry with the dataset's addresses, "commit", "files", "bytes"}."""
    from huggingface_hub import CommitOperationAdd
    repo = repo or dataset()
    api = _api(api)
    ops, n_bytes, mapping = [], 0, {}
    for r in refs(entry):
        f = Path(pano_dir) / r[6:]
        if not f.is_file():
            raise FileNotFoundError(f"{label}: {f} (named in its entry) is not in the stage")
        path = f"{label}/viewer/{f.name}"
        ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=str(f)))
        mapping[r] = path
        n_bytes += f.stat().st_size
    if record and station_out is not None:
        for f, rel in record_files(Path(station_out)):
            ops.append(CommitOperationAdd(path_in_repo=f"{label}/record/{rel}", path_or_fileobj=str(f)))
            n_bytes += f.stat().st_size
    if not ops:
        raise ValueError(f"{label}: nothing to upload")
    from .. import __version__
    info = api.create_commit(repo_id=repo, repo_type="dataset", operations=ops,
                             commit_message=f"{label}: panoramas (RPP {__version__})")
    rev = getattr(info, "oid", None) or getattr(info, "commit_id", None) or "main"
    urls = {r: resolve_url(repo, rev, path) for r, path in mapping.items()}
    return {"entry": rewrite(entry, urls), "commit": rev, "files": len(ops), "bytes": n_bytes}


def publish(web_dir, doc: Dict[str, Any], labels: Sequence[str], out_root, previous: Optional[Dict[str, Any]] = None,
            repo: Optional[str] = None, api=None, record: bool = True) -> Dict[str, Any]:
    """After ``web.publish(..., files_dir=stage_dir(out_root))``: every station of ``labels`` in ``doc`` uploaded
    (its own commit), its entry pointed at the dataset; a station that fails gets back its entry of ``previous``
    (the ``panos.json`` before the run) or leaves the list.  ``panos.json`` in ``web_dir`` is written again, and the
    pictures of these stations that an earlier GitHub upload left in ``web_dir/panos`` are removed (the dataset
    holds them; the next git upload records the removal).  Returns {"ok", "uploaded", "failed", "message"}."""
    web = Path(web_dir)
    pano_dir = stage_dir(out_root)
    repo = repo or dataset()
    prev = {p["id"]: p for p in (previous or {}).get("panos", [])}
    done, failed = [], {}
    try:
        api = _api(api)
        ensure_dataset(repo, api)
    except Exception as e:                                       # noqa: BLE001
        failed = {k: f"the dataset {repo}: {e}" for k in labels}
        api = None
    panos = []
    doc = old_to_new(doc, repo)                                  # 0.99.12: entries made for ctate7163/rpv-panoramas
    prev = {k: old_to_new(v, repo) for k, v in prev.items()}
    for p in doc.get("panos", []):
        lab = p.get("id")
        if lab not in labels or not refs(p):                    # another run's station, or already on the dataset
            panos.append(p)
            continue
        if api is None:                                          # the dataset is not usable: the entry of before
            if lab in prev:
                panos.append(prev[lab])
            continue
        t0 = time.time()
        try:
            res = upload_station(lab, p, pano_dir, Path(out_root) / lab, repo, api, record=record)
        except Exception as e:                                   # noqa: BLE001 - the next station still goes
            failed[lab] = str(e)
            log.warning("hf: %s not uploaded (%s) - its earlier entry stays", lab, e)
            if lab in prev:
                panos.append(prev[lab])
            continue
        entry = dict(res["entry"], hosted="huggingface", hf_commit=res["commit"])
        panos.append(entry)
        done.append(lab)
        log.info("hf: %s - %d files, %.0f MB in %.0f s (commit %s)", lab, res["files"], res["bytes"] / 1e6,
                 time.time() - t0, str(res["commit"])[:10])
    out = dict(doc, panos=panos)
    (web / "panos.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    removed = prune_local(web, out)
    msg = (f"Hugging Face {repo}: {len(done)} station(s) uploaded" + (f", {len(failed)} not: " + "; ".join(
        f"{k} ({v})" for k, v in failed.items()) if failed else "") + (f"; {removed} old picture(s) removed from "
                                                                       f"{web / 'panos'}" if removed else ""))
    return {"ok": not failed, "uploaded": done, "failed": failed, "message": msg}


def prune_local(web_dir, doc: Dict[str, Any]) -> int:
    """The pictures in ``web_dir/panos`` that ``doc`` no longer names (stations now on the dataset) removed - the
    page's own pictures (the welcome view, the maps) are named and stay."""
    web = Path(web_dir)
    keep = set(r[6:] for r in refs(doc))
    for k in ("ortho", "dtm"):
        f = ((doc.get("traverse") or {}).get(k) or {}).get("file") if isinstance((doc.get("traverse") or {}).get(k), dict) else None
        if f and str(f).startswith("panos/"):
            keep.add(str(f)[6:])
    n = 0
    for f in sorted((web / "panos").glob("*")):
        if f.is_file() and f.name not in keep and not f.name.startswith(("welcome", "traverse")):
            f.unlink()
            n += 1
    return n


def check(repo: Optional[str] = None, origin: str = PAGE_ORIGIN, timeout: float = 30.0) -> Dict[str, Any]:
    """The dataset's card through its resolve address, asked as the page asks (``Origin``): {"ok", "status",
    "allow_origin", "url", "message"}.  The page fetches the pictures, so the last response must allow the origin."""
    import urllib.request
    repo = repo or dataset()
    url = resolve_url(repo, "main", "README.md")
    req = urllib.request.Request(url, headers={"Origin": origin, "User-Agent": "rpp-hf-check"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            allow = r.headers.get("Access-Control-Allow-Origin")
            status = r.status
    except Exception as e:                                       # noqa: BLE001
        return {"ok": False, "status": None, "allow_origin": None, "url": url, "message": f"{url}: {e}"}
    ok = status == 200 and allow in ("*", origin)
    return {"ok": ok, "status": status, "allow_origin": allow, "url": url,
            "message": f"{url}: HTTP {status}, Access-Control-Allow-Origin {allow!r} - "
                       + ("the page can load the dataset's pictures" if ok else "the page could NOT load them")}


def add_parser(sub) -> None:
    p = sub.add_parser("hf-check", help="0.99.3: the panoramas' Hugging Face dataset is reachable and serves the "
                       "viewer's origin (rpp.ibr.hf)")
    p.add_argument("--repo", default=None, help=f"the dataset (default RPP_HF_DATASET or {DATASET})")
    p.add_argument("--create", action="store_true", help="create the dataset and its card when missing (needs the token)")


def cmd_hf_check(a) -> int:
    repo = a.repo or dataset()
    if getattr(a, "create", False):
        print(f"dataset: {ensure_dataset(repo)} (card licence {licence()})")
    r = check(repo)
    print(r["message"])
    return 0 if r["ok"] else 1

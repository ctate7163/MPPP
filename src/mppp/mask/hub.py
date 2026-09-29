"""
Released mask models (v0p13): safetensors export, the model registry, download
and local installation.

The released model is one ``.safetensors`` file with the model card embedded
in its metadata (no pickle, no code execution on load).  ``mppp/data/models.json``
lists each released model with its SHA-256 and download URLs (Hugging Face,
GitHub release).  ``fetch_model()`` downloads it once into the user cache
(``mppp.paths.cache_dir()/models``) and verifies the checksum;
``install_model()`` puts a local ``.pt`` / ``.safetensors`` there instead
(offline machines, or before the model is uploaded).

``resolve_checkpoint(spec)`` is what processing uses for
``config["masking"]["checkpoint"]``:

1. an existing file path -> that file;
2. a registry name (default ``"mppp_mask_v3"``) -> the released file from
   Hugging Face (``ctate7163/mppp-mask``), downloaded once into the cache;
3. a bare file name -> ``checkpoints_dir()/<name>`` (models you train).

A registry name always means the released file (0.31.2): the cached copy is
checked against the registry SHA-256 and downloaded again if it differs, and
a local checkpoint no longer stands in when the download fails unless
``MPPP_MASK_LOCAL_FALLBACK=1`` is set.  ``HF_TOKEN`` (or a ``huggingface-cli
login``) is sent to huggingface.co, for a private model repository.

Release (docs/RELEASING.md)::

    python -m mppp.mask.hub export checkpoints/convnext_tiny_s4_seg_20260925b.pt --name mppp_mask_v3 --update-registry
    python -m mppp.mask.hub upload --name mppp_mask_v3
    python -m mppp.mask.hub verify --name mppp_mask_v3
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import urllib.parse
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from ..paths import cache_dir, checkpoints_dir, data_dir

PathLike = Union[str, Path]
REGISTRY_FILE = "models.json"
FORMAT = "mppp-mask-safetensors-1"
HF_REPO = "ctate7163/mppp-mask"                 # default Hugging Face model repository (0.31.2)
LOCAL_FALLBACK_ENV = "MPPP_MASK_LOCAL_FALLBACK"
_SHA_MEMO: Dict[Tuple[str, int, int], str] = {}


# ------------------------------------------------------------------ registry
def registry_path() -> Path:
    return data_dir() / REGISTRY_FILE


def load_registry() -> Dict[str, Any]:
    return json.loads(registry_path().read_text(encoding="utf-8"))


def default_model_name() -> str:
    return load_registry()["default"]


def model_path(name: str) -> Path:
    """Where the registry model ``name`` lives in the user cache (may not exist yet)."""
    entry = load_registry()["models"][name]
    return cache_dir() / "models" / entry["file"]


def sha256_file(path: PathLike, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def _cached_sha256(path: Path) -> str:
    """SHA-256 of ``path``, remembered per (path, size, mtime) for this process."""
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    if key not in _SHA_MEMO:
        _SHA_MEMO[key] = sha256_file(path)
    return _SHA_MEMO[key]


def hf_repo(name: Optional[str] = None) -> str:
    """Hugging Face repository of registry model ``name`` (``hf_repo`` in the registry, else ``HF_REPO``)."""
    reg = load_registry()
    return reg["models"][name or reg["default"]].get("hf_repo") or HF_REPO


def _local_fallback_enabled() -> bool:
    return os.environ.get(LOCAL_FALLBACK_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# -------------------------------------------------------------------- export
def export_safetensors(checkpoint: PathLike, out: Optional[PathLike] = None, name: Optional[str] = None,
                       update_registry: bool = False, allow_default_card: bool = False) -> Dict[str, Any]:
    """
    ``.pt`` (+ its ``.json`` card) -> one ``.safetensors`` file with the card in
    its metadata.  ``name``: registry name; the output file name then comes
    from the registry (if the name is registered) or is ``<name>.safetensors``.
    ``update_registry``: write the new SHA-256 and size into
    ``mppp/data/models.json`` (in a source checkout; commit it).
    Deterministic: the same checkpoint (and card) always gives the same bytes,
    whatever the MPPP version.
    A ``.pt`` without its ``.json`` card is refused (0.31.2): the default card
    describes a different architecture (no stride-4 decoder), so the export
    would not load.  ``allow_default_card=True`` exports it anyway.
    Returns ``{"path", "sha256", "bytes"}``.
    """
    from safetensors.torch import save_file
    from .model import SAFETENSORS_CARD_KEY, read_card, read_state_dict_checkpoint
    src = Path(checkpoint)
    if not src.is_file():                                   # v0p22.4: a bare name, or a path relative to the
        for cand in (checkpoints_dir() / src.name, checkpoints_dir() / src):   # repo run from notebooks/
            if cand.is_file():
                src = cand
                break
        else:
            raise FileNotFoundError(f"checkpoint not found: {checkpoint} (cwd {Path.cwd()}; also looked in "
                                    f"{checkpoints_dir()})")
    from .model import card_path
    if src.suffix != ".safetensors" and not card_path(src).is_file() and not allow_default_card:
        raise FileNotFoundError(f"no model card {card_path(src).name} next to {src}: copy it there (training writes "
                                f"it beside the .pt); without it the export gets the default card and will not load")
    card = read_card(src)
    sd = read_state_dict_checkpoint(src, card)
    sd = {k: v.detach().cpu().contiguous() for k, v in sd.items()}
    reg = load_registry()
    if out is None:
        fname = reg["models"][name]["file"] if name and name in reg["models"] else f"{name or src.stem}.safetensors"
        out = src.parent / fname
    out = Path(out)
    # no MPPP version in the card (v0p14): the file - and its SHA-256 - depend only on the checkpoint
    card = dict(card, exported_from=src.name)
    if name:
        card["release_name"] = name
    meta = {SAFETENSORS_CARD_KEY: json.dumps(card, sort_keys=True), "format": FORMAT}
    out.parent.mkdir(parents=True, exist_ok=True)
    save_file(sd, str(out), metadata=meta)
    _canonical_header(out)
    info = {"path": str(out), "sha256": sha256_file(out), "bytes": out.stat().st_size}
    if update_registry:
        if not name:
            raise ValueError("update_registry needs name=")
        entry = reg["models"].setdefault(name, {"file": out.name, "urls": []})
        entry.update(file=out.name, sha256=info["sha256"], bytes=info["bytes"], source_checkpoint=src.name,
                     val_iou=card.get("val_iou"), backbone=card.get("backbone"), stride4=card.get("stride4"))
        registry_path().write_text(json.dumps(reg, indent=2) + "\n", encoding="utf-8")
        info["registry"] = str(registry_path())
    return info


def _canonical_header(path: Path) -> None:
    """
    Rewrite the safetensors JSON header with sorted keys.  The writer's header
    key order is not stable between calls (the tensor bytes are), so without
    this the same model could get a different SHA-256 on every export.
    """
    import struct
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n].decode("utf-8"))
    h = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    h += b" " * (-len(h) % 8)
    path.write_bytes(struct.pack("<Q", len(h)) + h + raw[8 + n:])


# ------------------------------------------------------------------ download
def _hf_token() -> Optional[str]:
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        return tok
    try:                                                    # a `huggingface-cli login` token, if the package is there
        from huggingface_hub import get_token
        return get_token()
    except Exception:                                       # noqa: BLE001
        return None


def _headers(url: str) -> Dict[str, str]:
    h = {"User-Agent": "mppp"}
    host = urllib.parse.urlparse(url).hostname or ""
    if host == "huggingface.co" or host.endswith(".huggingface.co"):
        tok = _hf_token()
        if tok:
            h["Authorization"] = f"Bearer {tok}"
    return h


def _download(url: str, dst: Path, timeout: float = 60.0) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers=_headers(url))
    with urllib.request.urlopen(req, timeout=timeout) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length") or 0)
        done, step = 0, 0
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
            done += len(b)
            if total and done * 10 // total > step:
                step = done * 10 // total
                print(f"  {done / 2**20:.0f} / {total / 2**20:.0f} MB", flush=True)
    tmp.replace(dst)


def local_candidates(name: str) -> List[Path]:
    """
    Local files that can stand in for registry model ``name`` when it cannot
    be downloaded (v0p14.1), in order: the released ``.safetensors`` in
    ``checkpoints_dir()``, the checkpoint it was exported from
    (``source_checkpoint``, e.g. ``convnext_tiny_s4_seg_best.pt``), and the
    promoted best of the same architecture.
    """
    entry = load_registry()["models"][name]
    d = checkpoints_dir()
    names = [entry["file"], entry.get("source_checkpoint")]
    if entry.get("backbone"):
        names.append(f"{entry['backbone']}{'_s4' if entry.get('stride4') else ''}_seg_best.pt")
    out: List[Path] = []
    for n in names:
        if n and (d / n).is_file() and (d / n) not in out:
            out.append(d / n)
    return out


def fetch_model(name: Optional[str] = None, force: bool = False, urls: Optional[List[str]] = None,
                local_fallback: Optional[bool] = None) -> Path:
    """
    Path of registry model ``name`` in the user cache: the released file,
    downloaded if it is not cached (URLs from the registry, in order: Hugging
    Face first) and checked against the registry SHA-256.  A cached file whose
    SHA-256 differs from the registry (an older export, or a local stand-in)
    is set aside as ``*.sha256-mismatch`` and downloaded again.
    ``local_fallback``: if every download fails, install the first of
    :func:`local_candidates` instead.  Default: off, unless the environment
    variable ``MPPP_MASK_LOCAL_FALLBACK=1`` is set (0.31.2; before, on).
    """
    reg = load_registry()
    name = name or reg["default"]
    if name not in reg["models"]:
        raise KeyError(f"unknown model {name!r}; registered: {list(reg['models'])}")
    if local_fallback is None:
        local_fallback = _local_fallback_enabled()
    entry = reg["models"][name]
    dst = model_path(name)
    want = entry.get("sha256")
    if dst.is_file() and not force:
        if not want or _cached_sha256(dst) == want:
            return dst
        bad = dst.with_suffix(".sha256-mismatch")
        print(f"[mppp] cached {dst.name} is not the released {name} (SHA-256 {_cached_sha256(dst)[:12]}…, "
              f"registry {want[:12]}…); downloading the released file", flush=True)
        dst.replace(bad)
    dst.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in (urls or entry.get("urls") or []):
        try:
            print(f"[mppp] downloading mask model {name} from {url}", flush=True)
            _download(url, dst)
        except Exception as e:                                  # noqa: BLE001
            errors.append(f"{url}: {type(e).__name__}: {e}")
            continue
        got = sha256_file(dst)
        if want and got != want:
            bad = dst.with_suffix(".sha256-mismatch")
            dst.replace(bad)
            errors.append(f"{url}: SHA-256 {got} != registry {want} (kept as {bad.name})")
            continue
        return dst
    if local_fallback:
        for cand in local_candidates(name):
            print(f"[mppp] could not download mask model {name}; installing the local {cand} instead", flush=True)
            try:
                return install_model(cand, name)
            except Exception as e:                              # noqa: BLE001
                errors.append(f"local {cand}: {type(e).__name__}: {e}")
        tail = (f"\nNo local fallback in {checkpoints_dir()} (looked for {entry['file']}, "
                f"{entry.get('source_checkpoint')}).")
    else:
        tail = (f"\nThe released model comes from {', '.join(entry.get('urls') or ['(no URL)'])}; a local checkpoint "
                f"stands in only with {LOCAL_FALLBACK_ENV}=1 (or fetch_model(..., local_fallback=True)).")
    raise FileNotFoundError(
        f"mask model {name!r} is not in the cache ({dst}) and could not be downloaded:\n  "
        + "\n  ".join(errors or ["no URLs in the registry"])
        + tail
        + f"  Offline: python -m mppp.mask.hub install <{entry['file']}> --name {name}, "
          f"or set config['masking']['checkpoint'] to a checkpoint path.")


def install_model(src: PathLike, name: Optional[str] = None, check_sha: bool = False) -> Path:
    """
    Put a local model into the cache under registry name ``name``: a
    ``.safetensors`` is copied, a ``.pt`` is exported to safetensors.  The
    SHA-256 is compared with the registry and a difference is reported (a
    different, e.g. newer, model); ``check_sha=True`` makes it an error.
    """
    reg = load_registry()
    name = name or reg["default"]
    dst = model_path(name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    if src.suffix == ".safetensors":
        shutil.copy2(src, dst)
    else:
        export_safetensors(src, out=dst, name=name)
    got, want = sha256_file(dst), reg["models"][name].get("sha256")
    if want and got != want:
        msg = f"installed {src.name} as {name}, but its SHA-256 {got[:12]}… differs from the registry {want[:12]}…"
        if check_sha:
            dst.unlink()
            raise ValueError(msg)
        warnings.warn(msg + " (a different model than the released one; resolve_checkpoint/fetch_model replace it "
                            "with the released file when they can download it - use the file path in the config "
                            "to run this model)")
    return dst


# ------------------------------------------------------------------- resolve
def resolve_checkpoint(spec: Optional[PathLike] = None, download: bool = True) -> Path:
    """``config["masking"]["checkpoint"]`` -> a file (see module docstring)."""
    reg = load_registry()
    spec = spec if spec not in (None, "") else reg["default"]
    p = Path(spec)
    if p.is_file():
        return p
    if str(spec) in reg["models"]:
        if not download:
            return model_path(str(spec))
        return fetch_model(str(spec))                    # cached and SHA-checked, else downloaded
    if not p.is_absolute():
        cand = checkpoints_dir() / p
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Mask checkpoint not found: {spec}\n"
        f"  not a file, not a registered model ({list(reg['models'])}), and not in {checkpoints_dir()}\n"
        f"  checkpoints there: {sorted(q.name for q in checkpoints_dir().glob('*.pt')) if checkpoints_dir().is_dir() else 'none'}")


# ------------------------------------------------------------------- publish
def _default_card_file() -> Optional[Path]:
    p = Path(__file__).resolve().parents[3] / "docs" / "hf_model_card.md"     # src/mppp/mask/hub.py -> repo root
    return p if p.is_file() else None


def upload_model(name: Optional[str] = None, repo_id: Optional[str] = None, card: Optional[PathLike] = None,
                 private: bool = False, file: Optional[PathLike] = None) -> str:
    """
    Upload registry model ``name`` to its Hugging Face repository (0.31.2):
    the ``.safetensors`` (``file``, else ``checkpoints_dir()/<file>``, exported
    from ``source_checkpoint`` if it is not there yet) and the model card
    (``card``, default ``docs/hf_model_card.md``) as ``README.md``.  Refuses a
    file whose SHA-256 is not the registry's, so the URL always serves what
    the registry describes.  Needs ``huggingface_hub`` and a write token
    (``huggingface-cli login`` or ``HF_TOKEN``).  Returns the download URL.
    """
    from huggingface_hub import HfApi
    reg = load_registry()
    name = name or reg["default"]
    entry = reg["models"][name]
    repo_id = repo_id or entry.get("hf_repo") or HF_REPO
    path = Path(file) if file else checkpoints_dir() / entry["file"]
    if not path.is_file():
        src = checkpoints_dir() / (entry.get("source_checkpoint") or "")
        if not src.is_file():
            raise FileNotFoundError(f"neither {path} nor its source checkpoint {src} exists")
        print(f"[mppp] exporting {src.name} -> {path.name}", flush=True)
        export_safetensors(src, out=path, name=name)
    got, want = sha256_file(path), entry.get("sha256")
    if want and got != want:
        raise ValueError(f"{path} has SHA-256 {got}, the registry {want}: not uploading.  Export it again "
                         f"(its .json card must be next to the .pt), or update the registry with "
                         f"export --update-registry and commit models.json.")
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"[mppp] uploading {path.name} ({path.stat().st_size / 2**20:.0f} MB) to {repo_id}", flush=True)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=entry["file"], repo_id=repo_id, repo_type="model",
                    commit_message=f"{name}: {entry['file']} (SHA-256 {got[:12]})")
    card = Path(card) if card else _default_card_file()
    if card is not None:
        api.upload_file(path_or_fileobj=str(card), path_in_repo="README.md", repo_id=repo_id, repo_type="model",
                        commit_message=f"model card ({name})")
    url = f"https://huggingface.co/{repo_id}/resolve/main/{entry['file']}"
    if url not in (entry.get("urls") or []):
        warnings.warn(f"{url} is not in the registry URLs of {name}; add it to src/mppp/data/models.json")
    return url


def verify_model(name: Optional[str] = None) -> List[Dict[str, Any]]:
    """Download registry model ``name`` from each of its URLs into a temporary folder and check the SHA-256."""
    reg = load_registry()
    name = name or reg["default"]
    entry = reg["models"][name]
    out = []
    with tempfile.TemporaryDirectory() as td:
        for url in entry.get("urls") or []:
            dst = Path(td) / entry["file"]
            try:
                _download(url, dst)
                got = sha256_file(dst)
                out.append({"url": url, "ok": got == entry.get("sha256"), "sha256": got, "bytes": dst.stat().st_size})
            except Exception as e:                              # noqa: BLE001
                out.append({"url": url, "ok": False, "error": f"{type(e).__name__}: {e}"})
            if dst.exists():
                dst.unlink()
    return out


# ----------------------------------------------------------------------- CLI
def _main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m mppp.mask.hub", description="MPPP mask model tools")
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export", help=".pt -> .safetensors with the card embedded")
    e.add_argument("checkpoint")
    e.add_argument("--out")
    e.add_argument("--name", help="registry name, e.g. mppp_mask_v3")
    e.add_argument("--update-registry", action="store_true")
    i = sub.add_parser("install", help="put a local .pt/.safetensors into the cache as a registry model")
    i.add_argument("src")
    i.add_argument("--name")
    f = sub.add_parser("fetch", help="download a registry model into the cache")
    f.add_argument("--name")
    f.add_argument("--force", action="store_true")
    u = sub.add_parser("upload", help="upload a registry model and its card to Hugging Face")
    u.add_argument("--name")
    u.add_argument("--repo", help=f"Hugging Face model repository (default: registry hf_repo, else {HF_REPO})")
    u.add_argument("--card", help="model card (default docs/hf_model_card.md), uploaded as README.md")
    u.add_argument("--file", help="the .safetensors (default checkpoints/<registry file>)")
    u.add_argument("--private", action="store_true")
    v = sub.add_parser("verify", help="download a registry model from each URL and check its SHA-256")
    v.add_argument("--name")
    sub.add_parser("list", help="registry and cache status")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(json.dumps(export_safetensors(a.checkpoint, a.out, a.name, a.update_registry), indent=1))
    elif a.cmd == "install":
        print(install_model(a.src, a.name))
    elif a.cmd == "fetch":
        print(fetch_model(a.name, a.force))
    elif a.cmd == "upload":
        print(upload_model(a.name, a.repo, a.card, a.private, a.file))
    elif a.cmd == "verify":
        res = verify_model(a.name)
        for r in res:
            print(f"{'OK  ' if r['ok'] else 'FAIL'} {r['url']}  {r.get('sha256', r.get('error', ''))}")
        return 0 if res and all(r["ok"] for r in res) else 1
    else:
        reg = load_registry()
        for n, ent in reg["models"].items():
            p = model_path(n)
            st = "not cached"
            if p.is_file():
                st = "cached" if not ent.get("sha256") or _cached_sha256(p) == ent["sha256"] else \
                    "cached, NOT the released file (replaced on next use)"
            print(f"{n}{' (default)' if n == reg['default'] else ''}: {ent['file']}  sha256 {ent.get('sha256', '?')[:16]}…  "
                  f"{st} ({p})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

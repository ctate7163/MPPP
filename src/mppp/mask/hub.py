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
2. a registry name (default ``"mppp_mask_v3"``) -> the cached file, downloaded if needed;
3. a bare file name -> ``checkpoints_dir()/<name>`` (models you train).

Export for a release (docs/RELEASING.md)::

    python -m mppp.mask.hub export checkpoints/convnext_tiny_s4_seg_20260925b.pt --name mppp_mask_v3 --update-registry
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import urllib.request
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ..paths import cache_dir, checkpoints_dir, data_dir

PathLike = Union[str, Path]
REGISTRY_FILE = "models.json"
FORMAT = "mppp-mask-safetensors-1"


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


# -------------------------------------------------------------------- export
def export_safetensors(checkpoint: PathLike, out: Optional[PathLike] = None, name: Optional[str] = None,
                       update_registry: bool = False) -> Dict[str, Any]:
    """
    ``.pt`` (+ its ``.json`` card) -> one ``.safetensors`` file with the card in
    its metadata.  ``name``: registry name; the output file name then comes
    from the registry (if the name is registered) or is ``<name>.safetensors``.
    ``update_registry``: write the new SHA-256 and size into
    ``mppp/data/models.json`` (in a source checkout; commit it).
    Deterministic: the same checkpoint (and card) always gives the same bytes,
    whatever the MPPP version.
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
def _download(url: str, dst: Path, timeout: float = 60.0) -> None:
    tmp = dst.with_suffix(dst.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "mppp"})
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
                local_fallback: bool = True) -> Path:
    """
    Path of registry model ``name`` in the user cache, downloading it first if
    needed (URLs from the registry, in order) and checking its SHA-256.
    ``local_fallback`` (default): if every download fails (offline, or the
    model is not published yet), install the first of :func:`local_candidates`
    into the cache instead, with a warning if it is not the released file.
    """
    reg = load_registry()
    name = name or reg["default"]
    if name not in reg["models"]:
        raise KeyError(f"unknown model {name!r}; registered: {list(reg['models'])}")
    entry = reg["models"][name]
    dst = model_path(name)
    want = entry.get("sha256")
    if dst.is_file() and not force:
        return dst
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
    raise FileNotFoundError(
        f"mask model {name!r} is not in the cache ({dst}) and could not be downloaded:\n  "
        + "\n  ".join(errors or ["no URLs in the registry"])
        + f"\nNo local fallback in {checkpoints_dir()} (looked for {entry['file']}, "
          f"{entry.get('source_checkpoint')}).  Install a local copy: mppp.mask.hub.install_model("
          f"'<your .pt or .safetensors>', name={name!r}), set MPPP_CHECKPOINTS to the folder that holds it, "
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
        warnings.warn(msg + " (a different model than the released one)")
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
        cached = model_path(str(spec))
        if cached.is_file() or not download:
            return cached
        return fetch_model(str(spec))
    if not p.is_absolute():
        cand = checkpoints_dir() / p
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"Mask checkpoint not found: {spec}\n"
        f"  not a file, not a registered model ({list(reg['models'])}), and not in {checkpoints_dir()}\n"
        f"  checkpoints there: {sorted(q.name for q in checkpoints_dir().glob('*.pt')) if checkpoints_dir().is_dir() else 'none'}")


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
    sub.add_parser("list", help="registry and cache status")
    a = ap.parse_args(argv)
    if a.cmd == "export":
        print(json.dumps(export_safetensors(a.checkpoint, a.out, a.name, a.update_registry), indent=1))
    elif a.cmd == "install":
        print(install_model(a.src, a.name))
    elif a.cmd == "fetch":
        print(fetch_model(a.name, a.force))
    else:
        reg = load_registry()
        for n, ent in reg["models"].items():
            p = model_path(n)
            print(f"{n}{' (default)' if n == reg['default'] else ''}: {ent['file']}  sha256 {ent.get('sha256', '?')[:16]}…  "
                  f"{'cached' if p.is_file() else 'not cached'} ({p})")
    return 0


if __name__ == "__main__":
    sys.exit(_main())

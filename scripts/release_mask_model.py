"""
Release tooling for the mask model (maintainers only): upload a registry model and its card to Hugging Face, and
verify every download URL of a model against the registry's SHA-256.  Needs ``huggingface_hub`` (``pip install
mppp[release]``) and a write token (``huggingface-cli login`` or ``HF_TOKEN``).

    python scripts/release_mask_model.py upload [NAME] [--repo REPO] [--card FILE] [--private] [--file FILE]
    python scripts/release_mask_model.py verify [NAME]
"""
from __future__ import annotations

import argparse
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from mppp.mask.hub import HF_REPO, _download, export_safetensors, load_registry, sha256_file
from mppp.paths import checkpoints_dir

PathLike = Union[str, Path]


def _default_card_file() -> Optional[Path]:
    p = Path(__file__).resolve().parents[1] / "docs" / "hf_model_card.md"     # scripts/ -> repo root
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




def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    u = sub.add_parser("upload")
    u.add_argument("name", nargs="?")
    u.add_argument("--repo")
    u.add_argument("--card")
    u.add_argument("--private", action="store_true")
    u.add_argument("--file")
    v = sub.add_parser("verify")
    v.add_argument("name", nargs="?")
    a = ap.parse_args(argv)
    if a.cmd == "upload":
        print(upload_model(a.name, a.repo, a.card, a.private, a.file))
    else:
        for r in verify_model(a.name):
            print(("ok  " if r.get("ok") else "FAIL"), r.get("url"), r.get("sha256", r.get("error", ""))[:16])
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Releasing MPPP: code on GitHub, the mask model as safetensors

This covers three things:

1. the code, pushed to `https://github.com/ctate7163/MPPP`;
2. the released mask model, exported to **safetensors** and attached to a **GitHub release**;
3. the same model on **Hugging Face**.

MPPP downloads the model from the URLs in `src/mppp/data/models.json` and checks the SHA-256 against the value stored there. From 0.31.2 the default model, `mppp_mask_v3`, has one URL, on Hugging Face (`ctate7163/mppp-mask`), and a registry name always means that released file: a cached copy with another SHA-256 is downloaded again, and a local checkpoint stands in only with `MPPP_MASK_LOCAL_FALLBACK=1`. Older models (v1, v2) keep their Hugging Face and GitHub-release URLs. The model is not stored in git, for two reasons: GitHub rejects files over 100 MB, and a large file in history makes every clone slower forever.

Commands are for Windows (Anaconda Prompt or PowerShell). On Linux and macOS the same commands work with `/` paths.

---

## 0. One-time setup

* **Git:** `git --version`. If it is missing, install [Git for Windows](https://git-scm.com/download/win) or run `conda install git`.
* **GitHub CLI** (optional, simplifies releases): `winget install GitHub.cli`, then `gh auth login`.
* **Hugging Face:** `pip install -U huggingface_hub`, then `huggingface-cli login` (or `hf auth login` in newer versions). Use a token with *write* access from https://huggingface.co/settings/tokens.
* On github.com, create an **empty** repository `ctate7163/MPPP`: no README, no license, no .gitignore. The files come from the bundle.

## 1. Push the code (first time, from the bundle)

The release was prepared as a git bundle: one file holding the repository and its history, with the tags `v0.13.0` to `v0.21.1`.

```bat
cd /d D:\code
git clone D:\code\MPPP\release_v0p21\mppp_v0p21.bundle MPPP_git
cd MPPP_git
git remote set-url origin https://github.com/ctate7163/MPPP.git
git push -u origin main
git push origin --tags
```

`git log --oneline --decorate` should show fourteen commits, tagged `v0.21.1`, `v0.21.0`, `v0.20.1`, `v0.20.0`, `v0.15.0`, `v0.14.7`, `v0.14.6`, `v0.14.5`, `v0.14.4`, `v0.14.3`, `v0.14.2`, `v0.14.1`, `v0.14.0` and `v0.13.0`. The first push opens a browser window (or asks for a token) to sign in to GitHub.

Then work in `D:\code\MPPP_git`:
* `pip install -e .[mask,sfm]` in your main environment.
* Your old `D:\code\MPPP` stays as it was, with your checkpoints, runs and older notebooks.
* To keep training into the old checkpoints folder, set `MPPP_CHECKPOINTS=D:\code\MPPP\checkpoints`, e.g. with `setx MPPP_CHECKPOINTS D:\code\MPPP\checkpoints` and a new terminal.

Later releases: commit, `git tag vX.Y.Z`, `git push && git push --tags`.

## 2. Export the mask model to safetensors

A `.pt` checkpoint is a Python pickle, and loading one can run code. A `.safetensors` file holds only tensors, plus text metadata. MPPP puts the model card JSON (backbone, canvas, threshold, training history) into that metadata, so the model is a single self-describing file.

From the repository folder (`D:\code\MPPP_git`), with the checkpoint and its `.json` card side by side:

```bat
python -m mppp.mask.hub export D:\code\MPPP\checkpoints\convnext_tiny_s4_seg_20260925.pt --name mppp_mask_v2 --update-registry
```

The command:
* writes `D:\code\MPPP\checkpoints\mppp_mask_convnext_tiny_s4_v2.safetensors` (124,825,300 bytes);
* prints its SHA-256;
* with `--update-registry`, writes the SHA-256 and size into `src/mppp/data/models.json`.

The export is deterministic: the file depends only on the checkpoint and its card. For the checkpoint prepared with this release (`convnext_tiny_s4_seg_20260925.pt` = the 25 Sep 2026 run, 4 epochs, val IoU 0.9771) the SHA-256 is

```
227e483369e11d6e36ce3517cf9f6d6ac60a9a50c1e301f9e7ae53d0cef569b9
```

and `models.json` already contains it. So `git status` should show **no change**. If it shows `models.json` as modified, the file you exported is a different model. That is fine, but commit the new values: `git commit -am "mask model sha256"` and `git push`.

To check the file in Python:

```python
from mppp.mask.model import load_model
model, card = load_model(r"D:\code\MPPP\checkpoints\mppp_mask_convnext_tiny_s4_v2.safetensors")
print(card["backbone"], card["val_iou"], card["exported_from"])
```

## 3. Attach the model to a GitHub release

The registry URL is `https://github.com/ctate7163/MPPP/releases/download/mask-v2/mppp_mask_convnext_tiny_s4_v2.safetensors`. That means a release with the tag **`mask-v2`**, and the file name unchanged. A model-specific tag keeps the URL stable across code versions.

With the GitHub CLI:

```bat
gh release create mask-v2 D:\code\MPPP\checkpoints\mppp_mask_convnext_tiny_s4_v2.safetensors --repo ctate7163/MPPP --title "Mask model v2" --notes "ConvNeXt-tiny + stride-4 decoder, masks_training_set_v7, 4 epochs, val IoU 0.977. SHA-256 227e483369e11d6e36ce3517cf9f6d6ac60a9a50c1e301f9e7ae53d0cef569b9"
```

Or on the web:
1. Open the repository, then **Releases**, then **Draft a new release**.
2. Choose a tag: type `mask-v2` and create it on `main`.
3. Title it "Mask model v2".
4. Drag the `.safetensors` file into the assets box, then **Publish release**.

GitHub release assets can be up to 2 GB each.

A release for the code itself: `gh release create v0.14.0 --repo ctate7163/MPPP --title "MPPP 0.14.0" --notes-file CHANGELOG.md` (or use the web page).

## 4. Upload the model to Hugging Face

The registry URL of `mppp_mask_v3` is `https://huggingface.co/ctate7163/mppp-mask/resolve/main/mppp_mask_convnext_tiny_s4_v3.safetensors`: a public **model** repository `ctate7163/mppp-mask` with the file at its root. MPPP does both steps (repository and upload) for you:

```bat
cd /d D:\code\MPPP
pip install -U huggingface_hub
hf auth login
python -m mppp.mask.hub upload --name mppp_mask_v3
```

`hf auth login` (older versions: `huggingface-cli login`) asks for a token with **write** access from https://huggingface.co/settings/tokens. `upload`:
* exports `checkpoints\convnext_tiny_s4_seg_20260925b.pt` (with its `.json` card next to it) to `checkpoints\mppp_mask_convnext_tiny_s4_v3.safetensors`, if that file is not there yet;
* refuses to upload unless the file's SHA-256 is the registry's (`46830126…`), so the URL always serves exactly what `models.json` describes;
* creates `ctate7163/mppp-mask` if needed (public; `--private` for a private one, then every user needs `HF_TOKEN` or a login to download);
* uploads the file and `docs/hf_model_card.md` as the repository's `README.md`.

Hugging Face stores large files with Xet automatically; nothing goes into git.

## 5. Check the download path

```bat
python -m mppp.mask.hub verify --name mppp_mask_v3
set MPPP_CACHE=%TEMP%\mppp_cache_test
python -m mppp.mask.hub fetch --name mppp_mask_v3
python -m mppp.mask.hub list
```

`verify` downloads the model from every registry URL into a temporary folder and prints `OK` when the SHA-256 matches. `fetch` with an empty cache is what a new user gets. `list` reports a cached file that is not the released one; it is replaced on next use.

## 6. Releasing a new model later

1. Train and promote it (`notebooks/training/02_train_mask.ipynb`).
2. Export it under a new name so the old URL keeps working:

   ```bat
   python -m mppp.mask.hub export <checkpoints>\convnext_tiny_s4_seg_best.pt --name mppp_mask_v4 --update-registry
   ```

   For a name not yet in `models.json`, the file is `mppp_mask_v4.safetensors`, and the entry is created with an empty URL list.
3. Add the Hugging Face URL (`https://huggingface.co/ctate7163/mppp-mask/resolve/main/<file>`) and `"hf_repo": "ctate7163/mppp-mask"` to the new entry in `src/mppp/data/models.json`, then `python -m mppp.mask.hub upload --name <name>` (step 4) and `verify`.
4. To make it the default, set `"default": "<name>"`.
5. Commit and push, and note the change in `CHANGELOG.md`.

Users who want the old model can set `config["masking"]["checkpoint"] = "mppp_mask_v1"`.

`mppp_mask_v2` (0.14.7) was released this way; for it the registry entry was created by hand with the file name `mppp_mask_convnext_tiny_s4_v2.safetensors`. `mppp_mask_v1` stays in the registry with its own URLs (`mask-v1`); if you already uploaded it, leave it there.

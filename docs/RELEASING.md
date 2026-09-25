# Releasing MPPP: code on GitHub, the mask model as safetensors

This covers three things:

1. the code, pushed to `https://github.com/ctate7163/MPPP`;
2. the released mask model, exported to **safetensors** and attached to a **GitHub release**;
3. the same model on **Hugging Face**.

MPPP downloads the model from the URLs in `src/mppp/data/models.json`: Hugging Face first, then the GitHub release. It checks the SHA-256 against the value stored there. The model is not stored in git, for two reasons: GitHub rejects files over 100 MB, and a large file in history makes every clone slower forever.

Commands are for Windows (Anaconda Prompt or PowerShell). On Linux and macOS the same commands work with `/` paths.

---

## 0. One-time setup

* **Git:** `git --version`. If it is missing, install [Git for Windows](https://git-scm.com/download/win) or run `conda install git`.
* **GitHub CLI** (optional, simplifies releases): `winget install GitHub.cli`, then `gh auth login`.
* **Hugging Face:** `pip install -U huggingface_hub`, then `huggingface-cli login` (or `hf auth login` in newer versions). Use a token with *write* access from https://huggingface.co/settings/tokens.
* On github.com, create an **empty** repository `ctate7163/MPPP`: no README, no license, no .gitignore. The files come from the bundle.

## 1. Push the code (first time, from the bundle)

The release was prepared as a git bundle: one file holding the repository and its history, with the tags `v0.13.0` to `v0.14.2`.

```bat
cd /d D:\code
git clone D:\code\MPPP\release_v0p14\mppp_v0p14.bundle MPPP_git
cd MPPP_git
git remote set-url origin https://github.com/ctate7163/MPPP.git
git push -u origin main
git push origin --tags
```

`git log --oneline --decorate` should show four commits, tagged `v0.14.2`, `v0.14.1`, `v0.14.0` and `v0.13.0`. The first push opens a browser window (or asks for a token) to sign in to GitHub.

Then work in `D:\code\MPPP_git`:
* `pip install -e .[mask,sfm]` in your main environment.
* Your old `D:\code\MPPP` stays as it was, with your checkpoints, runs and older notebooks.
* To keep training into the old checkpoints folder, set `MPPP_CHECKPOINTS=D:\code\MPPP\checkpoints`, e.g. with `setx MPPP_CHECKPOINTS D:\code\MPPP\checkpoints` and a new terminal.

Later releases: commit, `git tag vX.Y.Z`, `git push && git push --tags`.

## 2. Export the mask model to safetensors

A `.pt` checkpoint is a Python pickle, and loading one can run code. A `.safetensors` file holds only tensors, plus text metadata. MPPP puts the model card JSON (backbone, canvas, threshold, training history) into that metadata, so the model is a single self-describing file.

From the repository folder (`D:\code\MPPP_git`), with the best checkpoint and its `.json` card side by side:

```bat
python -m mppp.mask.hub export D:\code\MPPP\checkpoints\convnext_tiny_s4_seg_best.pt --name mppp_mask_v1 --update-registry
```

The command:
* writes `D:\code\MPPP\checkpoints\mppp_mask_convnext_tiny_s4_v1.safetensors` (124,825,116 bytes);
* prints its SHA-256;
* with `--update-registry`, writes the SHA-256 and size into `src/mppp/data/models.json`.

The export is deterministic: the file depends only on the checkpoint and its card. For the checkpoint prepared with this release (`convnext_tiny_s4_seg_best.pt` = the 24 Sep 2026 run, val IoU 0.9713) the SHA-256 is

```
59b8f29bc67d4d26357ef3bda4fc3449e6c2dd734ccb878239217a52dbb21f5f
```

and `models.json` already contains it. So `git status` should show **no change**. If it shows `models.json` as modified, the file you exported is a different model. That is fine, but commit the new values: `git commit -am "mask model sha256"` and `git push`.

To check the file in Python:

```python
from mppp.mask.model import load_model
model, card = load_model(r"D:\code\MPPP\checkpoints\mppp_mask_convnext_tiny_s4_v1.safetensors")
print(card["backbone"], card["val_iou"], card["exported_from"])
```

## 3. Attach the model to a GitHub release

The registry URL is `https://github.com/ctate7163/MPPP/releases/download/mask-v1/mppp_mask_convnext_tiny_s4_v1.safetensors`. That means a release with the tag **`mask-v1`**, and the file name unchanged. A model-specific tag keeps the URL stable across code versions.

With the GitHub CLI:

```bat
gh release create mask-v1 D:\code\MPPP\checkpoints\mppp_mask_convnext_tiny_s4_v1.safetensors --repo ctate7163/MPPP --title "Mask model v1" --notes "ConvNeXt-tiny + stride-4 decoder, masks_training_set_v7, val IoU 0.971. SHA-256 59b8f29bc67d4d26357ef3bda4fc3449e6c2dd734ccb878239217a52dbb21f5f"
```

Or on the web:
1. Open the repository, then **Releases**, then **Draft a new release**.
2. Choose a tag: type `mask-v1` and create it on `main`.
3. Title it "Mask model v1".
4. Drag the `.safetensors` file into the assets box, then **Publish release**.

GitHub release assets can be up to 2 GB each.

A release for the code itself: `gh release create v0.14.0 --repo ctate7163/MPPP --title "MPPP 0.14.0" --notes-file CHANGELOG.md` (or use the web page).

## 4. Upload the model to Hugging Face

The registry URL is `https://huggingface.co/ctate7163/mppp-mask/resolve/main/mppp_mask_convnext_tiny_s4_v1.safetensors`. That means a **model** repository `ctate7163/mppp-mask` with the file at its root. If your Hugging Face user name is not `ctate7163`, change the URL in `src/mppp/data/models.json` and commit.

```bat
huggingface-cli repo create mppp-mask --type model
huggingface-cli upload ctate7163/mppp-mask D:\code\MPPP\checkpoints\mppp_mask_convnext_tiny_s4_v1.safetensors mppp_mask_convnext_tiny_s4_v1.safetensors
huggingface-cli upload ctate7163/mppp-mask docs\hf_model_card.md README.md
```

In newer `huggingface_hub` versions the command is `hf` (`hf repo create …`, `hf upload …`). Or in Python:

```python
from huggingface_hub import HfApi
api = HfApi()
api.create_repo("ctate7163/mppp-mask", repo_type="model", exist_ok=True)
api.upload_file(path_or_fileobj=r"D:\code\MPPP\checkpoints\mppp_mask_convnext_tiny_s4_v1.safetensors",
                path_in_repo="mppp_mask_convnext_tiny_s4_v1.safetensors", repo_id="ctate7163/mppp-mask")
api.upload_file(path_or_fileobj=r"docs\hf_model_card.md", path_in_repo="README.md", repo_id="ctate7163/mppp-mask")
```

Or on the web:
1. Go to https://huggingface.co/new and create a model repository named `mppp-mask`.
2. Open **Files**, then **Add file**, then **Upload files**, and add the `.safetensors` file.
3. Upload `docs/hf_model_card.md` renamed to `README.md`, as the model card.

Hugging Face stores large files with Git LFS/Xet automatically.

## 5. Check the download path

In a fresh cache, both URLs should serve the same file:

```bat
set MPPP_CACHE=%TEMP%\mppp_cache_test
python -m mppp.mask.hub fetch --name mppp_mask_v1
python -m mppp.mask.hub list
```

`fetch` downloads from the first URL that works and verifies the SHA-256. To test the GitHub URL on its own, remove the Hugging Face URL temporarily, or use `fetch_model(urls=[...])` in Python.

## 6. Releasing a new model later

1. Train and promote it (`notebooks/training/02_train_mask.ipynb`).
2. Export it under a new name so the old URL keeps working:

   ```bat
   python -m mppp.mask.hub export <checkpoints>\convnext_tiny_s4_seg_best.pt --name mppp_mask_v2 --update-registry
   ```

   For a name not yet in `models.json`, the file is `mppp_mask_v2.safetensors`, and the entry is created with an empty URL list.
3. Add the two URLs to the new entry in `src/mppp/data/models.json`: Hugging Face, and a GitHub release tagged `mask-v2`. Upload the file to both places (steps 3 and 4).
4. To make it the default, set `"default": "mppp_mask_v2"`.
5. Commit and push, and note the change in `CHANGELOG.md`.

Users who want the old model can set `config["masking"]["checkpoint"] = "mppp_mask_v1"`.

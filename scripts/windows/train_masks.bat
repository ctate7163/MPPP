@echo off
REM ======================================================================================================
REM RPP - train_masks.bat (0.95.2, Christian, 30 Oct 2026): the rover-mask model trained all the way through
REM on the newest training set, then the JOSS paper's outputs.
REM
REM   train_masks.bat [more train_mask.py options]        e.g.   train_masks.bat --epochs 12
REM   train_masks.bat --recipe A --epochs 30 --tag v9A     0.96 (9 Nov 2026): the online augmentations of the mask
REM   train_masks.bat --recipe B --epochs 30 --tag v9B     plan (rpp_studies.mask_training.augment): A lighting + LMST
REM   train_masks.bat --recipe C --epochs 30 --tag v9C     sampler; B + angular scale, defocus, quad split; C + copy-
REM                                                        paste, color matrix, family batches.  A later --tag wins
REM                                                        over the default v%VER%; --recipe turns the tone curve off.
REM   Add --test-list D:\masks\frozen_test_list.txt to every run so they are scored on the same held-out frames
REM   (make it once: python -m rpp_studies.mask_training.train_mask --write-test-list D:\masks\frozen_test_list.txt --from-checkpoint <v4.2 .pt>).
REM   train_masks.bat --recipe A --tag v9A --compare           0.96 (9 Nov, "will the mask comparison run after the
REM                                                        training?"): right after the training, compare_mask_models
REM                                                        of the new checkpoint against rpp_mask_v3 (the blocks' model;
REM                                                        add names or .pt paths after --compare for more) on its held-out
REM                                                        frames -> Claude outputs\mask_<checkpoint>\mask_ab\ and
REM                                                        mask_verdict.json, which freeze.bat --mask-model auto reads.
REM
REM 1. The training set: RPP_MASK_DATA, else the highest D:\masks\masks_training_set_vN (RPP_MASK_ROOT).
REM 2. images\ regenerated from the PDS archive (RPP_PDS) for every mask through the standard RPP pipeline
REM    (resumable: a rerun skips the done ones). Training uses images\ only (Christian, 30 Oct: no images_variable\;
REM    add --variable to train on the brightness-varied copies too).
REM 3. ConvNeXt-tiny (ImageNet-22k start), 20 epochs (as v4), split grouped by mask; augmentation: horizontal flips,
REM    the large Navcam / Hazcam frames zoomed in by up to 2x somewhere in the frame (half of them) before they are
REM    downsampled to the canvas, and (0.95.3, mask v4.1, Christian 1 Nov) a random tone curve on half of the samples
REM    (black and white point, gamma, channel gains - v3's images_variable recipe, drawn anew each time).  -> checkpoints\convnext_tiny_s4_seg_<date>_vN.pt + .json card + _debug\ (curves,
REM    held-out previews, frame audit, worst frames).
REM 4. "Claude outputs\mask_<checkpoint>\": figure1_masks.png/.pdf (held-out frames, predicted exclusion orange, label
REM    boundary blue), mask_metrics_by_camera.md/.csv, training_curves.png, the card, summary.md (the training set and
REM    the numbers for the paper) and the release candidate .safetensors.  Nothing is promoted or released.
REM The Python must have a CUDA torch, timm and rpp: RPP_TRAIN_PYTHON, else the pipeline's python, else
REM GPU_PYTHON - the first with CUDA is used.  The GPU is shared with any running alignment.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "RPP_HOME=%%~fI"
set "RPP_WIN=%RPP_HOME%\scripts\windows"
call "%RPP_WIN%\rpp_env.bat" || ( pause & exit /b 1 )
set "KMP_DUPLICATE_LIB_OK=TRUE"
if not defined RPP_PDS set "RPP_PDS=D:\data\m2020"
if not defined RPP_MASK_ROOT set "RPP_MASK_ROOT=D:\masks"

REM --- 1. the newest training set
set "VER="
if defined RPP_MASK_DATA (
  for %%D in ("%RPP_MASK_DATA%") do set "VN=%%~nxD"
  set "VER=!VN:masks_training_set_v=!"
) else (
  set "BEST=-1"
  for /d %%D in ("%RPP_MASK_ROOT%\masks_training_set_v*") do (
    set "VN=%%~nxD"
    set "VN=!VN:masks_training_set_v=!"
    set /a "VI=!VN!" 2>nul
    if "!VI!"=="!VN!" if !VI! GTR !BEST! ( set "BEST=!VI!" & set "RPP_MASK_DATA=%%~fD" )
  )
  set "VER=!BEST!"
)
if not defined RPP_MASK_DATA ( echo no %RPP_MASK_ROOT%\masks_training_set_vN found: set RPP_MASK_DATA & pause & exit /b 1 )
if not exist "%RPP_MASK_DATA%\masks\" ( echo %RPP_MASK_DATA% has no masks\ folder & pause & exit /b 1 )

REM --- 2. a Python with a CUDA torch
set "TPY="
for %%P in ("%RPP_TRAIN_PYTHON%" "python" "%GPU_PYTHON%") do if not defined TPY if not "%%~P"=="" (
  "%%~P" -c "import sys, torch, timm, rpp; sys.exit(0 if torch.cuda.is_available() else 3)" >nul 2>nul && set "TPY=%%~P"
)
if not defined TPY (
  echo No Python with a CUDA torch, timm and rpp. Set RPP_TRAIN_PYTHON to one, e.g. the notebooks' python
  echo ^(in a notebook:  import sys; print^(sys.executable^)^); check:  python -c "import torch; print(torch.cuda.is_available())"
  pause
  exit /b 1
)

echo %DATE% %TIME%  train_masks: %RPP_MASK_DATA% (v%VER%), PDS %RPP_PDS%, python %TPY%
"%TPY%" -m rpp_studies.mask_training.train_mask --data "%RPP_MASK_DATA%" --pds "%RPP_PDS%" --regen --epochs 20 --tag v%VER% --zoom-max 2 --zoom-prob 0.5 --tone-prob 0.5 --paper %*
set "RC=%ERRORLEVEL%"
echo %DATE% %TIME%  train_masks done ^(exit %RC%^): checkpoints in %RPP_HOME%\checkpoints, paper outputs in "%RPP_HOME%\Claude outputs\mask_*"
if not defined RPP_NOPAUSE pause
exit /b %RC%

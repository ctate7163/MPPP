@echo off
REM ======================================================================================================
REM MPPP - train_masks.bat (0.95.2, Christian, 30 Oct 2026): the rover-mask model trained all the way through
REM on the newest training set, then the JOSS paper's outputs.
REM
REM   train_masks.bat [more train_mask.py options]        e.g.   train_masks.bat --epochs 12
REM
REM 1. The training set: MPPP_MASK_DATA, else the highest D:\masks\masks_training_set_vN (MPPP_MASK_ROOT).
REM 2. images\ regenerated from the PDS archive (MPPP_PDS) for every mask through the standard MPPP pipeline
REM    (resumable: a rerun skips the done ones). Training uses images\ only (Christian, 30 Oct: no images_variable\;
REM    add --variable to train on the brightness-varied copies too).
REM 3. ConvNeXt-tiny (ImageNet-22k start), 10 epochs, split grouped by mask; augmentation: horizontal flips, and the
REM    large Navcam / Hazcam frames zoomed in by up to 2x somewhere in the frame (half of them) before they are
REM    downsampled to the canvas.  -> checkpoints\convnext_tiny_s4_seg_<date>_vN.pt + .json card + _debug\ (curves,
REM    held-out previews, frame audit, worst frames).
REM 4. "Claude outputs\mask_<checkpoint>\": figure1_masks.png/.pdf (held-out frames, predicted exclusion orange, label
REM    boundary blue), mask_metrics_by_camera.md/.csv, training_curves.png, the card, summary.md (the training set and
REM    the numbers for the paper) and the release candidate .safetensors.  Nothing is promoted or released.
REM The Python must have a CUDA torch, timm and mppp: MPPP_TRAIN_PYTHON, else the pipeline's python, else
REM GPU_PYTHON - the first with CUDA is used.  The GPU is shared with any running alignment.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
set "KMP_DUPLICATE_LIB_OK=TRUE"
if not defined MPPP_PDS set "MPPP_PDS=D:\data\m2020"
if not defined MPPP_MASK_ROOT set "MPPP_MASK_ROOT=D:\masks"

REM --- 1. the newest training set
set "VER="
if defined MPPP_MASK_DATA (
  for %%D in ("%MPPP_MASK_DATA%") do set "VN=%%~nxD"
  set "VER=!VN:masks_training_set_v=!"
) else (
  set "BEST=-1"
  for /d %%D in ("%MPPP_MASK_ROOT%\masks_training_set_v*") do (
    set "VN=%%~nxD"
    set "VN=!VN:masks_training_set_v=!"
    set /a "VI=!VN!" 2>nul
    if "!VI!"=="!VN!" if !VI! GTR !BEST! ( set "BEST=!VI!" & set "MPPP_MASK_DATA=%%~fD" )
  )
  set "VER=!BEST!"
)
if not defined MPPP_MASK_DATA ( echo no %MPPP_MASK_ROOT%\masks_training_set_vN found: set MPPP_MASK_DATA & pause & exit /b 1 )
if not exist "%MPPP_MASK_DATA%\masks\" ( echo %MPPP_MASK_DATA% has no masks\ folder & pause & exit /b 1 )

REM --- 2. a Python with a CUDA torch
set "TPY="
for %%P in ("%MPPP_TRAIN_PYTHON%" "python" "%GPU_PYTHON%") do if not defined TPY if not "%%~P"=="" (
  "%%~P" -c "import sys, torch, timm, mppp; sys.exit(0 if torch.cuda.is_available() else 3)" >nul 2>nul && set "TPY=%%~P"
)
if not defined TPY (
  echo No Python with a CUDA torch, timm and mppp. Set MPPP_TRAIN_PYTHON to one, e.g. the notebooks' python
  echo ^(in a notebook:  import sys; print^(sys.executable^)^); check:  python -c "import torch; print(torch.cuda.is_available())"
  pause
  exit /b 1
)

echo %DATE% %TIME%  train_masks: %MPPP_MASK_DATA% (v%VER%), PDS %MPPP_PDS%, python %TPY%
"%TPY%" "%MPPP_HOME%\scripts\train_mask.py" --data "%MPPP_MASK_DATA%" --pds "%MPPP_PDS%" --regen --epochs 10 --tag v%VER% --zoom-max 2 --zoom-prob 0.5 --paper %*
set "RC=%ERRORLEVEL%"
echo %DATE% %TIME%  train_masks done ^(exit %RC%^): checkpoints in %MPPP_HOME%\checkpoints, paper outputs in "%MPPP_HOME%\Claude outputs\mask_*"
if not defined MPPP_NOPAUSE pause
exit /b %RC%

@echo off
REM MPPP v0p22 batch: Taylor Fjellet, Rockytop, Belva, Three Forks 670-694 and the landing site,
REM Navcam + Mastcam-Z 34 mm, every image processed again, then notebooks 04 (tracks >= 3) and 05.
REM Progress: D:\scapes\v0p22\batch_log.txt (one line per notebook cell).
REM Run it from an Anaconda Prompt in the environment that runs your notebooks (conda "base"),
REM or double-click it: it then tries to activate conda "base" itself.
cd /d D:\code\MPPP
where python >nul 2>nul
if errorlevel 1 (
  if exist "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" call "%LOCALAPPDATA%\miniconda3\Scripts\activate.bat" base
  if exist "%USERPROFILE%\miniconda3\Scripts\activate.bat" call "%USERPROFILE%\miniconda3\Scripts\activate.bat" base
)
python -c "import pycolmap, pyceres, nbclient; print('python OK:', __import__('sys').executable)" || (echo This python lacks pycolmap/pyceres/nbclient - open an Anaconda Prompt in the notebook environment and run this file again. & pause & exit /b 1)
python scripts\run_scapes.py --sites taylorfjellet rockytop belva threeforks_large landing --zcam --reprocess ^
    --root D:\scapes\v0p22 ^
    --gpu-py "%LOCALAPPDATA%\miniconda3\envs\mppp_gpu\python.exe" ^
    --mask-off threeforks_large=S032D1184
pause

@echo off
REM run_single_site_ibr.bat [LABEL ...] [--steps ...]  (written by python -m mppp single-sites --install)
REM 1. D:\code\MPPP fast-forwarded to the newest git bundle in %TRANSFER% - if that bundle is ahead of it
REM    (a bundle behind it, e.g. after a pull from GitHub, is left alone; nothing is overwritten)
REM 2. python -m mppp single-sites with the arguments given: the stations of src\mppp\data\single_sites.json
REM    0.97.0: the Perseverance model in the rover mesh, the panorama blended in 16 bits, ONE 16-to-8-bit scaling per
REM    station for the rover-shown and rover-masked versions (<station>\tone_levels.json), the per-image pictures as
REM    jpg in photometry_jpg (rad, depth linear / log, normal, incidence, emission, phase, shadow), a centre panorama
REM    with its depth, the viewpoint on the azimuth axis at the Mastcam-Z height
REM    examples:  run_single_site_ibr.bat Sol0052_S003D2046            one station
REM               run_single_site_ibr.bat --steps ods          the panoramas again (mesh and maps kept)
REM               run_single_site_ibr.bat --dry-run            the commands, not run
REM set MPPP_NO_PAUSE=1 before it: no pause at the end, and the exit code is the run's (for unattended chains)
REM Nothing in this file names a version, so it never needs replacing.
REM It also lives in scripts\windows of the repository (0.97.0): there it works on the tree it is in.  Everything
REM after the settings is one parenthesised block, read whole before it runs - the merge may replace this file.
REM A copy put into scripts\windows by hand (untracked) is removed before the merge, which brings the tracked one.
setlocal EnableDelayedExpansion
set "REPO=%MPPP_HOME%"
if not defined REPO if exist "%~dp0mppp_env.bat" for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
if not defined REPO set "REPO=D:\code\MPPP"
set "TRANSFER=D:\rover_blocks\_transfer"
cd /d "%REPO%" || ( echo no %REPO% & pause & exit /b 1 )
set "B="
(
  for /f "delims=" %%f in ('dir /b /o-d "%TRANSFER%\mppp_*.bundle" 2^>nul') do if not defined B set "B=%TRANSFER%\%%f"
  for /f %%h in ('git rev-parse HEAD') do set "H0=%%h"
  if defined B (
    git fetch -q "!B!" +main:refs/remotes/bundle/main
    if errorlevel 1 ( echo fetch from !B! failed - send Claude the lines above & pause & exit /b 1 )
    git merge-base --is-ancestor HEAD refs/remotes/bundle/main
    if errorlevel 1 (
      echo the newest bundle !B! is not ahead of %REPO% - the code is kept as it is
    ) else (
      git ls-files --error-unmatch scripts/windows/run_single_site_ibr.bat >nul 2>nul
      if errorlevel 1 if exist scripts\windows\run_single_site_ibr.bat del /q scripts\windows\run_single_site_ibr.bat
      git merge -q --ff-only refs/remotes/bundle/main
      if errorlevel 1 ( echo fast-forward failed - send Claude the lines above & pause & exit /b 1 )
    )
  )
  for /f %%h in ('git rev-parse HEAD') do set "H1=%%h"
  echo %REPO%: !H0:~0,7! -^> !H1:~0,7!
  call scripts\windows\mppp_env.bat || ( pause & exit /b 1 )
  python -m mppp single-sites %*
  set "RC=!errorlevel!"
  if not defined MPPP_NO_PAUSE pause
  exit /b !RC!
)

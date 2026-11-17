@echo off
REM run_single_site_ibr.bat [LABEL ...] [--steps ...]  (written by python -m rpp single-sites --install)
REM 1. the tree fast-forwarded to the newest git bundle in %TRANSFER% - if that bundle is ahead of it; merged
REM    when both have commits of their own (0.97.21); left alone when it already holds the bundle; nothing overwritten.
REM    0.99.3: the tree's own line - on branch rpp (D:\code\RPP) the newest rpp_v*.bundle's rpp, on main
REM    (D:\code\MPPP) the newest mppp_v*.bundle's main (0.99.0-0.99.2 fetched main from an rpp bundle, which has none)
REM 2. python -m rpp single-sites with the arguments given: the stations of src\rpp\data\single_sites.json
REM    0.97.0: the Perseverance model in the rover mesh, the panorama blended in 16 bits, ONE 16-to-8-bit scaling per
REM    station for the rover-shown and rover-masked versions (<station>\tone_levels.json), the per-image pictures as
REM    jpg in photometry_jpg (rad, depth linear / log, normal, incidence, emission, phase, shadow), a center panorama
REM    with its depth, the viewpoint on the azimuth axis at the Mastcam-Z height
REM    examples:  run_single_site_ibr.bat Sol0052_S003D2046            one station
REM               run_single_site_ibr.bat --steps ods          the panoramas again (mesh and maps kept)
REM               run_single_site_ibr.bat --dry-run            the commands, not run
REM               run_single_site_ibr.bat Sol0052_S003D2046 --no-upload  the run without the upload at its end
REM               run_single_site_ibr.bat --publish-only                 no rendering: publish and push what is there
REM    0.97.24: when the run is done its panoramas are published and uploaded (--upload is given here; --no-upload
REM    after it wins).  0.99.3: the pictures go to the Hugging Face dataset ctate7163/rpv-panoramas, station by station,
REM    and panos.json to the viewer's GitHub checkout (D:\code\RPV), committed and pushed; --host github puts the
REM    pictures into that checkout as before.  A push that fails is tried three times and goes up with the next run:
REM    see "NOT (all) uploaded" at the end and upload_pending.json in the viewer's folder.
REM set RPP_NO_PAUSE=1 before it: no pause at the end, and the exit code is the run's (for unattended chains)
REM Nothing in this file names a version, so it never needs replacing.
REM It also lives in scripts\windows of the repository (0.97.0): there it works on the tree it is in.  Everything
REM after the settings is one parenthesized block, read whole before it runs - the merge may replace this file.
REM A copy put into scripts\windows by hand (untracked) is removed before the merge, which brings the tracked one.
setlocal EnableDelayedExpansion
set "REPO="
if exist "%~dp0rpp_env.bat" for %%I in ("%~dp0..\..") do set "REPO=%%~fI"
if not defined REPO set "REPO=%RPP_HOME%"
if not defined REPO set "REPO=D:\code\RPP"
set "TRANSFER=D:\rover_blocks\_transfer"
cd /d "%REPO%" || ( echo no %REPO% & pause & exit /b 1 )
set "B="
set "BR=main"
(
  for /f %%b in ('git rev-parse --abbrev-ref HEAD') do set "BR=%%b"
  set "PAT=mppp_v*.bundle"
  if /i "!BR!"=="rpp" set "PAT=rpp_v*.bundle"
  for /f "delims=" %%f in ('dir /b /o-d "%TRANSFER%\!PAT!" 2^>nul') do if not defined B set "B=%TRANSFER%\%%f"
  for /f %%h in ('git rev-parse HEAD') do set "H0=%%h"
  if defined B (
    git fetch -q "!B!" +!BR!:refs/remotes/bundle/!BR!
    if errorlevel 1 ( echo fetch of !BR! from !B! failed - send Claude the lines above & pause & exit /b 1 )
    git merge-base --is-ancestor HEAD refs/remotes/bundle/!BR!
    if errorlevel 1 (
      git merge-base --is-ancestor refs/remotes/bundle/!BR! HEAD
      if errorlevel 1 (
        git merge -q --no-edit refs/remotes/bundle/!BR!
        if errorlevel 1 ( git merge --abort & echo the newest bundle !B! does not merge cleanly into %REPO% - send Claude the lines above & pause & exit /b 1 )
        echo the newest bundle !B! merged into %REPO%
      ) else (
        echo %REPO% already holds the newest bundle !B!
      )
    ) else (
      git ls-files --error-unmatch scripts/windows/run_single_site_ibr.bat >nul 2>nul
      if errorlevel 1 if exist scripts\windows\run_single_site_ibr.bat del /q scripts\windows\run_single_site_ibr.bat
      git merge -q --ff-only refs/remotes/bundle/!BR!
      if errorlevel 1 ( echo fast-forward failed - send Claude the lines above & pause & exit /b 1 )
    )
  )
  for /f %%h in ('git rev-parse HEAD') do set "H1=%%h"
  echo %REPO%: !H0:~0,7! -^> !H1:~0,7!
  call scripts\windows\rpp_env.bat || ( pause & exit /b 1 )
  python -m rpp single-sites --upload %*
  set "RC=!errorlevel!"
  if not defined RPP_NO_PAUSE pause
  exit /b !RC!
)

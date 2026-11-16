@echo off
REM ======================================================================================================
REM RPP 0.96.0 - sync_code.bat (9 Nov 2026, Christian: "let's clean up the MPPP directory so that these conflicts
REM don't keep happening").  One way to bring code onto this PC, nothing else:
REM
REM   sync_code.bat                    0.98.19 (Christian, 16 Nov 2026: "so that I can just run sync_code.bat
REM                                    without needing to install specific bundles"): the newest bundle Claude
REM                                    delivered - the latest rpp_v*.bundle in RPP_TRANSFER (default
REM                                    D:\rover_blocks\_transfer); every bundle holds the whole history since a314fa6
REM   sync_code.bat BUNDLE [BRANCH]    this tree's main fast-forwarded to BRANCH (default main) of a git bundle
REM                                    Claude delivered (D:\rover_blocks\_transfer\rpp_<...>.bundle)
REM   sync_code.bat --github           this tree's main fast-forwarded to origin/main (git pull --ff-only)
REM
REM Then, always: the quick tests, one file per process with MKL sequential (torch's and MKL's OpenMP in one process
REM aborted the 9 Nov runs; failures are listed, not fatal), and the freeze checkout (RPP_FREEZE_CODE, default
REM D:\code\MPPP_freeze) fast-forwarded to this tree's main - unless a freeze is running on it (freeze.bat --status
REM says so; the code must not change under a run).  Nothing is started.  Replaces claude_sync.bat (0.95.0).
REM
REM When main cannot fast-forward (this tree has commits the delivery lacks - the viewer's "Viewer: panoramas" publish
REM commits) the bundle is merged (0.98.25); it stops, changing nothing, only when that merge has conflicts (send
REM Claude `git log --oneline -5` and `git status`).  Uncommitted changes to tracked files go to a stash and come back after the merge.
REM Runs from a copy in %TEMP% (cmd.exe reads a batch file line by line, and the merge may replace this file).
REM Log: %TEMP%\rpp_sync_<test>.log per test file.  Then: git push origin main (you), freeze.bat.
REM ======================================================================================================
if /i not "%~1"=="--from-copy" (
  copy /y "%~f0" "%TEMP%\rpp_sync_code_run.bat" >nul || ( echo cannot copy %~f0 to %TEMP% & pause & exit /b 1 )
  "%TEMP%\rpp_sync_code_run.bat" --from-copy "%~dp0" %*
  exit /b %ERRORLEVEL%
)
setlocal EnableExtensions EnableDelayedExpansion
set "HERE=%~2"
shift
shift
for %%I in ("%HERE%..\..") do set "REPO=%%~fI"
call "%HERE%rpp_env.bat" --legacy-only
if not defined RPP_FREEZE_CODE set "RPP_FREEZE_CODE=D:\code\MPPP_freeze"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "MKL_THREADING_LAYER=SEQUENTIAL"
set "RPP_NOPAUSE=1"
cd /d "%REPO%" || ( echo no %REPO% & exit /b 1 )
REM 0.99.0: this file updates main and the freeze checkout; the rpp worktree D:\code\RPP has sync_rpp.bat
set "BRANCH_HERE="
for /f %%b in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH_HERE=%%b"
if /i not "!BRANCH_HERE!"=="main" ( echo %REPO% is on branch !BRANCH_HERE!, not main - sync_code.bat is for main: use sync_rpp.bat here & exit /b 1 )
call "%REPO%\scripts\windows\rpp_env.bat" || exit /b 1
if exist .git\rebase-apply ( echo removing a stray .git\rebase-apply & rmdir /s /q .git\rebase-apply )
for /f %%h in ('git rev-parse --short HEAD') do set "HEAD0=%%h"
set "STASHED="
for /f "delims=" %%s in ('git status --porcelain --untracked-files=no') do set "STASHED=1"
if defined STASHED ( echo stashing local changes to tracked files & git stash push -q -m "sync_code %DATE% %TIME%" || exit /b 1 )
set "BUNDLE=%~1"
if not defined RPP_TRANSFER set "RPP_TRANSFER=D:\rover_blocks\_transfer"
if "%~1"=="" call :newest_bundle
if /i "%~1"=="--github" (
  git pull -q --ff-only origin main || ( echo fast-forward from GitHub failed - this tree has commits origin/main lacks & goto :unstash_fail )
) else (
  if "!BUNDLE!"=="" ( echo no rpp_v*.bundle in %RPP_TRANSFER% - usage: sync_code.bat [BUNDLE [BRANCH]]  ^|  sync_code.bat --github & goto :unstash_fail )
  echo bundle: !BUNDLE!
  set "BRANCH=%~2"
  if "!BRANCH!"=="" set "BRANCH=main"
  if not exist "!BUNDLE!" ( echo !BUNDLE! not found & goto :unstash_fail )
  git bundle verify "!BUNDLE!" >nul || ( echo the bundle does not fit this repository ^(a commit it builds on is missing here^) & goto :unstash_fail )
  git fetch -q "!BUNDLE!" !BRANCH!:refs/remotes/bundle/!BRANCH! || ( echo fetch failed & goto :unstash_fail )
  git merge -q --ff-only refs/remotes/bundle/!BRANCH! 2>nul || (
    echo main has commits the bundle does not ^(the viewer's publish commits^): merging
    git merge -q --no-edit refs/remotes/bundle/!BRANCH! || ( git merge --abort & echo the merge has conflicts - nothing changed; send Claude: git log --oneline -5 and git status & goto :unstash_fail )
  )
)
if defined STASHED ( git stash pop -q || echo WARNING: stash pop had conflicts - resolve them )
for /f %%h in ('git rev-parse --short HEAD') do set "HEAD1=%%h"
for /f "delims=" %%v in ('python -c "import rpp; print(rpp.__version__)"') do set "VER=%%v"
echo %REPO%: %HEAD0% -^> %HEAD1%  (rpp %VER%)
set "FAILED="
for %%T in (test_package test_cli test_scripts test_freeze test_waypoints_sites test_sfm_zcam test_hazfit_zfixed test_mask_augment) do (
  if exist "tests\%%T.py" (
    python -m pytest "tests\%%T.py" -q -x -p no:cacheprovider > "%TEMP%\rpp_sync_%%T.log" 2>&1
    if errorlevel 1 ( set "FAILED=!FAILED! %%T" & echo   FAIL %%T  ^(log %TEMP%\rpp_sync_%%T.log^) ) else ( echo   ok   %%T )
  )
)
if exist "%RPP_FREEZE_CODE%\.git" (
  python -m rpp_studies.calibration.freeze --code "%RPP_FREEZE_CODE%" --from-tree "%REPO%" --update-code
) else (
  echo freeze checkout %RPP_FREEZE_CODE%: none yet ^(freeze.bat clones it^)
)
echo.
if defined FAILED ( echo sync done; tests FAILED:%FAILED%  - send Claude the logs named above before starting a freeze ) else ( echo sync done, the quick tests passed )
echo next:  git push origin main   ^(you^)      freeze.bat --status  /  freeze.bat
exit /b 0
:unstash_fail
if defined STASHED git stash pop -q
exit /b 1
:newest_bundle
set "BUNDLE="
for /f "delims=" %%b in ('dir /b /a-d /o-d "%RPP_TRANSFER%\rpp_v*.bundle" 2^>nul') do if not defined BUNDLE set "BUNDLE=%RPP_TRANSFER%\%%b"
exit /b 0

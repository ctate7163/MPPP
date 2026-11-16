@echo off
REM ======================================================================================================
REM RPP 0.99.0 - sync_rpp.bat: brings v0p99 code into the rpp worktree D:\code\RPP, nothing else.
REM
REM   sync_rpp.bat               the newest rpp_v*.bundle in RPP_TRANSFER (default D:\rover_blocks\_transfer)
REM   sync_rpp.bat BUNDLE        that bundle (its branch rpp)
REM
REM This tree's branch rpp is fast-forwarded to the bundle's rpp (merged when this tree has commits of its own; on a
REM conflict it stops and changes nothing), then the quick tests run.  It never touches main, the checkout in
REM D:\code\MPPP or the freeze checkout D:\code\MPPP_freeze: sync_code.bat does those, from D:\code\MPPP.
REM Uncommitted changes to tracked files go to a stash and come back after the merge.
REM Runs from a copy in %TEMP% (cmd.exe reads a batch file line by line, and the merge may replace this file).
REM Log: %TEMP%\rpp_sync_rpp_<test>.log per test file.
REM ======================================================================================================
if /i not "%~1"=="--from-copy" (
  copy /y "%~f0" "%TEMP%\rpp_sync_rpp_run.bat" >nul || ( echo cannot copy %~f0 to %TEMP% & pause & exit /b 1 )
  "%TEMP%\rpp_sync_rpp_run.bat" --from-copy "%~dp0" %*
  exit /b %ERRORLEVEL%
)
setlocal EnableExtensions EnableDelayedExpansion
set "HERE=%~2"
shift
shift
for %%I in ("%HERE%..\..") do set "REPO=%%~fI"
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "MKL_THREADING_LAYER=SEQUENTIAL"
set "RPP_NOPAUSE=1"
cd /d "%REPO%" || ( echo no %REPO% & exit /b 1 )
set "BRANCH_HERE="
for /f %%b in ('git rev-parse --abbrev-ref HEAD') do set "BRANCH_HERE=%%b"
if /i not "!BRANCH_HERE!"=="rpp" ( echo %REPO% is on branch !BRANCH_HERE!, not rpp - sync_rpp.bat is for the rpp worktree D:\code\RPP & exit /b 1 )
call "%REPO%\scripts\windows\rpp_env.bat" || exit /b 1
for /f %%h in ('git rev-parse --short HEAD') do set "HEAD0=%%h"
set "STASHED="
for /f "delims=" %%s in ('git status --porcelain --untracked-files=no') do set "STASHED=1"
if defined STASHED ( echo stashing local changes to tracked files & git stash push -q -m "sync_rpp %DATE% %TIME%" || exit /b 1 )
if not defined RPP_TRANSFER set "RPP_TRANSFER=D:\rover_blocks\_transfer"
set "BUNDLE=%~1"
if "!BUNDLE!"=="" for /f "delims=" %%b in ('dir /b /a-d /o-d "%RPP_TRANSFER%\rpp_v*.bundle" 2^>nul') do if "!BUNDLE!"=="" set "BUNDLE=%RPP_TRANSFER%\%%b"
if "!BUNDLE!"=="" ( echo no rpp_v*.bundle in %RPP_TRANSFER% & goto :unstash_fail )
echo bundle: !BUNDLE!
git bundle verify "!BUNDLE!" >nul || ( echo the bundle does not fit this repository - a commit it builds on is missing & goto :unstash_fail )
git fetch -q "!BUNDLE!" rpp:refs/remotes/bundle/rpp || ( echo fetch failed & goto :unstash_fail )
git merge -q --ff-only refs/remotes/bundle/rpp 2>nul || (
  echo rpp has commits the bundle does not: merging
  git merge -q --no-edit refs/remotes/bundle/rpp || ( git merge --abort & echo the merge has conflicts - nothing changed; send Claude: git log --oneline -5 and git status & goto :unstash_fail )
)
if defined STASHED ( git stash pop -q || echo WARNING: stash pop had conflicts - resolve them )
for /f %%h in ('git rev-parse --short HEAD') do set "HEAD1=%%h"
for /f "delims=" %%v in ('python -c "import rpp; print(rpp.__version__)"') do set "VER=%%v"
echo %REPO%: %HEAD0% -^> %HEAD1%  (rpp %VER%)
set "FAILED="
for %%T in (test_package test_rename test_cli test_scripts test_waypoints_sites test_processing) do (
  if exist "tests\%%T.py" (
    python -m pytest "tests\%%T.py" -q -x -p no:cacheprovider > "%TEMP%\rpp_sync_rpp_%%T.log" 2>&1
    if errorlevel 1 ( set "FAILED=!FAILED! %%T" & echo   FAIL %%T  ^(log %TEMP%\rpp_sync_rpp_%%T.log^) ) else ( echo   ok   %%T )
  )
)
echo.
if defined FAILED ( echo sync done; tests FAILED:!FAILED! - send Claude the logs named above ) else ( echo sync done, the quick tests passed )
echo next: scripts\windows\check_rpp_reuse.bat   (a dry run: v0p99 and v0p98 must skip the same finished blocks)
exit /b 0
:unstash_fail
if defined STASHED git stash pop -q
exit /b 1

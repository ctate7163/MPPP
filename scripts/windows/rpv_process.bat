@echo off
REM ======================================================================================================
REM RPP 1.0 - rpv_process.bat: the station products for RPV, the Rover Photogrammetry Viewer, from the aligned
REM blocks of a root (default D:\rover_blocks_f3), into <root>\single_sites, and onto the viewer.  It is the
REM rpv_products_0p99p53.bat of RPP 0.99.53 (Christian, 20 Nov 2026: "Start from the rpv_products process of version
REM v0p99p52. in v1p0, call this rpv_process.bat"), with update_rpp_rpv.bat and unpublish_station.bat in it.
REM
REM   rpv_process.bat                       all stations: the Rockytop ones first (sol order, each Mastcam-Z version
REM                                         right after its Navcam station), then all others from the latest sol back
REM   rpv_process.bat --recolor             no meshing, projection or rendering: every made station's colors again
REM                                         from its 16-bit panoramas and textures (8K or 16K), then uploaded
REM   rpv_process.bat --dry-run             which stations, in which order (nothing run, nothing uploaded)
REM   rpv_process.bat --no-upload           made, not uploaded
REM   rpv_process.bat --force               every station made again
REM   rpv_process.bat --unpublish LABEL ... those stations taken off the viewer page (their files stay on Hugging Face;
REM                                         a later run of a station puts it back)
REM
REM What a run does:
REM   - only aligned blocks (colmap\run_done.json); a station's block is its own, else the root's block with the most of
REM     the station's frames; blocks not aligned yet are looked at every 15 minutes; it stops when no station is left,
REM     when no freeze runs on the root and nothing is left to run, or when no block came in for 36 hours;
REM   - a station whose products were made by RPP 0.99 or newer with the same station code (rpp.ibr code_key) is
REM     skipped; a station another window holds (its station.lock) is left for later; one window makes a block's
REM     cloud at a time, and one publishes at a time;
REM   - the viewer page goes up first, and again with every station's upload (Hugging Face ctate7163/rpp-rpv, the page
REM     and panos.json in D:\code\rpv).
REM
REM Start it in as many windows as you like, up to four (for example one run and one --recolor): each window takes the
REM first free checkout of D:\code\RPP_rpv, RPP_rpv_b, RPP_rpv_c, RPP_rpv_d - free when no station run with its name
REM (--instance rpv, rpv_b, ...) is going - so that the code of a run never changes under it.  Before a run its checkout
REM is brought to this tree's branch (cloned from this tree the first time).
REM Stop station runs of RPP 0.99.52 or older first: they take no locks.
REM Settings: RPP_RPV_ROOT (default D:\rover_blocks_f3); "set RPP_IBR_WORKERS=4" for fewer render processes beside a
REM freeze.  Log: <root>\single_sites\rpv_process_log_<checkout>.txt; status: single_sites_status_<checkout>.json there.
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "TREE=%%~fI"
if not defined RPP_RPV_ROOT set "RPP_RPV_ROOT=D:\rover_blocks_f3"
set "ROOT=%RPP_RPV_ROOT%"
if not exist "%ROOT%" ( echo %ROOT% not found - set RPP_RPV_ROOT to the blocks' root & exit /b 1 )
if not exist "%ROOT%\single_sites" mkdir "%ROOT%\single_sites"
set "MODE=run"
set "DRY="
for %%a in (%*) do if /i "%%~a"=="--recolor" set "MODE=recolor"
for %%a in (%*) do if /i "%%~a"=="--unpublish" set "MODE=unpublish"
for %%a in (%*) do if /i "%%~a"=="--dry-run" set "DRY=1"
for %%a in (%*) do if /i "%%~a"=="--no-upload" set "DRY=1"
if exist "D:\code\rpv\.git\index.lock" (
  echo D:\code\rpv\.git\index.lock is there: a git command in D:\code\rpv was stopped or is still running.
  echo If no git command runs, delete that file, then start this again.
  exit /b 1
)

REM ---- the first free checkout: no station run with its name is going (a checkout in use is never changed)
set "SLOT="
for %%s in (rpv rpv_b rpv_c rpv_d) do if not defined SLOT (
  powershell -NoProfile -Command "$p = @(Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'single-sites' -and $_.CommandLine -match '--instance %%s( |$)' }); foreach ($x in $p) { Write-Output ('  D:\code\RPP_%%s is in use by pid ' + $x.ProcessId) }; if ($p.Count -gt 0) { exit 3 }; exit 0"
  if not errorlevel 3 set "SLOT=%%s"
)
if not defined SLOT ( echo all four station-run checkouts are in use - wait for one to end & exit /b 1 )
set "T=D:\code\RPP_%SLOT%"
set "LOG=%ROOT%\single_sites\rpv_process_log_%SLOT%.txt"
echo station-run checkout: %T%

REM ---- the checkout: this tree's branch
set "BR="
for /f %%b in ('git -C "%TREE%" rev-parse --abbrev-ref HEAD') do set "BR=%%b"
if not defined BR ( echo %TREE% is not a git checkout & exit /b 1 )
if /i "%BR%"=="HEAD" ( echo %TREE% has no branch checked out ^(a detached HEAD^): git -C "%TREE%" checkout main & exit /b 1 )
if exist "%T%\.git\refs\remotes\home\%BR%.lock" (
  echo %T%\.git\refs\remotes\home\%BR%.lock is there: an update of %T% was stopped or is still running.
  echo If no other rpv_process window is open, delete that file, then start this again.
  exit /b 1
)
if not exist "%T%\.git" (
  echo making the checkout %T% from %TREE%
  git clone -q --no-hardlinks -b %BR% "%TREE%" "%T%" || ( echo the clone failed & exit /b 1 )
)
git -C "%T%" fetch -q "%TREE%" +%BR%:refs/remotes/home/%BR% || ( echo the fetch from %TREE% failed & exit /b 1 )
git -C "%T%" checkout -q -f -B %BR% refs/remotes/home/%BR% || ( echo the checkout of %BR% in %T% failed & exit /b 1 )
for %%f in ("%TREE%\scripts\windows\rpp_*.txt" "%TREE%\scripts\windows\mppp_*.txt") do copy /y "%%~f" "%T%\scripts\windows\" >nul 2>nul
set "RPP_HOME=%T%"
set "PYTHONPATH="
if not defined RPP_CHECKPOINTS if exist "%TREE%\checkpoints" set "RPP_CHECKPOINTS=%TREE%\checkpoints"
set "RPP_NOPAUSE=1"
call "%T%\scripts\windows\rpp_env.bat" || exit /b 1
cd /d "%T%" || exit /b 1
python -c "import rpp, sys; print('rpp', rpp.__version__, rpp.__file__); ok = rpp.__file__.lower().startswith(r'%T%'.lower()) and tuple(int(x) for x in rpp.__version__.split('.')[:2]) >= (1, 0); sys.exit(0 if ok else 1)"
if errorlevel 1 ( echo python does not run RPP 1.0 or later from %T% & exit /b 1 )

if "%MODE%"=="unpublish" (
  python -m rpp --log "%LOG%" single-sites %* --upload
  exit /b !ERRORLEVEL!
)
if not defined DRY (
  echo ---- the viewer page first ^(published with Van Zyl of D:\rover_blocks, nothing rendered^)
  python -m rpp --log "%LOG%" single-sites Sol0052_S003D2046 --root D:\rover_blocks --any-block --publish-only --upload
)
set "RPP_SCAPES=%ROOT%"
echo ==== %DATE% %TIME% rpv_process %SLOT% %* (rpp in %T%, branch %BR%) >> "%LOG%"
set "MORE=--wait-for-blocks 36 --skip-done"
if "%MODE%"=="recolor" set "MORE="
python -m rpp --log "%LOG%" single-sites --all --root "%ROOT%" --block-cameras NF --terrain block --first rockytop --reverse --instance %SLOT% %MORE% --upload %*
set "RC=%ERRORLEVEL%"
echo ==== %DATE% %TIME% exit %RC% >> "%LOG%"
echo exit %RC% - %LOG%
exit /b %RC%

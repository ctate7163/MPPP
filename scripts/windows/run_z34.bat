@echo off
REM ======================================================================================================
REM MPPP 0.91 - the long Mastcam-Z 34 mm run and calibration (Christian, 20 Oct 2026): the 11 blocks of the
REM zcam34_consensus group of sites.json (chal_rocks, rochette, issole_seitah_south, threeforks_north,
REM enchanted_lake, rockytop, belva_crater, bunsen_peak, south_arm, taylorfjellet_large, marble_mountain), each
REM aligned as Navcam + Mastcam-Z 34 mm with the Navcam models in use - the tied joint navcal_tied_20261020_2,
REM promoted 20 Oct (src\mppp\data\cmods\CHANGES.md) - and the Mastcam-Z 34 mm start model in use; then
REM zcam_calibration.py calibrate over the finished blocks (zcam-v2: p1, p2 refined, with the no-tangential twin,
REM the rig's sol / temperature trends, baseline vs CAHV), promoted when every check passes.
REM
REM   run_z34.bat            the blocks (finished ones with the same run key are skipped; the new Navcam models
REM                          change the key, so every block runs once), then the calibration
REM   run_z34.bat --force    every block again
REM
REM Log of the runs: %MPPP_SCAPES%\run_sites_log.txt; the calibration: %MPPP_SCAPES%\camera_analysis\zcam34_calibration_<date>\report.md
REM Stop with stop_mppp.bat. Expect about 1-1.5 h per block (exhaustive SIFT matching), ~12-16 h in all.
REM ======================================================================================================
setlocal
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
set "ROOT=%MPPP_SCAPES%"
echo scapes root: %ROOT%
python -m mppp waypoints --refresh
python -m mppp cmods list
echo.
echo %DATE% %TIME%  Mastcam-Z 34 mm blocks: zcam34_consensus
python "%MPPP_HOME%\scripts\run_sites.py" --root "%ROOT%" --group zcam34_consensus --zcam --set ZCAM_ZOOMS=[34] %*
if errorlevel 1 echo some blocks did not finish - the calibration grades what is there
echo.
echo %DATE% %TIME%  Mastcam-Z 34 mm calibration (promoted only if every check passes)
python "%MPPP_HOME%\studies\calibration\zcam_calibration.py" calibrate --root "%ROOT%" --groups zcam34_consensus --n 0 --promote
if errorlevel 1 (echo calibration NOT promoted - see %ROOT%\camera_analysis\zcam34_calibration_*\report.md) else (echo calibration PROMOTED to src\mppp\data\cmods - run claude_sync.bat to push the commit)
echo.
echo %DATE% %TIME%  done
pause

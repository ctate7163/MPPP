@echo off
REM ======================================================================================================
REM MPPP 0.90 - the Mastcam-Z 34 mm run and calibration (Christian, 20 Oct 2026): the zcam34_consensus group of
REM sites.json (chal_rocks, rochette, south_arm, marble_mountain), each block aligned as Navcam + Mastcam-Z 34 mm
REM with the Navcam models in use (src\mppp\data\cmods: the 20 Oct pass-2 consensus) and the Mastcam-Z 34 mm start
REM model in use (the 0.87 South Arm model), then zcam_calibration.py calibrate over the finished blocks, promoted
REM when every check passes (at least 3 graded blocks; one of the four may fail its grade).
REM
REM   run_z34.bat            the four blocks (finished ones with the same settings are skipped), then the calibration
REM   run_z34.bat --force    every block again
REM
REM Log of the runs: %MPPP_SCAPES%\run_sites_log.txt; the calibration: %MPPP_SCAPES%\camera_analysis\zcam34_calibration_<date>\report.md
REM Stop with stop_mppp.bat. Expect about 1-1.5 h per block (SIFT), the calibration itself a few minutes.
REM ======================================================================================================
setlocal
if not defined MPPP_HOME for %%I in ("%~dp0..\..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES (echo set MPPP_SCAPES to the scapes root, e.g. setx MPPP_SCAPES D:\scapes & pause & exit /b 1)
set "ROOT=%MPPP_SCAPES%"
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

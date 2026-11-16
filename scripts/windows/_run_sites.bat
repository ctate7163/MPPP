@echo off
REM RPP: the minimised window of run_sites.bat (environment already set up by it).
title RPP run_sites
python "%RPP_HOME%\scripts\run_sites.py" %*
set "RC=%ERRORLEVEL%"
REM 31 Oct: stop_runs (stop_rpp.bat) writes mppp_stopped.txt in the scapes root - then no RPP_THEN
set "RPP_STOPROOT=%ROOT%"
if not defined RPP_STOPROOT set "RPP_STOPROOT=%RPP_SCAPES%"
if defined RPP_THEN if exist "%RPP_STOPROOT%\mppp_stopped.txt" (
  echo run_sites was stopped by stop_runs: %RPP_THEN% not started
  set "RPP_THEN="
)
REM 20 Oct: RPP_THEN (set by run_sites_zcam.bat: zcam_joints.bat) runs after the batch when every block
REM finished or some failed (exit 0 / 1); not when the batch did not start (2 unknown sites, 3 another batch, 4 forms)
if defined RPP_THEN if %RC% LEQ 1 (
  echo.
  echo %DATE% %TIME%  then: %RPP_THEN%
  call "%RPP_HOME%\scripts\windows\%RPP_THEN%" %RPP_THEN_ARGS%
  exit /b
)
if %RC% GEQ 1 (
  echo.
  echo run_sites stopped with errors - see run_sites_log.txt and the sites' runs\*\log.txt
  pause
)

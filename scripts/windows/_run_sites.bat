@echo off
REM MPPP: the minimised window of run_sites.bat (environment already set up by it).
title MPPP run_sites
python "%MPPP_HOME%\scripts\run_sites.py" %*
set "RC=%ERRORLEVEL%"
REM 31 Oct: stop_runs (stop_mppp.bat) writes mppp_stopped.txt in the scapes root - then no MPPP_THEN
set "MPPP_STOPROOT=%ROOT%"
if not defined MPPP_STOPROOT set "MPPP_STOPROOT=%MPPP_SCAPES%"
if defined MPPP_THEN if exist "%MPPP_STOPROOT%\mppp_stopped.txt" (
  echo run_sites was stopped by stop_runs: %MPPP_THEN% not started
  set "MPPP_THEN="
)
REM 20 Oct: MPPP_THEN (set by run_sites_zcam.bat: zcam_joints.bat) runs after the batch when every block
REM finished or some failed (exit 0 / 1); not when the batch did not start (2 unknown sites, 3 another batch, 4 forms)
if defined MPPP_THEN if %RC% LEQ 1 (
  echo.
  echo %DATE% %TIME%  then: %MPPP_THEN%
  call "%MPPP_HOME%\scripts\windows\%MPPP_THEN%" %MPPP_THEN_ARGS%
  exit /b
)
if %RC% GEQ 1 (
  echo.
  echo run_sites stopped with errors - see run_sites_log.txt and the sites' runs\*\log.txt
  pause
)

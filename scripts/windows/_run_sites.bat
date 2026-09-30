@echo off
REM MPPP v0p43: the minimised window of run_all_sites.bat (environment already set up by it).
title MPPP run_sites
python "%MPPP_HOME%\scripts\run_sites.py" %*
if errorlevel 1 (
  echo.
  echo run_sites stopped with errors - see run_sites_log.txt and the sites' runs\*\log.txt
  pause
)

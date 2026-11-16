@echo off
REM RPP: the minimised window of process_sites.bat (environment already set up by it).
title RPP process_sites
python "%RPP_HOME%\scripts\process_sites.py" %*
if errorlevel 1 (
  echo.
  echo process_sites stopped with errors - see process_sites_log.txt and the sites' runs\*_process\log.txt
  pause
)

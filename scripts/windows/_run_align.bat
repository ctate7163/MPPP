@echo off
REM MPPP: the minimised window of align_here.bat (environment already set up by it).
title MPPP align - %~1
python "%MPPP_HOME%\scripts\align_scape.py" %*
if errorlevel 1 (
  echo.
  echo The alignment stopped with an error - see the newest runs\*\log.txt in %~1
  pause
)

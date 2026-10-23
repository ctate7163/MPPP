@echo off
REM ======================================================================================================
REM MPPP 0.93.8 - promote_models.bat: make camera model files the models in use (mppp cmods promote).
REM Drag model files or a candidate folder onto this file, or:
REM   promote_models.bat D:\scapes\camera_analysis\hazcam_fit_zcam_20261023\M2020_R_rig.json  [more files]
REM Each file is checked against its form first; CHANGES.md in the cmods folder records the promotion.
REM ======================================================================================================
setlocal
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if "%~1"=="" (
  echo Drag model files or a candidate folder onto promote_models.bat
  pause
  exit /b 1
)
python -m mppp cmods promote %* --note "promote_models.bat"
python -m mppp cmods list
pause
exit /b 0

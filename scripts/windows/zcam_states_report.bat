@echo off
REM MPPP 0.94.6: after run_zcam_states.bat - the Mastcam-Z state of every image of the "states" blocks and the
REM proposal for the found list (%MPPP_SCAPES%\camera_analysis\zcam_states_<date>\report.md).  Arguments go to
REM "mppp zcam-states" (e.g. --apply to write the proposal into the found list, a dated backup first).
setlocal
if not defined MPPP_HOME for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
python -m mppp zcam-states --root "%MPPP_SCAPES%" --variant states %*
if not defined MPPP_NOPAUSE_END pause

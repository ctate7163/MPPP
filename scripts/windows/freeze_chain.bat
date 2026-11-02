@echo off
REM ======================================================================================================
REM MPPP 0.95.3 - freeze_chain.bat (2 Nov 2026, Christian: "I will be away from the computer when these finish")
REM Unattended: waits for the mask training, runs the mask A/B (v3, v4, v4.1), the point-thinning study S2 on
REM rockytop_example, then the freeze run (run_pipeline --reset; the S2 cap when it passes), then
REM D:\scapes\paper_studies\run_paper_studies.py if it is there.  See scripts\freeze_chain.py.
REM   freeze_chain.bat                  everything
REM   freeze_chain.bat --skip-s2        no S2 (no point cap)
REM   freeze_chain.bat --no-freeze      the A/B and S2 only
REM Log: D:\scapes\mppp_pipeline\freeze_chain_log.txt (and the run_pipeline logs next to it).
REM Keep the computer awake and the Claude app open (the results are read from afar).
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
set "KMP_DUPLICATE_LIB_OK=TRUE"
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
set "TPY="
for %%P in ("%MPPP_TRAIN_PYTHON%" "python" "%GPU_PYTHON%") do if not defined TPY if not "%%~P"=="" (
  "%%~P" -c "import sys, torch, timm, mppp; sys.exit(0 if torch.cuda.is_available() else 3)" >nul 2>nul && set "TPY=%%~P"
)
if not defined TPY set "TPY=python"
echo %DATE% %TIME%  freeze chain: mask A/B with %TPY%; log %MPPP_SCAPES%\mppp_pipeline\freeze_chain_log.txt
python "%MPPP_HOME%\scripts\freeze_chain.py" --root "%MPPP_SCAPES%" --ab-python "%TPY%" %*
echo %DATE% %TIME%  freeze chain finished (exit %ERRORLEVEL%)
pause

@echo off
REM ======================================================================================================
REM MPPP 0.95.3 - freeze_chain.bat (2 Nov 2026, Christian: "I will be away from the computer when these finish";
REM 14:52 "The blocks should be saved to D:\rover_blocks"; 14:54 "don't add the mask training to the freeze,
REM just delay the start so that it will be done by the time it runs the mask comparison")
REM Unattended: the new root (the found lists copied from D:\scapes; DEMs in D:\rover_blocks\dem), waits for the
REM mask v4.2 training started by hand (train_masks.bat --tag v8_v4.2 --zoom-pairs), the mask A/B (v3, v4.1,
REM v4.2) - v4.2 is used when it beats v3 -, the freeze run (run_pipeline --reset, no point cap) in D:\rover_blocks,
REM D:\rover_blocks\paper_studies\run_paper_studies.py if it is there, and last (14:57) the point-cap study S2 on
REM rockytop_example in D:\scapes, where its block is.  See scripts\freeze_chain.py.
REM   freeze_chain.bat                  everything
REM   freeze_chain.bat --no-wait        no wait for the v4.2 training
REM   freeze_chain.bat --no-freeze      the A/B and S2 only
REM   freeze_chain.bat --skip-s2        no S2 at the end
REM Root: MPPP_FREEZE_ROOT if set, else D:\rover_blocks.
REM Log: D:\rover_blocks\mppp_pipeline\freeze_chain_log.txt (and the run_pipeline logs next to it).
REM Keep the computer awake and the Claude app open (the results are read from afar).
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
set "KMP_DUPLICATE_LIB_OK=TRUE"
if not defined MPPP_FREEZE_ROOT set "MPPP_FREEZE_ROOT=D:\rover_blocks"
set "MPPP_SCAPES=%MPPP_FREEZE_ROOT%"
call "%MPPP_HOME%\scripts\windows\m20_dem_env.bat"
set "TPY="
for %%P in ("%MPPP_TRAIN_PYTHON%" "python" "%GPU_PYTHON%") do if not defined TPY if not "%%~P"=="" (
  "%%~P" -c "import sys, torch, timm, mppp; sys.exit(0 if torch.cuda.is_available() else 3)" >nul 2>nul && set "TPY=%%~P"
)
if not defined TPY set "TPY=python"
echo %DATE% %TIME%  freeze chain: root %MPPP_SCAPES%, mask A/B with %TPY%; log %MPPP_SCAPES%\mppp_pipeline\freeze_chain_log.txt
python "%MPPP_HOME%\scripts\freeze_chain.py" --root "%MPPP_SCAPES%" --old-root "D:\scapes" --ab-python "%TPY%" %*
echo %DATE% %TIME%  freeze chain finished (exit %ERRORLEVEL%)
pause

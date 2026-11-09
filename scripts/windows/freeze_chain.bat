@echo off
REM ======================================================================================================
REM MPPP 0.95.4 - freeze_chain.bat (2 Nov 2026, Christian: "I will be away from the computer when these finish";
REM 15:52: "make sure it is robust against faults. I want it to run to completion over several days, at the end of
REM which I want a standard set of blocks using the best camera and rig models available followed by the time
REM consuming tests and studies needed to really understand the results").  See mppp_studies.calibration.freeze_chain.
REM Unattended, in D:\rover_blocks: the root prepared (the found lists from D:\scapes, the DEMs in dem\), a wait
REM for the mask v4.2 training started by hand (train_masks.bat --tag v8_v4.2 --zoom-pairs), the mask A/B (v3,
REM v4.1, v4.2; v4.2 used when it beats v3), then run_pipeline in three calls: the blocks and the joints (the
REM Navcam, Hazcam and Mastcam-Z models promoted when their checks pass), the final pass (every block whose
REM models changed aligned again with the promoted ones), the studies (error analysis, rover frame, LightGlue
REM A/B, Mastcam-Z states, the report); then the studies that exist (make_figures.bat, make_maps.bat, the Mastcam-Z
REM aspect geometry, S6) on those blocks, and last the tie-point studies on the tiny blocks in D:\scapes: S2 (the
REM point cap) and S1, S3, S4, S5 (finished variants skipped).
REM Faults: a failed step (its exit code, or a bug in the chain's code for it) is logged and the chain goes on; a
REM step past its time limit is stopped with its process tree; the computer is kept awake and QuickEdit is off; the
REM free disk space is logged before every step.  After a crash, a reboot or a closed window: double-click this
REM file again - it continues where it was (the mask choice kept, the pipeline after its last finished stage, the
REM finished blocks skipped; a pipeline left running by a closed window is waited for).  After a reboot the chain
REM starts itself again at your next logon (a scheduled task "MPPP freeze chain", removed when the chain ends).
REM A second window on the same root does not start.  The status of an older chain is kept aside, not used.
REM   freeze_chain.bat                  everything (again after a stop: continues)
REM   freeze_chain.bat --status         where it is (the chain's steps, the pipeline's stage); nothing run
REM   freeze_chain.bat --no-wait        no wait for the v4.2 training
REM   freeze_chain.bat --no-final-pass  the blocks as the joints found them
REM   freeze_chain.bat --skip-s2        no S2 at the end;  --skip-tie-studies  no S1 / S3 / S4 / S5
REM   freeze_chain.bat --no-logon-task  no scheduled task after a reboot
REM   freeze_chain.bat --redo s2        a step again (root wait_v42 mask_ab blocks final_pass studies figures maps
REM                                     aspect_geometry s6 s2 s1 s3 s4 s5)
REM Root: MPPP_FREEZE_ROOT if set, else D:\rover_blocks.
REM Log: D:\rover_blocks\mppp_pipeline\freeze_chain_log.txt, status freeze_chain_status.json, each step's output
REM freeze_chain_<step>.log, the pipeline's own logs next to them.  Keep the Claude app open (read from afar).
REM Before leaving: Settings > Windows Update > "Pause updates" for the weeks it runs (a restart for an update is
REM the one fault the chain cannot ride out by itself - it needs your logon afterwards).
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
if not defined MPPP_FREEZE_ROOT set "MPPP_FREEZE_ROOT=D:\rover_blocks"
title MPPP freeze chain - %MPPP_FREEZE_ROOT%
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
set "_FC_LOG=%MPPP_FREEZE_ROOT%\mppp_pipeline\freeze_chain_log.txt"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat"
if errorlevel 1 (
  echo %DATE% %TIME%  freeze chain: no Python with mppp found - not started>>"%_FC_LOG%" 2>nul
  if not defined MPPP_NOPAUSE pause
  exit /b 1
)
set "KMP_DUPLICATE_LIB_OK=TRUE"
set "MPPP_SCAPES=%MPPP_FREEZE_ROOT%"
call "%MPPP_HOME%\scripts\windows\m20_dem_env.bat"
if /i "%~1"=="--status" (
  python -m mppp_studies.calibration.freeze_chain --root "%MPPP_SCAPES%" --status
  if not defined MPPP_NOPAUSE pause
  exit /b 0
)
set "TPY="
for %%P in ("%MPPP_TRAIN_PYTHON%" "python" "%GPU_PYTHON%") do if not defined TPY if not "%%~P"=="" (
  "%%~P" -c "import sys, torch, timm, mppp; sys.exit(0 if torch.cuda.is_available() else 3)" >nul 2>nul && set "TPY=%%~P"
)
if not defined TPY set "TPY=python"
echo %DATE% %TIME%  freeze chain: root %MPPP_SCAPES%, mask A/B with %TPY%; log %_FC_LOG%
python -m mppp_studies.calibration.freeze_chain --root "%MPPP_SCAPES%" --old-root "D:\scapes" --ab-python "%TPY%" %*
set "_FC_RC=%ERRORLEVEL%"
echo %DATE% %TIME%  freeze chain finished (exit %_FC_RC%)
if not defined MPPP_NOPAUSE pause
exit /b %_FC_RC%

@echo off
REM ======================================================================================================
REM RPP - ecam_block.bat (0.95.2, Christian, 30 Oct 2026): one site's Navcam + Hazcam block from the PDS
REM archive (processing and the COLMAP alignment), then its Metashape job and its Nerfstudio export.
REM
REM   ecam_block.bat SITE [more rpp run options]        e.g.   ecam_block.bat hippo_pools
REM
REM Writes %RPP_SCAPES%\mars2020_sol_<first sol>_<site>\:
REM   colmap\                 the alignment (open_in_colmap.bat, health\, error_input\)
REM   colmap\metashape\       make_metashape_project.bat makes <block>.psx (Metashape Professional)
REM   colmap\nerfstudio\      images\, masks\ (2048 x 1536 pinhole), transforms.json, sparse_pc.ply
REM                           ns-train splatfacto --data <that folder> nerfstudio-data
REM set RPP_METASHAPE_RUN=1 builds the .psx at the end too. Runs in this window; the log is
REM %RPP_SCAPES%\ecam_block_<site>_log.txt. Another batch may run at the same time (other folders).
REM ======================================================================================================
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "RPP_HOME=%%~fI"
set "RPP_WIN=%RPP_HOME%\scripts\windows"
call "%RPP_WIN%\rpp_env.bat" || ( pause & exit /b 1 )
if not defined RPP_SCAPES set "RPP_SCAPES=D:\rover_blocks"
if "%~1"=="" (
  echo usage: ecam_block.bat SITE   ^(a site of rpp\data\sites.json, e.g. hippo_pools^)
  pause
  exit /b 2
)
echo %DATE% %TIME%  ecam block %~1: process, align, Metashape job, Nerfstudio export
python -m rpp --log "%RPP_SCAPES%\ecam_block_%~1_log.txt" run %* --root "%RPP_SCAPES%" --gui --metashape --nerfstudio
set "RC=%ERRORLEVEL%"
echo %DATE% %TIME%  ecam block %~1 done ^(exit %RC%; 0 ok, 1 warnings^) - see %RPP_SCAPES%\ecam_block_%~1_log.txt
if not defined RPP_NOPAUSE pause
exit /b %RC%

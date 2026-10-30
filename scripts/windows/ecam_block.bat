@echo off
REM ======================================================================================================
REM MPPP - ecam_block.bat (0.95.2, Christian, 30 Oct 2026): one site's Navcam + Hazcam block from the PDS
REM archive (processing and the COLMAP alignment), then its Metashape job and its Nerfstudio export.
REM
REM   ecam_block.bat SITE [more mppp run options]        e.g.   ecam_block.bat hippo_pools
REM
REM Writes %MPPP_SCAPES%\mars2020_sol_<first sol>_<site>\:
REM   colmap\                 the alignment (open_in_colmap.bat, health\, error_input\)
REM   colmap\metashape\       make_metashape_project.bat makes <block>.psx (Metashape Professional)
REM   colmap\nerfstudio\      images\, masks\ (2048 x 1536 pinhole), transforms.json, sparse_pc.ply
REM                           ns-train splatfacto --data <that folder> nerfstudio-data
REM set MPPP_METASHAPE_RUN=1 builds the .psx at the end too. Runs in this window; the log is
REM %MPPP_SCAPES%\ecam_block_<site>_log.txt. Another batch may run at the same time (other folders).
REM ======================================================================================================
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
set "MPPP_WIN=%MPPP_HOME%\scripts\windows"
call "%MPPP_WIN%\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
if "%~1"=="" (
  echo usage: ecam_block.bat SITE   ^(a site of mppp\data\sites.json, e.g. hippo_pools^)
  pause
  exit /b 2
)
echo %DATE% %TIME%  ecam block %~1: process, align, Metashape job, Nerfstudio export
python -m mppp --log "%MPPP_SCAPES%\ecam_block_%~1_log.txt" run %* --root "%MPPP_SCAPES%" --gui --metashape --nerfstudio
set "RC=%ERRORLEVEL%"
echo %DATE% %TIME%  ecam block %~1 done ^(exit %RC%; 0 ok, 1 warnings^) - see %MPPP_SCAPES%\ecam_block_%~1_log.txt
if not defined MPPP_NOPAUSE pause
exit /b %RC%

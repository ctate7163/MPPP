@echo off
REM ======================================================================================================
REM MPPP 0.93.9 - test_lightglue.bat (Christian, 23 Oct 2026): the LightGlue A/B test on Chal Rocks sols 91-98
REM (site chal_rocks_tiny: Chal Rocks tiny, Navcam + Mastcam-Z + Hazcams): the block processed once, aligned twice - COLMAP's SIFT
REM matcher into colmap_sift, SIFT + LightGlue into colmap_lightglue - then compared (matcher that ran, pairs,
REM inlier matches, registered images, cross-station points, residuals, every health check, matching and total
REM time) in <WORK>\lightglue_ab.md.  First a check that torch and kornia import and see the GPU.
REM
REM   test_lightglue.bat                       both alignments, then the comparison
REM   test_lightglue.bat --only lightglue      one of them
REM   test_lightglue.bat --compare-only        the comparison again
REM   test_lightglue.bat --site rockytop       another site (its Navcam + Mastcam-Z block)
REM ======================================================================================================
setlocal
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
python "%MPPP_HOME%\scripts\lightglue_ab.py" --root "%MPPP_SCAPES%" %*
pause
exit /b 0

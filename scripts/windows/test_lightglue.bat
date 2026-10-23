@echo off
REM ======================================================================================================
REM MPPP 0.94.1 - test_lightglue.bat (Christian, 23 Oct 2026): the LightGlue A/B test on the two tiny blocks -
REM Chal Rocks tiny (sols 91-98, without S004D0218 and ZCAM08390) and Seitah Overlook tiny (sols 116-121), Navcam +
REM Mastcam-Z + Hazcams: each processed once, aligned twice - COLMAP's SIFT matcher into colmap_sift, SIFT +
REM LightGlue into colmap_lightglue - and compared in <WORK>\lightglue_ab.md (matcher that ran, pairs, inlier
REM matches, registered images, cross-station points, residuals, every health check, matching and total time).
REM Then the tiny joint over their colmap_lightglue blocks: the Hazcam fit and the Mastcam-Z joint of every zoom,
REM promoted when every check passes, with as few blocks as there are - only models of which none is in use
REM (a tiny joint does not replace a model made from more blocks).  First a check that torch and kornia see the GPU.
REM
REM   test_lightglue.bat                       both sites, both alignments, the comparisons, the tiny joint
REM   test_lightglue.bat --only lightglue      one alignment per site
REM   test_lightglue.bat --compare-only        the comparisons again (and the tiny joint)
REM   test_lightglue.bat --joint-only          only the tiny joint
REM   test_lightglue.bat --no-joint            no tiny joint;   --no-promote   the tiny joint promotes nothing
REM   test_lightglue.bat --site rockytop       other sites (their Navcam + Mastcam-Z blocks)
REM ======================================================================================================
setlocal
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
python "%MPPP_HOME%\scripts\lightglue_ab.py" --root "%MPPP_SCAPES%" %*
pause
exit /b 0

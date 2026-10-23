@echo off
REM ======================================================================================================
REM MPPP 0.94.4 - test_lightglue.bat (Christian, 23 Oct 2026): the matcher A/B test on the two tiny blocks -
REM Chal Rocks tiny (sols 91-98, without S004D0218 and ZCAM08390) and Seitah Overlook tiny (sols 116-121), Navcam +
REM Mastcam-Z + Hazcams: each processed once, aligned with four matchers -
REM   colmap_sift               COLMAP's SIFT matcher
REM   colmap_lightglue          LightGlue on the 8192 largest-scale keypoints of each image
REM   colmap_lightglue_octaves  LightGlue on 8192 keypoints spread over every octave of scale (0.94.4)
REM   colmap_sift_lightglue     SIFT, then LightGlue's inliers added to SIFT's (0.94.4)
REM - a variant already finished is not run again (--rerun), and compared in <WORK>\lightglue_ab.md (Navcam -
REM Mastcam-Z tie points, cross-station points, residuals, points off the terrain, every health check, times).
REM Then the tiny joint over the variant with the most Navcam - Mastcam-Z tie points (--joint-variant to choose):
REM the Hazcam fit and the Mastcam-Z joint of every zoom, promoted when every check passes, with as few blocks as
REM there are - only models of which none is in use.  First a check that torch and kornia see the GPU.
REM
REM   test_lightglue.bat                                   both sites, all four variants, the comparisons, the joint
REM   test_lightglue.bat --variants sift sift_lightglue    some variants only
REM   test_lightglue.bat --rerun                           run finished variants again
REM   test_lightglue.bat --compare-only                    the comparisons again (and the tiny joint)
REM   test_lightglue.bat --joint-only                      only the tiny joint
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

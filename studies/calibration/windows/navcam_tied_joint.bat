@echo off
REM ======================================================================================================
REM MPPP 0.91 - the Navcam joint as ONE bundle adjustment, once, on the finished Navcam blocks under D:\scapes
REM (Christian, 20 Oct 2026: run it on the 0.90 blocks before the full 0.91 re-run; no promotion).
REM
REM   mppp calib navcam: every term in one adjustment (offsets at -20 degC + slopes, per-eye early offsets,
REM   rig at sol 0 with linear yaw/roll rates and the pitch hinge at sol 250), 8000 points per block, the
REM   covariance, then the diagnostic pass (per block / bin / epoch about the solution), the knot comparison
REM   250 / 300 / 350 / free and the roll profile (linear vs a step at sol 900) on its epochs, the promotion check
REM   (reported, nothing promoted), then the refined figures from it.
REM
REM   Output: D:\scapes\camera_analysis\navcal_tied_<yyyymmdd>\ (navcam_joint\ = the candidate files, tied_joint.json,
REM           consensus.json, rig_residuals.json, thermal_bins.json, promotion_check.json)
REM           D:\code\MPPP\Claude outputs\v0p91\navcam_*_tied_<yyyymmdd>.png / .md
REM   Arguments are passed through (e.g. navcam_tied_joint.bat --knot 300, --points 3000, --promote-if-ok).
REM   Time: about an hour for 16-23 blocks (CPU only; can run beside a Mastcam-Z run).
REM ======================================================================================================
setlocal
if not defined MPPP_HOME for %%I in ("%~dp0..\..\..") do set "MPPP_HOME=%%~fI"
call "%MPPP_HOME%\scripts\windows\mppp_env.bat" || ( pause & exit /b 1 )
if not defined MPPP_SCAPES set "MPPP_SCAPES=D:\scapes"
for /f %%d in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd"') do set "TODAY=%%d"
set "OUT=%MPPP_SCAPES%\camera_analysis\navcal_tied_%TODAY%"
set "FIG=%MPPP_HOME%\Claude outputs\v0p91"
if not exist "%FIG%" mkdir "%FIG%"
echo tied joint: blocks under %MPPP_SCAPES% (group navcam_consensus) -^> %OUT%
python -m mppp --log "%OUT%_log.txt" calib navcam --root "%MPPP_SCAPES%" --out "%OUT%" --knot 250 --points 8000 %*
if errorlevel 1 ( echo tied joint FAILED (see %OUT%_log.txt) & pause & exit /b 1 )
echo figures -^> %FIG%
python "%MPPP_HOME%\scripts\paper_figures.py" --root "%MPPP_SCAPES%" --study "%OUT%" --out "%FIG%" --tag tied_%TODAY%
echo done: %OUT%  and  %FIG%
pause
exit /b 0

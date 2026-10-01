@echo off
REM ======================================================================================================
REM MPPP - bring GitHub and this git working copy together, then push (v0p50.1; v0p61: stash, K for shared history).
REM
REM Fetches github.com/<you>/MPPP first ("pull before push"):
REM   - GitHub has nothing new             -> push.
REM   - GitHub has commits this copy lacks -> (v0p61) K keep them as a branch and push this copy as main
REM                                           (recommended), M merge them in (this copy wins), Q quit.
REM   - GitHub has an unrelated history    -> (e.g. an earlier upload of MPPP) you choose:
REM       K  keep it as the branch github-before-v0p50 on GitHub and make main this history   (recommended)
REM       M  merge it into this history (this folder's files win where both changed; its other files are added)
REM       Q  quit, nothing changed
REM Afterwards _transfer\mppp_pc.bundle holds this history for Claude's next session.
REM ======================================================================================================
setlocal EnableExtensions
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
cd /d "%MPPP_HOME%" || (pause & exit /b 1)
if not exist ".git" (echo not a git working copy yet: run setup_github.bat first & pause & exit /b 1)
git remote get-url origin >nul 2>nul || (echo no GitHub remote yet: run setup_github.bat first & pause & exit /b 1)

REM v0p61: a delivery from Claude that this copy has not taken yet: sync_from_claude.bat first (it calls this file)
if exist "_transfer\mppp_latest.bundle" (
  git fetch -q "_transfer\mppp_latest.bundle" "+refs/heads/main:refs/remotes/claude/main" 2>nul
  git merge-base --is-ancestor claude/main HEAD 2>nul
  if errorlevel 1 (
    echo _transfer\mppp_latest.bundle has a newer delivery from Claude than this copy: run sync_from_claude.bat
    echo ^(it takes the delivery and then runs this push^). Nothing changed.
    pause & exit /b 1
  )
)
REM v0p61: files changed here but not committed (e.g. a delivery copied in before sync_from_claude.bat ran) would
REM block any merge ("Your local changes ... would be overwritten"): put them aside in a stash first, with untracked ones
git diff --quiet HEAD -- 2>nul && git diff --cached --quiet 2>nul
if errorlevel 1 goto :stash
for /f %%U in ('git ls-files --others --exclude-standard ^| find /c /v ""') do if not "%%U"=="0" goto :stash
goto :fetch
:stash
echo This copy has changes that are not committed - putting them aside (git stash list shows them):
git status --short
git stash push -u -q -m "github_push %DATE% %TIME%: uncommitted changes" || (echo could not stash - nothing changed & pause & exit /b 1)

:fetch
echo fetching GitHub ...
git fetch origin || (echo cannot reach GitHub - check that you are signed in & pause & exit /b 1)
git rev-parse -q --verify origin/main >nul || goto :push
git merge-base --is-ancestor origin/main HEAD && goto :push

echo.
echo GitHub main has commits that this copy does not have:
git log --oneline -8 origin/main
echo.
git merge-base HEAD origin/main >nul 2>nul
if errorlevel 1 goto :unrelated

REM v0p61: usually these are commits Claude rewrote later (e.g. 0.50.0-0.51.0 before the 0.51.1 rewrite) or an older
REM delivery: this copy (after sync_from_claude.bat) is the current one.
echo They share history with this copy, but this copy does not contain them (e.g. Claude's older, rewritten
echo commits that were pushed before). Choose:
echo   K  keep GitHub's main as a branch github-before-^<date-time^> and make main this copy (recommended)
echo   M  merge them into this copy (this copy's version wins where both changed)
echo   Q  quit, nothing changed
choice /C KMQ /M "K, M or Q"
if errorlevel 3 (echo nothing changed & pause & exit /b 1)
if errorlevel 2 goto :merge_shared
for /f %%D in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd-HHmm"') do set "STAMP=%%D"
git push origin "refs/remotes/origin/main:refs/heads/github-before-%STAMP%" || (pause & exit /b 1)
echo GitHub's previous main is kept as the branch github-before-%STAMP%
git push --force-with-lease=main:origin/main -u origin main --tags || (pause & exit /b 1)
goto :done

:merge_shared
git merge --no-edit -X ours origin/main
if errorlevel 1 (
  git merge --abort 2>nul
  echo The merge failed - nothing changed. Run this again and choose K, or tell Claude.
  pause & exit /b 1
)
goto :push

:unrelated
echo This is a different history (not made from this one), e.g. an earlier upload of MPPP.
echo Files on GitHub:
git ls-tree --name-only origin/main
echo.
choice /C KMQ /M "K = keep it as branch github-before-v0p50 and replace main (recommended), M = merge, Q = quit"
if errorlevel 3 (echo nothing changed & pause & exit /b 1)
if errorlevel 2 goto :merge_unrelated
git push origin "refs/remotes/origin/main:refs/heads/github-before-v0p50" || (pause & exit /b 1)
echo the old GitHub main is kept as the branch github-before-v0p50
git push --force-with-lease=main:origin/main -u origin main --tags || (pause & exit /b 1)
goto :done

:merge_unrelated
git merge --no-edit --allow-unrelated-histories -X ours origin/main -m "Merge the earlier GitHub history of MPPP" || (git merge --abort & echo merge failed - nothing changed & pause & exit /b 1)

:push
git push -u origin main --tags || (echo push failed & pause & exit /b 1)

:done
if not exist "_transfer" mkdir "_transfer"
git bundle create -q "_transfer\mppp_pc.bundle" --all && echo history for Claude: _transfer\mppp_pc.bundle
echo.
git log --oneline -3
echo GitHub is up to date.
pause

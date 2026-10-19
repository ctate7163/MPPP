@echo off
REM ======================================================================================================
REM MPPP v0p86 - render_paper.bat (Christian, 19 Oct 2026): paper\paper.md (the JOSS paper) -> paper\paper.pdf
REM
REM claude_sync.bat calls it (render_paper.bat --quiet) after every sync, so paper.pdf always matches the code that
REM was just synced; nothing else renders it.  Run it by hand to render without syncing.
REM
REM pandoc paper\paper.md --citeproc --bibliography paper\paper.bib --resource-path paper -o paper\paper.pdf
REM with the first LaTeX engine found: xelatex (Unicode: theta, epsilon), lualatex, pdflatex, tectonic.
REM Needs pandoc (winget install JohnMacFarlane.Pandoc) and a LaTeX (MiKTeX: winget install MiKTeX.MiKTeX).
REM Log: _transfer\render_paper_log.txt.  paper.pdf is not in git (.gitignore).
REM ======================================================================================================
setlocal EnableExtensions EnableDelayedExpansion
for %%I in ("%~dp0..\..") do set "MPPP_HOME=%%~fI"
set "PAPER=%MPPP_HOME%\paper"
set "LOG=%MPPP_HOME%\_transfer\render_paper_log.txt"
set "QUIET="
if /i "%~1"=="--quiet" set "QUIET=1"

set "VER="
for /f "tokens=2 delims==" %%v in ('findstr /b /c:"__version__" "%MPPP_HOME%\src\mppp\__init__.py"') do set "VER=%%~v"
set "VER=%VER: =%"
set "VER=%VER:"=%"
if not exist "%PAPER%\paper.md" (
  echo render_paper: no %PAPER%\paper.md - nothing to render
  goto :done
)

set "PANDOC="
for /f "delims=" %%p in ('where pandoc 2^>nul') do if not defined PANDOC set "PANDOC=%%p"
for %%p in ("%LOCALAPPDATA%\Pandoc\pandoc.exe" "%ProgramFiles%\Pandoc\pandoc.exe") do if not defined PANDOC if exist "%%~p" set "PANDOC=%%~p"
if not defined PANDOC (
  echo render_paper: pandoc not found - install it with   winget install JohnMacFarlane.Pandoc
  goto :done
)
set "ENGINE="
for %%e in (xelatex lualatex pdflatex tectonic) do if not defined ENGINE (where %%e >nul 2>nul && set "ENGINE=%%e")
if not defined ENGINE for %%e in ("%LOCALAPPDATA%\Programs\MiKTeX\miktex\bin\x64\xelatex.exe" "%ProgramFiles%\MiKTeX\miktex\bin\x64\xelatex.exe") do if not defined ENGINE if exist "%%~e" set "ENGINE=%%~e"
if not defined ENGINE (
  echo render_paper: no LaTeX engine found - install MiKTeX:   winget install MiKTeX.MiKTeX
  goto :done
)
set "BIB="
if exist "%PAPER%\paper.bib" set BIB=--citeproc --bibliography "%PAPER%\paper.bib"

echo render_paper: MPPP %VER%, %ENGINE% ...
echo ==== render_paper %DATE% %TIME% MPPP %VER% engine %ENGINE% > "%LOG%"
pushd "%PAPER%"
"%PANDOC%" "%PAPER%\paper.md" %BIB% --resource-path "%PAPER%" --pdf-engine "%ENGINE%" -o "%PAPER%\paper.pdf" >> "%LOG%" 2>&1
set "RC=%ERRORLEVEL%"
popd
if not "%RC%"=="0" (
  echo render_paper: FAILED ^(pandoc exit %RC%^), see %LOG%:
  type "%LOG%"
  goto :done
)
echo render_paper: %PAPER%\paper.pdf (MPPP %VER%)
echo done: paper.pdf >> "%LOG%"

:done
if not defined QUIET pause
exit /b 0

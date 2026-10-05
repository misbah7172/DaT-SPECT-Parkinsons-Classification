@echo off
REM Build script for IEEEtran conference paper (Windows)

cd /d "%~dp0"

echo Building IEEEtran conference paper...

REM First pass
pdflatex -interaction=nonstopmode -shell-escape main.tex

REM Bibliography
biber main

REM Second pass
pdflatex -interaction=nonstopmode -shell-escape main.tex

REM Third pass
pdflatex -interaction=nonstopmode -shell-escape main.tex

echo Build complete. Output: main.pdf
pause
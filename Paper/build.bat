@echo off
REM Build script for the research paper (Windows)
REM Requires: MiKTeX or TeX Live with latexmk

cd /d "%~dp0"

echo Building paper...

REM First pass: generate aux files
pdflatex -interaction=nonstopmode -shell-escape main.tex

REM Bibliography
biber main

REM Second pass: resolve citations
pdflatex -interaction=nonstopmode -shell-escape main.tex

REM Third pass: finalize cross-references
pdflatex -interaction=nonstopmode -shell-escape main.tex

echo Build complete. Output: main.pdf
pause
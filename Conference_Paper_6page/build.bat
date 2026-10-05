@echo off
cd /d "%~dp0"
pdflatex -interaction=nonstopmode -shell-escape main.tex
biber main
pdflatex -interaction=nonstopmode -shell-escape main.tex
pdflatex -interaction=nonstopmode -shell-escape main.tex
echo Build complete: main.pdf
pause
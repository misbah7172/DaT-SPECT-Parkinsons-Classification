#!/bin/bash
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -shell-escape main.tex
biber main
pdflatex -interaction=nonstopmode -shell-escape main.tex
pdflatex -interaction=nonstopmode -shell-escape main.tex
echo "Build complete: main.pdf"
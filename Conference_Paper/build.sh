#!/bin/bash
# Build script for IEEEtran conference paper

cd "$(dirname "$0")"

echo "Building IEEEtran conference paper..."

# First pass
pdflatex -interaction=nonstopmode -shell-escape main.tex

# Bibliography
biber main

# Second pass
pdflatex -interaction=nonstopmode -shell-escape main.tex

# Third pass
pdflatex -interaction=nonstopmode -shell-escape main.tex

echo "Build complete. Output: main.pdf"
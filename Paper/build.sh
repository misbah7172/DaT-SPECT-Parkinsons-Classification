#!/bin/bash
# Build script for the research paper
# Requires: texlive-full or miktex with latexmk

cd "$(dirname "$0")"

echo "Building paper..."

# First pass: generate aux files
pdflatex -interaction=nonstopmode -shell-escape main.tex

# Bibliography
biber main

# Second pass: resolve citations
pdflatex -interaction=nonstopmode -shell-escape main.tex

# Third pass: finalize cross-references
pdflatex -interaction=nonstopmode -shell-escape main.tex

echo "Build complete. Output: main.pdf"
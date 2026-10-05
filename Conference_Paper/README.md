# Conference Paper: IEEEtran Format

## Paper Title
**A Robust Blueprint for 3D Neuroimaging Classification Across Heterogeneous Scanners: Spatial Standardization, Leakage-Free Validation, and Probability Calibration**

## Structure
- `main.tex` - Main LaTeX file (IEEEtran conference format)
- `references.bib` - Bibliography (35 references)
- `sections/` - 17 section files
- `figures/` - 10 figures (PNG/PDF)
- `references.bib` - BibTeX bibliography (35 entries)

## Build Instructions

### Windows
```cmd
build.bat
```

### Linux/Mac
```bash
chmod +x build.sh
./build.sh
```

### Manual Build
```bash
pdflatex -interaction=nonstopmode -shell-escape main.tex
biber main
pdflatex -interaction=nonstopmode -shell-escape main.tex
pdflatex -interaction=nonstopmode -shell-escape main.tex
```

## Requirements
- TeX Live 2024+ or MiKTeX
- biber (for bibliography)
- Required packages: graphicx, amsmath, amssymb, booktabs, multirow, array, float, subcaption, makecell, colortbl, xcolor, algorithm, algorithmic, adjustbox, cite, url, balance

## Output
- `main.pdf` - Compiled paper (approx. 4-5 pages, two-column IEEE format)

## Figures (10 total)
1. `method_pipeline.pdf` - 4-stage pipeline diagram
2. `voxel_distribution.png` - Voxel size distribution
3. `preprocessing_before_after.png` - Before/after preprocessing
4. `voxel_size_details.png` - Voxel size details table
5. `architecture_comparison.png` - Model architecture comparison
6. `training_curves.png` - Training curves
7. `calibration.png` - Calibration curves
7. `fold_metrics.png` - Per-fold metrics
8. `scanner_performance.png` - Scanner tier performance
9. `version_comparison.png` - Version comparison
10. `voxel_size_details.png` - Detailed voxel sizes

## Sections (17 total)
1. `introduction.tex` - Introduction
2. `methodological_framework.tex` - 4-stage framework
3. `spatial_standardization.tex` - Data preprocessing
4. `leakage_prevention.tex` - Leakage-free CV
5. `model_architecture.tex` - 3D ResNet architecture
6. `biomarker_baseline.tex` - Biomarker comparison
7. `calibration.tex` - Platt calibration
8. `experiments.tex` - Experimental setup
9. `results.tex` - Main results
10. `ablation_study.tex` - Ablation studies
10. `guidelines.tex` - 5 actionable rules
11. `discussion.tex` - Discussion
12. `conclusion.tex` - Conclusion
13. `appendix.tex` - Hyperparameters, compute, reproducibility
14. `methodological_framework.tex` - Framework overview
15. `spatial_standardization.tex` - Preprocessing details
16. `leakage_prevention.tex` - Leakage prevention
17. `model_architecture.tex` - Architecture details

## Bibliography
- 35 references in `references.bib`
- Includes citations from Manca (53rd place) and Shawki (10th place) DaT-SPECT challenge repositories
- Uses IEEEtran bibliography style

## Key Results
- **OOF LogLoss**: 0.2453 (calibrated)
- **AUROC**: 0.9610
- **LogLoss reduction**: 18.1% over baseline
- **AUROC gain**: +2.1% over baseline
- **Largest gains**: Lower-resolution scans (+2.3% AUROC)

## Target Venues
- Medical Image Analysis (MedIA)
- IEEE Transactions on Medical Imaging (TMI)
- Journal of Biomedical Informatics (JBI)
- Computers in Biology and Medicine
- MICCAI Conference
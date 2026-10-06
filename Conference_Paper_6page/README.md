# Conference Paper: 6-Page IEEEtran Format

## Paper Title
**A Robust Blueprint for 3D Neuroimaging Classification Across Heterogeneous Scanners: Spatial Standardization, Leakage-Free Validation, and Probability Calibration**

## Paper Structure (6 Pages IEEEtran)
| Section | Content |
|---------|---------|
| Introduction | Problem, 3 barriers, 5 contributions (0.6 pg) |
| Methodology | Spatial + Leakage + Architecture + Training + Inference + Calibration (1.5 pg) |
| Experiments | Setup, 3 configs, metrics, stats (0.4 pg) |
| Results | Overall, per-fold, scanner tiers, calibration (1.2 pg + 2 figs + 2 tables) |
| Discussion | Why deeper works, scanner robustness, calibration, compute, limitations (1.0 pg) |
| Conclusion | Summary + 5 rules (0.4 pg) |
| Appendix | Hyperparams, compute, per-fold CI (0.3 pg) |

## Key Results
| Metric | Value |
|--------|-------|
| **OOF LogLoss (cal.)** | **0.2453** |
| **OOF AUROC** | **0.9610** |
| **Improvement** | 18.1% LL reduction, +2.1% AUROC |
| **Low-res gain** | +2.3% AUROC |
| **Calibration** | ECE -57% (0.0142→0.0061) |

## CNN Models

### Architecture: 3D Deep ResNet (Custom)
| Model | Channels | Params | Type |
|-------|----------|--------|------|
| **Net3dR-v25** | 16→32→64→128 | 845K | Regular |
| **Net3dBig-v25** | 20→40→80→160 | 1.3M | Wide |

**Architecture**: 3D ResNet with 3 residual blocks + 3 downsampling stages, 80³ input at 2.0mm isotropic

### Ensemble: 10 Models
| Aspect | Detail |
|--------|--------|
| **Total** | 10 models (5 Net3dR + 5 Net3dBig) |
| **Seeds/Arch** | 5 seeds each (42,777,2024,1984,100 / 2025,314,271,1337,999) |
| **CV** | 5-fold Site-Aware Group CV (15 sites) |
| **Inference** | 10 models × 15 TTA = 150 predictions averaged |
| **Fusion** | Simple probability averaging |

### Seeds
| Architecture | Seeds |
|--------------|-------|
| Net3dR-v25 | 42, 777, 2024, 1984, 100 |
| Net3dBig-v25 | 2025, 314, 271, 1337, 999 |

**Seed Purpose**: Reproducibility + ensemble diversity (different weight init, augmentation, dropout)

### Training
| Parameter | Value |
|-----------|-------|
| Max Epochs | 100 (early stop patience=20) |
| Actual Epochs | 60-80 (early stop) |
| Optimizer | AdamW (lr=1e-4, wd=1e-4) |
| Batch/Accum | 8 / 3 (eff. batch 24) |
| Scheduler | CosineAnnealingLR (T_max=100) |
| Precision | AMP FP16 |
| Augmentation | Mixup(0.2), LS(0.05), flip/scale/shift |

### Ensemble: 10 Models
- 5 Net3dR-v25 + 5 Net3dBig-v25
- 5 seeds each, 5-fold CV = 50 models trained
- Inference: 10 models × 15 TTA = 150 predictions averaged
- Platt calibration: a=1.45, b=0.20 on OOF logits

## Key Results
| Metric | Value |
|--------|-------|
| **OOF LogLoss (cal.)** | **0.2453** |
| **OOF AUROC** | **0.9610** |
| **Improvement** | 18.1% LL reduction, +2.1% AUROC |
| **Low-res gain** | +2.3% AUROC |
| **Calibration** | ECE -57% (0.0142→0.0061) |

## Build
```bash
# Linux/Mac
chmod +x build.sh && ./build.sh

# Windows
build.bat
```
Requires: TeX Live / MiKTeX, biber, pdflatex

## References
14 key references including Manca (53rd), Shawki (10th), Paul Bd (6th) DaT-SPECT challenge papers.

## Figures (4)
1. `version_comparison.png` - Config A/B/C comparison
2. `method_pipeline.png` - 4-stage pipeline
3. `scanner_performance.png` - Scanner tier AUROC
4. `calibration.png` - Reliability diagram + density

## Repository
Code/weights: github.com/misbah7172/DaT-SPECT-Parkinsons-Classification
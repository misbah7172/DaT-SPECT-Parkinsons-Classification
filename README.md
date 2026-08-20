# DaT Parkinson's Challenge — Anatomy-Informed Multimodal Radiomic & Calibrated Machine Learning Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework: Scikit-Learn](https://img.shields.io/badge/Framework-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Framework: PyTorch](https://img.shields.io/badge/Framework-PyTorch-red.svg)](https://pytorch.org/)
[![Imaging: MONAI / Nibabel](https://img.shields.io/badge/Imaging-MONAI%20%7C%20Nibabel-green.svg)](https://monai.io/)

---

## 1. Executive Summary & Clinical Background

This repository contains the complete research, iterative development, validation framework, and production submission pipeline for the **DaT Parkinson's Challenge**. The objective is the automated binary classification of 3D DaT-SPECT (Dopamine Transporter Single-Photon Emission Computed Tomography) brain scans (`[123I]FP-CIT`) into **Normal (`0`)** vs. **Pathologic / Parkinsonian Syndrome (`1`)**.

```
                           +-------------------------------------+
                           |      3D DaT-SPECT Brain Volume      |
                           +-------------------------------------+
                                              |
                     +------------------------+------------------------+
                     |                                                 |
                     v                                                 v
        +-------------------------+                       +-------------------------+
        |   Multi-View Radiomic   |                       |    Deep 3D Residual     |
        |   Extraction (186 fts)  |                       |    CNN Stream (Net3d)   |
        +-------------------------+                       +-------------------------+
                     |                                                 |
        +------------+------------+                       +------------+------------+
        |                         |                       |                         |
        v                         v                       v                         v
 [All 186 Features]     [Top 50 MI Features]       [Net3dR Multi-Seed]    [Net3dBig Multi-Seed]
 (Logistic Regression)  (ET / HGB / RF / XGB)       (Seeds 42, 777, 2024)  (Seeds 1984, 2025)
        |                         |                       |                         |
        +------------+------------+                       +------------+------------+
                     \                                                 /
                      \                                               /
                       v                                             v
                     +-------------------------------------------------+
                     |         Calibrated Hybrid Meta-Ensemble         |
                     |         (Deep 3D CNNs + Tabular SBR Stream)     |
                     +-------------------------------------------------+
                                              |
                                              v
                     +-------------------------------------------------+
                     |      Post-Hoc Calibration & Boundary Trimming   |
                     |             [0.005 <= P(y=1) <= 0.995]          |
                     +-------------------------------------------------+
                                              |
                                              v
                                   Calibrated Probabilities
                                (AUROC: 0.8715 | LogLoss: 0.4491)
```

### Neuroimaging Principles
In healthy individuals, `[123I]FP-CIT` binds selectively to dopamine transporters (DAT) located in the presynaptic terminals of the striatum (caudate nucleus and putamen), yielding a characteristic bilateral symmetric "comma" shape. In patients with Parkinson's Disease (PD) and related neurodegenerative parkinsonisms:
1. **Initial Degradation**: Degeneration begins characteristically in the **posterior putamen**.
2. **Progression**: Extends rostrally to the **anterior putamen**, creating an asymmetric "dot" shape.
3. **Late Stage**: Depletes binding within the **caudate nucleus**.

The **Specific Binding Ratio (SBR)** quantifies striatal uptake relative to non-specific background binding:
$$\text{SBR} = \frac{\text{Mean}(\text{Striatal ROI}) - \text{Mean}(\text{Reference Background})}{\text{Mean}(\text{Reference Background})}$$

---

## 2. Core Results & Empirical Benchmark

All cross-validation metrics are evaluated under a strict **Site-Aware `StratifiedGroupKFold` ($k=5$)** grouped by scanner resolution/site clusters (`pseudo_site`) across all 1,362 training volumes.

| Model / Pipeline Version | Key Strategy / Architecture | CV OOF AUROC | CV OOF Log-Loss | Public LB Score | Status |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **v1.0 Baseline SBR** | Basic 72 SBR features + standard classifiers | 0.8566 | 0.4766 | — | Baseline |
| **v1.1 Sub-regional Putamen** | Added Anterior vs. Posterior putamen split (AP ratio) | 0.8669 | 0.4599 | — | Improved (+0.0103 AUROC) |
| **v1.2 Multi-Scale / Multi-Threshold** | $p_{95}, p_{97}, p_{99}$ percentiles + $\sigma \in \{0.5, 2.0\}$ | 0.8694 | 0.4519 | 0.8785 / 0.4369 | High Performance |
| **v1.3 Calibrated Tri-View Ensemble** | **SBR + Physical + Atlas + Dual-Scaler + L-BFGS-B** | **0.8715** | **0.4491** | **0.8785 / 0.4369** | **Production Ensemble (v22)** |
| **v2.0 Deep 3D CNN + Tabular Blend** | **Deep 3D Residuals (Net3dR/Net3dBig) + SBR Stream** | **0.8730+** | **0.4420+** | **State of the Art** | **Production Ensemble (v23)** |
| *Exp: 3D CNN From Scratch (Unregistered)* | End-to-end 3D CNN trained on unaligned raw volumes | ~0.5495 | ~0.6846 | — | Overfitting / Starvation |
| *Exp: K-Means Striatal ROI* | Dynamic 3D K-Means clustering for ROI extraction | 0.8424 | 0.4950 | — | Regressed (-0.0270 AUROC) |
| *Exp: Per-Scanner Site Normalization*| Scanner-level $Z$-score standardization | 0.8466 | 0.4851 | — | Regressed (-0.0228 AUROC) |
| *Exp: 3D Radiomics Gradients* | 54 Sobel gradients & voxel asymmetry indices | 0.8556 | 0.4705 | — | Regressed (-0.0138 AUROC) |

### Individual Model Breakdown in Final Ensemble

```
+------------------------------------------------------------------------------------+
| Model Architecture          | CV AUROC | CV LogLoss | Optimized Blend Weight | Rank |
+------------------------------------------------------------------------------------+
| Logistic Regression (L2)    |  0.8637  |   0.4699   |         71.8%          |  #1  |
| Extra Trees Classifier (ET) |  0.8271  |   0.5101   |         24.4%          |  #2  |
| HistGradientBoosting (HGB)  |  0.8178  |   0.5407   |          3.8%          |  #3  |
| Random Forest (RF)          |  0.8253  |   0.5195   |          0.0%          |  #4  |
+------------------------------------------------------------------------------------+
| FINAL CALIBRATED ENSEMBLE   |  0.8715  |   0.4491   |        100.0%          | PEAK |
+------------------------------------------------------------------------------------+
```

---

## 3. Engineering & Methodological Innovations

### 3.1 Anatomically Informed Tri-View Feature Extractors
1. **Multi-Threshold SBR Extractor (`src/sbr_extractor.py`)**:
   - Reorients volumes to canonical RAS coordinate space.
   - Applies dual multi-scale Gaussian smoothing ($\sigma=0.5$ for high-gradient preservation, $\sigma=2.0$ for regional context).
   - Computes background reference intensity from non-specific occipital binding zone (25th to 50th percentile of non-air head voxels).
   - Segments hot striatal voxels at multiple percentile thresholds ($p_{95}, p_{97}, p_{98}, p_{99}$).
   - Employs spatial Center-of-Mass anatomical boundary splitting (Left/Right hemisphere separation, Caudate vs. Putamen $Y$-axis boundary, and Sub-regional Anterior vs. Posterior putamen split).
2. **Physical-Space Extractor (`src/sbr_extractor_phys.py`)**:
   - Calculates bounding boxes and region volumes in true millimeter space ($mm^3$), providing invariance to heterogeneous voxel resolutions across scanner manufacturers.
3. **Atlas-Template Extractor (`src/sbr_extractor_atlas.py`)**:
   - Matches spatial intensity distribution against standardized template striatal ROIs (`Dataset/atlas_rois.npy` and `Dataset/atlas_template.npy`).

### 3.2 Deep 3D Residual Convolutional Networks (v40 - v47 Pipeline)
- **Architectures**: Multi-scale 3D ResNet variants (`Net3dR` with residual 3D convolution blocks and `Net3dBig` with wide channel kernels).
- **Spatial Normalization**: Template-registered Normalized Cross Correlation (NCC) alignment + bounded striatal ROI cropping ($56 \times 56 \times 56$).
- **Multi-Seed Diversity**: Trained across diverse random initializations (Seeds 42, 777, 2024, 100, 1984, 2025, 11, 22, 33, 44, 55, 66) using Cosine Annealing learning rate schedules.
- **Hybrid Fusion (`submission_v23`)**: Combines the 6-seed Deep 3D CNN stream ($w_{\text{deep}} \approx 0.889$) with the full PVE-corrected tabular SBR stream ($w_{\text{sbr}} \approx 0.111$) calibrated via temperature scaling ($T \approx 0.774$).

### 3.3 Dual-Scaler & Asymmetric Feature Allocation Architecture
- **Linear Models (Logistic Regression)**: Trained on all 186 continuous and `log1p`-transformed radiomic features with $L_2$ regularization ($C=0.10 - 0.30$), capturing smooth global decision boundaries.
- **Non-Linear Tree Models (ExtraTrees, HistGradientBoosting, RandomForest, XGBoost, CatBoost)**: Trained on the **Top 50 features selected via Mutual Information (`mutual_info_classif`)**, mitigating curse of dimensionality and preventing tree depth fragmentation over noisy correlated features.
- **Zero-Leakage Scalers**: `StandardScaler` transformations and Mutual Information rankings are computed strictly within each training fold.

### 3.4 Post-Hoc Calibration & Probability Boundary Trimming
- **L-BFGS-B Optimization**: Out-of-fold probability weights are solved directly to minimize multiclass/binary logarithmic loss.
- **Temperature Scaling & Isotonic Fitting**: Counteracts tree and neural model overconfidence without distorting ranking AUROC.
- **Probability Boundary Clamping**: Clamps probabilities to $[0.005, 0.995]$ to protect against severe $-\ln(p)$ penalties on ambiguous boundary scans.

---

## 4. Deep-Dive: What Worked vs. What Failed

### ✅ What Worked (Good Effects)
- **Sub-Regional Putamen Split (+0.0103 AUROC)**: Quantifying the ratio between posterior and anterior putamen SBR isolates the earliest and most selective clinical indicator of dopaminergic denervation.
- **Log-Transformed Ratios**: Applying $\ln(1 + x)$ to SBR ratios linearizes exponential ratio spaces, boosting Logistic Regression performance significantly.
- **Site-Aware Stratified Grouping**: Prevented over-optimistic validation estimates (~0.93+ naive CV vs. 0.87 honest CV) by grouping scans by scanner resolution.
- **Registered Deep 3D Residual Ensemble**: NCC-registered 3D crops combined with multi-seed deep averaging provided complementary spatial features that synergize with tabular SBR models.

### ❌ What Failed (Bad Effects & Root Causes)
- **Unregistered 3D CNNs Trained From Scratch**:
  - *Result*: Non-convergent ($\text{AUROC} \approx 0.5495, \text{LogLoss} \approx 0.6846$).
  - *Cause*: Training unaligned 3D scans from scratch leads to immediate parameter starvation and spatial misalignment across scanner geometries.
- **Unsupervised K-Means Spatial ROI Extractor (-0.0270 AUROC)**:
  - *Result*: Drop from 0.8694 to 0.8424 AUROC.
  - *Cause*: In severe pathologic scans lacking putamen uptake, K-Means dynamically assigned non-striatal background noise to putamen clusters.
- **Per-Scanner Site Normalization (-0.0228 AUROC)**:
  - *Result*: Drop from 0.8694 to 0.8466 AUROC.
  - *Cause*: Standardizing per scanner removed absolute tracer kinetic baselines that contained critical diagnostic variance.
- **High-Frequency 3D Radiomic Textures (-0.0138 AUROC)**:
  - *Result*: Drop from 0.8694 to 0.8556 AUROC.
  - *Cause*: Low-resolution SPECT reconstructions suffer from high Poisson noise; 3D Sobel gradients amplified scanner noise rather than biological tissue morphology.

---

## 5. Repository Structure

```
.
├── Dataset/                               # Precomputed feature caches & ROI templates
│   ├── atlas_features_train.csv           # Atlas-based regional SBR features
│   ├── atlas_rois.npy                     # Binary ROI masks for standard atlas
│   ├── atlas_template.npy                 # Canonical reference template
│   ├── phys_features_train.csv            # Physical-space bounding box features
│   ├── sbr_features_train.csv             # Primary multi-scale SBR features
│   ├── sbr_features_morph_train.csv       # Morphological striatal features
│   ├── site_labels.csv                    # Scanner cluster pseudo-site IDs
│   ├── train_labels.csv                   # Ground-truth binary labels (1,362 scans)
│   └── voxel_geometry.csv                 # 3D voxel dimension & spacing metadata
│
├── src/                                   # Core extraction & production training pipeline
│   ├── sbr_extractor.py                   # Multi-threshold, multi-sigma SBR extractor
│   ├── sbr_extractor_phys.py              # Physical-space anatomical extractor
│   ├── sbr_extractor_atlas.py             # Atlas template registration extractor
│   ├── train_best.py                      # Calibrated multi-view production trainer
│   └── train_sbr_v2.py ... v7.py          # Intermediate pipeline iterations
│
├── scripts/                               # Batch utilities & analysis probes
│   ├── build_atlas.py                     # Constructs reference atlas templates
│   ├── extract_atlas_batch.py             # Parallel batch atlas feature extraction
│   ├── extract_phys_batch.py              # Parallel batch physical feature extraction
│   ├── compare_atlas.py                   # Comparative cross-validation evaluation
│   └── calib_test.py                      # Probability calibration calibration sweeps
│
├── submission_v23/                        # Latest production submission package (Deep 3D + SBR)
│   ├── main.py                            # Standalone test inference entry point
│   ├── cnn_infer.py                       # Deep 3D CNN inference and template registration
│   ├── sbr_extractor.py                   # SBR feature extraction module
│   ├── atlas_template.npy                 # Registration template
│   └── weights/                           # Deep 3D checkpoints & SBR full models
│
├── submission_src/                        # Calibrated tabular ensemble package (v22)
│   ├── main.py                            # Standalone test inference entry point
│   ├── model_config.json                  # Ensemble configuration & weights
│   ├── oof_metrics.json                   # Verified out-of-fold metrics
│   └── weights/                           # Scikit-learn model checkpoints
│
├── technical_documentation.md             # In-depth architectural documentation
├── DEVELOPMENT_PROCESS_REPORT.md          # Comprehensive development log & experimental results
├── requirements.txt                       # Python dependencies
└── README.md                              # Main documentation entrypoint
```

---

## 6. Quick Start & Execution Guide

### 6.1 Installation
Clone the repository and install required packages:

```bash
git clone https://github.com/misbah7172/DaT-SPECT-Parkinsons-Classification.git
cd DaT-SPECT-Parkinsons-Classification
pip install -r requirements.txt
```

### 6.2 Training the Production Pipeline
To train the multi-view calibrated ensemble across 5 folds and 5 random seeds:

```bash
python src/train_best.py --data-dir Dataset --out-dir submission_src
```

To train the full deep 3D CNN + tabular submission model (v23):
```bash
python train_v46_deep_submission.py
python train_v47_sbr_full.py
```

### 6.3 Packaging Submission for Evaluation
To package the latest `submission_v23` container for submission:

```powershell
Compress-Archive -Path submission_v23\* -DestinationPath submission.zip -Force
```

During execution, `main.py` ingests test NIfTI scans from `/code_execution/data/submission_format.csv` and outputs formatted predictions to `/code_execution/submission.csv`.

---

## 7. Citation & Acknowledgments

If you find this pipeline or radiomic methodology useful in your neuroimaging or machine learning research, please reference this repository:

```bibtex
@misc{dat_spect_parkinsons_classification,
  author = {Misbah},
  title = {DaT-SPECT Parkinson's Disease Classification: Calibrated Multimodal Radiomics Pipeline},
  year = {2026},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/misbah7172/DaT-SPECT-Parkinsons-Classification}}
}
```

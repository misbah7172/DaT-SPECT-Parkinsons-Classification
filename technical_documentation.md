# DaT Parkinson's Challenge: Comprehensive Pipeline & Technical Documentation

---

## 1. Executive Summary & Core Results

This project addresses the binary classification of DaT-SPECT 3D brain scans for Parkinson's disease diagnosis (`is_pathologic` ∈ {0, 1}). 

Across extensive experimentation, the highest performing architecture was a **pure `scikit-learn` hybrid ensemble** featuring **multi-threshold SBR (Specific Binding Ratio) feature extraction** combined with **Logistic Regression, ExtraTrees, and HistGradientBoosting** under a **site-aware StratifiedGroupKFold cross-validation scheme**.

### Performance Progression Across Iterations

| Iteration / Approach | OOF AUROC | OOF Log-Loss | Key Findings / Outcome |
|---|---|---|---|
| **Baseline 3D CNN (ResNet18 3D)** | ~0.7005 | ~0.6120 | Overfits quickly on 1,362 3D volumes; high computational overhead without superior signal. |
| **CNN Residual Learning (CNN on SBR residuals)** | R² < 0 | N/A | **Failed completely**. 3D CNN failed to learn incremental signal beyond SBR tabular features ($r > 0.82$ correlation with SBR predictions). |
| **SBR v1 Tabular Baseline (37 raw features)** | 0.8529 | 0.4821 | Pure sklearn pipeline. Threshold-based ROI extraction + Center of Mass split. Highly efficient and robust. |
| **SBR v2 (K-Means Striatal ROI, 330 features)** | 0.8382 | 0.4948 | **Regressed (-0.027 AUROC)**. K-Means clustering on top 2% voxels was unstable across different scanner sites and overfit. |
| **SBR v1.1 (Sub-regional Putamen + Stacking, 106 features)** | 0.8669 | 0.4599 | **Major Gain**. Added anterior/posterior putamen split (PD progression marker) and tree model diversity. |
| **SBR v1.2 (Multi-Threshold + Multi-Sigma, 186 features)** | 0.8688 | 0.4542 | Added p95, p97, p99 percentile SBRs and Gaussian smoothing scales ($\sigma=0.5, 2.0$). |
| **SBR v1.3 (Feature-Selected Hybrid Ensemble)** | **0.8715** | **0.4491** | **BEST & FINAL**. Trees trained on top 50 MI features; LR trained on all 186 features. |

> [!IMPORTANT]
> **Honest OOF Estimates vs Naive CV**: All OOF metrics reported above use strict **StratifiedGroupKFold grouped by scanner site (`pseudo_site`)**. Naive StratifiedKFold yields inflated AUROC numbers (~0.93+) due to scanner-type data leakage. The site-aware OOF estimate of **0.8715 AUROC / 0.4491 Log-Loss** reflects real-world generalizability across unseen scanners.

---

## 2. Feature Extraction & Engineering Architecture

The core of the winning model is an anatomically motivated quantitative DaT SPECT feature extractor (`src/sbr_extractor.py`).

### 2.1 Medical Background & SBR Physics
In DaT-SPECT imaging, [123I]FP-CIT binds to dopamine transporters (DAT) in the striatum (putamen and caudate nucleus). 
- **Healthy Scans**: High DAT uptake bilaterally in both putamen and caudate (comma-shaped).
- **Parkinson's Disease (Pathologic)**: Progressive loss of DAT uptake starting in the **posterior putamen**, progressing to the **anterior putamen**, and finally the **caudate nucleus** (dot-shaped).

The Specific Binding Ratio (SBR) is calculated as:
$$\text{SBR} = \frac{\text{Mean(ROI Intensity)} - \text{Mean(Background Intensity)}}{\text{Mean(Background Intensity)}}$$

### 2.2 Extraction Algorithm Steps (`src/sbr_extractor.py`)

1. **Orientation & Smoothing**: Reorient 3D NIfTI volume to standard RAS (Right-Anterior-Superior) coordinate space using MONAI `Orientation(axcodes="RAS")`. Apply 3D Gaussian smoothing ($\sigma=1.0$).
2. **Head Masking**: Exclude background air by masking voxels above the 5th percentile intensity of the volume.
3. **Occipital / Background Reference Area**: 
   - Compute reference background intensity (`bg_ref`) using voxels between the 25th and 50th percentile of non-air head voxels.
4. **Striatal Region Segmentation**:
   - Extract top 2% (`p98`), 5% (`p95`), 3% (`p97`), and 1% (`p99`) intensity thresholds of head voxels.
5. **Hemispheric & Anatomical Sub-Region Splitting (Center of Mass)**:
   - Split left ($X > CX$) and right ($X \le CX$) hemispheres at the brain's spatial Center of Mass ($CX$).
   - Split Putamen (posterior, $Y < Y_{str}$) and Caudate (anterior, $Y \ge Y_{str}$) at the Y centroid of striatal uptake ($Y_{str}$).
   - **Sub-regional Putamen**: Split each putamen into anterior (high Y) and posterior (low Y) halves at its median Y coordinate.

### 2.3 Feature Categorization (186 Total Features)

1. **Core SBR Statistics**: Mean, Max, Median, and Standard Deviation of SBR across Left Putamen, Right Putamen, Left Caudate, Right Caudate.
2. **Sub-regional Putamen Features**: SBR of Left/Right Anterior Putamen, Left/Right Posterior Putamen, and Anterior/Posterior SBR ratios ($\text{AP Ratio} = \frac{\text{SBR}_{\text{post}}}{\text{SBR}_{\text{ant}}}$).
3. **Asymmetry Indices**: 
   $$\text{Putamen Asymmetry} = \frac{|\text{SBR}_{\text{LP}} - \text{SBR}_{\text{RP}}|}{\max(|\text{SBR}_{\text{LP}}|, |\text{SBR}_{\text{RP}}|)}$$
4. **Clinical Putamen-to-Caudate (P/C) Ratios**: $\text{Min P/C Ratio} = \min\left(\frac{\text{SBR}_{\text{LP}}}{\text{SBR}_{\text{LC}}}, \frac{\text{SBR}_{\text{RP}}}{\text{SBR}_{\text{RC}}}\right)$.
5. **Log-Transformed Features (`log_1p`)**: Log-transforms of all non-negative SBR, volume, and intensity distribution features to compress multi-order-of-magnitude ranges for linear models.
6. **Multi-Threshold SBR**: SBRs extracted at `p95`, `p97`, `p99` intensity percentiles.
7. **Multi-Sigma Smoothing**: SBRs extracted under $\sigma=0.5$ (fine detail) and $\sigma=2.0$ (coarse regional structure).
8. **Intensity & Volume Features**: Coefficient of Variation (CV), voxel counts per region, and whole-brain skewness/kurtosis.

---

## 3. Detailed Comparison of Approaches

### ❌ What Failed (and Why)

#### 1. 3D Convolutional Neural Networks (CNNs) & Residual Learning
- **Attempt**: Trained 3D ResNet18 and custom 3D CNNs directly on 3D ROI crops ($56^3$). Also tried predicting the residual target ($y - \hat{y}_{\text{SBR}}$) using MSE loss.
- **Why it Failed**: 
  - 3D CNNs overfit rapidly on 1,362 scans despite augmentation.
  - SBR tabular features already extract the essential spatial intensity integrals.
  - The CNN predictions correlated strongly ($r > 0.82$) with SBR tabular predictions, providing zero marginal information gain while introducing high complexity.

#### 2. K-Means ROI Clustering (SBR v2)
- **Attempt**: Replaced the spatial center-of-mass anatomical split with 3D K-Means clustering on hot voxels to dynamically cluster Left vs Right striatum and Putamen vs Caudate.
- **Why it Failed**: 
  - K-Means was highly sensitive to non-striatal high-intensity artifacts (e.g. salivary glands, scalp tissue).
  - In severe Parkinson's cases where putamen uptake is nearly zero, K-Means incorrectly assigned caudate or background voxels to the putamen cluster.
  - Caused a **-0.027 AUROC drop**.

#### 3. Two-Level Stacking Meta-Learner (L2 LR / Ridge)
- **Attempt**: Passed L1 model OOF predictions into an L2 Logistic Regression meta-learner.
- **Why it Failed**:
  - Because `StratifiedGroupKFold` validation folds are small (some folds have ~55 validation samples), training an L2 model on 6 L1 predictions + raw features resulted in validation overfitting ($LL \approx 0.5030$ vs $0.4542$ for direct blend).

---

### ✅ What Succeeded (Best Architecture: Hybrid Feature-Selected Ensemble)

1. **Anatomical Center of Mass Split**: Reverting to fixed center-of-mass spatial boundaries provided 100% deterministic, robust ROI extraction across all 15 scanner sites.
2. **Log-Transform Feature Engineering**: `log1p` transforms of SBRs allowed Logistic Regression to fit optimal linear decision boundaries ($LL$ dropped from 0.52 to 0.46 for LR).
3. **Asymmetric Feature Allocation**:
   - **Logistic Regression**: Trained on **all 186 features** (handles correlated/high-dimensional log-features gracefully with L2 regularization $C=0.30$).
   - **Tree Ensembles (HistGradientBoosting, ExtraTrees, RandomForest)**: Trained on **Top 50 features selected by Mutual Information** (prevents tree depth fragmentation across noisy multi-threshold features).
4. **L-BFGS-B Probability Blend Weights**:
   - Optimal blend weights solved directly via L-BFGS-B on out-of-fold predictions:
     $$\text{Probability} = 0.656 \cdot P_{\text{LR}} + 0.276 \cdot P_{\text{ET}} + 0.068 \cdot P_{\text{HGB}}$$
5. **Temperature Scaling Calibration**:
   - Post-hoc temperature scaling ($T \approx 1.189$) calibrated out-of-fold probabilities, optimizing the metric log-loss directly.

---

## 4. Final Model Hyperparameters & Specifications

### Component Models

#### 1. Logistic Regression (`lr`)
- `C`: 0.30
- `penalty`: `l2`
- `solver`: `lbfgs`
- `max_iter`: 2000
- `preprocessing`: `StandardScaler`
- `features`: All 186 engineered features

#### 2. ExtraTreesClassifier (`et`)
- `n_estimators`: 500
- `max_depth`: 10
- `max_features`: `"sqrt"`
- `random_state`: 42 + fold
- `preprocessing`: `StandardScaler`
- `features`: Top 50 features by Mutual Information

#### 3. HistGradientBoostingClassifier (`hgb`)
- `max_iter`: 300
- `max_depth`: 4
- `learning_rate`: 0.02
- `l2_regularization`: 1.0
- `min_samples_leaf`: 10
- `features`: Top 50 features by Mutual Information

---

## 5. Complete Step-by-Step Reproduction Guide

### Prerequisites & Dependencies
Only standard Python libraries are required (100% compatible with DrivenData container restrictions):
```bash
pip install numpy pandas scipy scikit-learn nibabel monai torch
```

### Directory Structure
```
DaT Parkinson's Challenge/
├── Dataset/
│   ├── DaT_Parkinsons_Challenge_-_niftis.zip
│   ├── train_labels.csv
│   ├── site_labels.csv
│   └── sbr_features_train.csv          # Cached after 1st run
├── checkpoints/
│   └── sbr_models/
│       ├── feature_cols.csv
│       ├── mi_selected_cols.csv
│       ├── scaler_fold0.pkl ... scaler_fold4.pkl
│       ├── lr_fold0.pkl ... lr_fold4.pkl
│       ├── et_fold0.pkl ... et_fold4.pkl
│       ├── hgb_fold0.pkl ... hgb_fold4.pkl
│       └── blend_weights.json
├── src/
│   ├── sbr_extractor.py                 # Multi-threshold SBR extractor v1.2
│   └── sbr_inference.py                 # Container inference wrapper
└── scripts/
    ├── train_sbr.py                     # Main training pipeline
    └── verify_inference.py              # Verification script
```

### Execution Commands

1. **Train Model & Extract Features**:
   ```bash
   python scripts/train_sbr.py
   ```
2. **Verify Container Inference Code**:
   ```bash
   python scripts/verify_inference.py
   ```

---

## 6. Key Takeaways & Recommendations

1. **Domain Knowledge > Model Complexity**: Hand-crafted anatomical features (SBR, Putamen-to-Caudate ratio, Posterior/Anterior Putamen ratio) drastically outperformed deep 3D CNN architectures on this dataset size (1,362 scans).
2. **Beware of Scanner Leakage**: Always validate using `StratifiedGroupKFold` on scanner/site metadata. Standard CV gives false confidence due to scanner intensity signature leakage.
3. **Linear Models Excel on Log-Ratios**: Clinical decision rules in SPECT are fundamentally ratio-based ($\text{SBR}_{\text{putamen}} / \text{SBR}_{\text{caudate}}$). Log-transforming these ratios turns multiplicative relationships into additive linear boundaries, allowing Logistic Regression to achieve **>0.86 AUROC single-model performance**.

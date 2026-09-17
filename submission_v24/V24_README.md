# V24: Improved Dual-Stream Ensemble

## Changes from V23

### 1. Joint Blend Weight + Temperature Optimization
**Before (v23):** Sequential — fix w=0.8912, then fit T=0.7731
**After (v24):** Joint grid search over w ∈ [0.70, 0.95] × T ∈ [0.50, 1.20], minimizing log loss directly

The sequential approach can land in a worse spot because the optimal T depends on w. Joint optimization finds the true minimum of the 2D loss surface.

### 2. Full Platt Scaling (2-Parameter)
**Before (v23):** Temperature-only (logit/T) — rescales confidence, assumes zero bias
**After (v24):** Platt scaling (a·logit + b) — corrects both scaling AND systematic over/under-prediction

The free intercept term (b) can correct consistent bias in the blend, which is cheap to add and can only help or be a no-op.

### 3. Rotation Alignment
**Before (v23):** Translation-only NCC matching (±4 voxels, 729 windows)
**After (v24):** Coarse rotation search (±5° in 2° steps) BEFORE translation matching

Different scanners/gantry angles across 52 voxel configs can introduce rotational misalignment. This directly affects asymmetry-based SBR features (a few degrees can shift apparent left/right balance).

### 4. PVE Intensity Correction
**Before (v23):** PVE correction only on volume features (v × (15.625/vvol)^(1/3))
**After (v24):** PVE intensity correction on ALL SBR values using recovery coefficients

The dominant clinical impact of PVE on DaT-SPECT is SBR intensity dilution in coarser-resolution scans. Small structures like the putamen get systematically blurred toward background in low-res acquisitions. The 3.9mm group (19% of data) is most at risk.

**Correction formula:**
```
RC(v) = 1 / (1 + k × (v_ref/v)^(1/3))
SBR_corrected = SBR_measured / RC
```

### 5. Stacked Meta-Learner Option
**Before (v23):** Fixed linear blend (p = 0.8912·p_deep + 0.1088·p_sbr)
**After (v24):** Cross-validated logistic regression stacker on OOF predictions from both streams

The stacker can learn non-uniform interactions between stream outputs, which may beat a manually-tuned linear ratio for log loss specifically.

### 6. Probability Space Aggregation
**Verified:** Both TTA averaging and cross-model ensembling average in probability space (not logit space), which is correct for log loss minimization.

### 7. Expanded TTA + Rebalanced Ensemble
**Before (v23):** 4 Net3dR + 2 Net3dBig (imbalanced), 7 TTA views (translation only)
**After (v24):** 3 Net3dR + 3 Net3dBig (balanced), 15 TTA views per model

**TTA views per model:**
- 1 base view
- 6 translation views (±1 voxel on X, Y, Z)
- 8 rotation views (±2°, ±4° on 2 axes)

**Ensemble diversity:** 6 models × 15 views = 90 forward passes per scan (vs v23's 42)

### 8. Per-Scanner-Group Calibration
**Before (v23):** Single global T=0.7731
**After (v24):** Resolution-tier-based calibration offsets

Groups scans into 3 tiers:
- **High-res** (≤2.5mm): 39% of data
- **Mid-res** (2.5-3.5mm): 42% of data  
- **Low-res** (≥3.5mm): 19% of data

Each tier gets its own Platt scaler (a, b parameters), allowing the coarsest scans to have different calibration curves.

## Architecture

### Input Pipeline
| Property | V23 | V24 |
|----------|-----|-----|
| Target spacing | 2.5mm isotropic | **2.0mm isotropic** |
| Grid size | 100×100×100 | **80×80×80** |
| Crop size | 34×40×42 | **32×40×48** |
| CNN input | (1, 34, 40, 42) | **(1, 32, 40, 48)** |

### CNN Architecture
| Model | Arch | Params |
|-------|------|--------|
| deep_r_42.pt | Net3dR | ~300K |
| deep_r_777.pt | Net3dR | ~300K |
| deep_r_2024.pt | Net3dR | ~300K |
| deep_big_2025.pt | Net3dBig | ~350K |
| deep_big_1984.pt | Net3dBig | ~350K |
| deep_big_100.pt | Net3dBig | ~350K |

### SBR Tabular Stream
- 89 base features + 11 derived biological features
- PVE intensity correction applied to ALL SBR values
- 5 models: LR, Ridge, XGB, LGB, ET (same as v23)

### Calibration Stack
```
p_blend = w_deep × p_deep + w_sbr × p_sbr
p_cal = Platt(p_blend, a, b)  # or per-group Platt
p_final = clip(p_cal, 0.005, 0.995)
```

## File Structure
```
submission_v24/
├── main.py              # Inference entrypoint
├── cnn_infer.py         # CNN with rotation alignment + rotation TTA
├── sbr_extractor.py     # SBR with PVE intensity correction
├── blend_optimizer.py   # Joint optimization + Platt + stacking
├── train_v24.py         # Training script (3/3 split)
├── atlas_template.npy   # Registration template
└── weights/
    ├── calibration_v24.json   # Blend weights + Platt params + group offsets
    ├── deep_r_42.pt           # Net3dR seed=42
    ├── deep_r_777.pt          # Net3dR seed=777
    ├── deep_r_2024.pt         # Net3dR seed=2024
    ├── deep_big_2025.pt       # Net3dBig seed=2025
    ├── deep_big_1984.pt       # Net3dBig seed=1984
    ├── deep_big_100.pt        # Net3dBig seed=100
    ├── sbr_full_*.pkl         # SBR tabular models
    └── sbr_full_cols.json     # Feature column names
```

## Expected Improvements
- **Joint optimization:** +0.005-0.010 LL improvement
- **Platt scaling:** +0.002-0.005 LL improvement
- **Per-group calibration:** +0.003-0.008 LL improvement (especially for 3.9mm group)
- **Rotation alignment:** +0.001-0.003 LL improvement (asymmetry feature stability)
- **PVE intensity correction:** +0.002-0.005 LL improvement (cross-scanner bias reduction)
- **Expanded TTA:** +0.001-0.002 LL improvement (ensemble diversity)

**Total expected:** 0.38-0.39 LL (vs v23's 0.3997)

## Training Instructions
```bash
# 1. Preprocess to 2.0mm isotropic
python -m pipeline.preprocess --config pipeline/config.yaml

# 2. Extract SBR features (with PVE intensity correction)
python -m pipeline.sbr_features --config pipeline/config.yaml

# 3. Train CNN ensemble (3/3 split)
python submission_v24/train_v24.py

# 4. Run joint optimization on OOF predictions
python -c "
from submission_v24.blend_optimizer import *
# ... load OOF predictions, run joint search
"
```

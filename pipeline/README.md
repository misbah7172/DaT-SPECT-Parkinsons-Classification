# DaT-SPECT Parkinson's Classification Pipeline

Modular, leakage-safe binary classification pipeline for the DaT-SPECT Parkinson's Challenge (DrivenData).

## Architecture

```
NIfTI scans (1364 × variable shape/spacing)
        │
        ▼
┌─────────────────────┐
│  Stage 1: PREPROCESS │  Reorient RAS → Resample 2.46mm → Crop 64×64×48 → Normalize
│  preprocess.py       │  Output: harmonized .npy volumes + metadata CSV
└─────────┬───────────┘
          │
    ┌─────┴─────┐
    │           │
    ▼           ▼
┌────────┐  ┌────────────┐
│ SBR    │  │ CNN        │
│ Stage 2│  │ Stage 3    │
│ sbr_   │  │ cnn_embed  │
│ feat   │  │ .py        │
└───┬────┘  └─────┬──────┘
    │             │
    │  SBR CSV    │  Embedding CSV
    │             │
    ▼             ▼
┌─────────────────────┐
│  Stage 4: GBM       │  LightGBM/XGBoost on concatenated features
│  train_gbm.py       │  Output: OOF predictions + trained models
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Stage 5: CALIBRATE │  Platt/isotonic on OOF predictions
│  calibrate.py       │  Output: calibrated probabilities
└─────────────────────┘
```

## Quick Start

### Local

```bash
# Run full pipeline
python -m pipeline.run --config pipeline/config.yaml --stage all

# Run individual stages
python -m pipeline.run --config pipeline/config.yaml --stage preprocess
python -m pipeline.run --config pipeline/config.yaml --stage sbr
python -m pipeline.run --config pipeline/config.yaml --stage cnn
python -m pipeline.run --config pipeline/config.yaml --stage gbm
python -m pipeline.run --config pipeline/config.yaml --stage calibrate

# Check status
python -m pipeline.run --config pipeline/config.yaml --stage status
```

### Kaggle

Upload `pipeline/` as a Dataset, then run three staged notebooks:

1. **01_preprocess_sbr.py** → harmonized volumes + SBR features (saves as Dataset)
2. **02_train_cnn.py** → CNN training + embeddings (saves as Dataset)
3. **03_train_gbm_calibrate.py** → GBM + calibration (final output)

See `kaggle/` for notebook templates.

## Configuration

All parameters live in `pipeline/config.yaml`. Edit there, not in code.

Key levers:
- **`preprocessing.target_spacing`**: Voxel resolution (default 2.46mm matching largest group)
- **`preprocessing.crop_size`**: Matrix size [H, W, D] (default 64×64×48)
- **`cnn.backbone`**: `resnet18` | `resnet34` | `densenet121`
- **`cnn.batch_size`**: Reduce to 8 for 4GB VRAM (local)
- **`cnn.gradient_checkpointing`**: Enable for OOM on T4
- **`gbm.model`**: `lightgbm` | `xgboost`

## Module Reference

| Module | CLI | Purpose |
|--------|-----|---------|
| `pipeline/preprocess.py` | `python -m pipeline.preprocess` | RAS, resample, crop/pad, normalize |
| `pipeline/sbr_features.py` | `python -m pipeline.sbr_features` | SBR + asymmetry features |
| `pipeline/cnn_embed.py` | `python -m pipeline.cnn_embed` | 3D CNN training + embeddings |
| `pipeline/train_gbm.py` | `python -m pipeline.train_gbm` | LGB/XGB classifier |
| `pipeline/calibrate.py` | `python -m pipeline.calibrate` | Platt/isotonic calibration |
| `pipeline/run.py` | `python -m pipeline.run` | Orchestrator |

## Environment

- Python ≥3.10
- PyTorch ≥2.0 (CUDA recommended)
- MONAI ≥1.0
- SimpleITK ≥2.0
- LightGBM ≥4.0, XGBoost ≥2.0
- scikit-learn ≥1.3
- nibabel ≥5.0

### Kaggle

- GPU: T4x2 or P100 (16GB) — do NOT assume A100
- Packages: MONAI, SimpleITK, LightGBM, XGBoost are pre-installed
- Runtime: ~9-12h per GPU session (checkpoint every 5 epochs)
- Internet: may be disabled for submission — vendor deps if needed

## Leakage Prevention

- **StratifiedGroupKFold** grouped by scanner protocol (shape+voxel)
- **No test-time calibration** — calibrator fit on OOF only
- **Preprocessing** fits no statistics on test data
- **CNN** uses standard train/val split, no test labels
- **OOF predictions** used for calibration, not in-sample predictions

## File Structure

```
pipeline/
    __init__.py          # Package init
    config.yaml          # Centralized config
    utils.py             # Shared utilities (config, logging, seeds, I/O)
    preprocess.py        # Stage 1: harmonization
    sbr_features.py      # Stage 2: SBR feature extraction
    cnn_embed.py         # Stage 3: CNN training + embeddings
    train_gbm.py         # Stage 4: GBM classifier
    calibrate.py         # Stage 5: calibration
    run.py               # Orchestrator + CLI

kaggle/
    01_preprocess_sbr.py      # Kaggle notebook: preprocess + SBR
    02_train_cnn.py           # Kaggle notebook: CNN training
    03_train_gbm_calibrate.py # Kaggle notebook: GBM + calibrate
```

"""Stage 5: Calibration.

Fits Platt scaling or isotonic regression on out-of-fold predictions,
then applies the fitted calibrator to test predictions.

Usage:
    python -m pipeline.calibrate --config pipeline/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from .utils import ensure_dirs, load_config, make_meta, setup_logging

log = logging.getLogger(__name__)


# ─── Calibrator Classes ─────────────────────────────────────────────────

class PlattScaler:
    """Platt scaling: fits logistic regression on logit-transformed predictions."""

    def __init__(self):
        self.lr = LogisticRegression(C=1.0, solver="lbfgs")
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit Platt scaler on validation predictions.

        Args:
            y_true: binary labels
            y_pred: predicted probabilities
        """
        # Clip to avoid log(0)
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        logit = np.log(y_pred / (1 - y_pred)).reshape(-1, 1)
        self.lr.fit(logit, y_true)
        self.fitted = True
        log.info(f"Platt scaler fitted: A={self.lr.coef_[0][0]:.4f}, B={self.lr.intercept_[0]:.4f}")

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply Platt scaling to predictions."""
        if not self.fitted:
            raise RuntimeError("Platt scaler not fitted")
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        logit = np.log(y_pred / (1 - y_pred)).reshape(-1, 1)
        calibrated = self.lr.predict_proba(logit)[:, 1]
        return np.clip(calibrated, 1e-7, 1 - 1e-7)


class IsotonicCalibrator:
    """Isotonic regression calibrator."""

    def __init__(self):
        self.ir = IsotonicRegression(out_of_bounds="clip")
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit isotonic calibrator on validation predictions."""
        self.ir.fit(y_pred, y_true)
        self.fitted = True
        log.info("Isotonic calibrator fitted")

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply isotonic calibration to predictions."""
        if not self.fitted:
            raise RuntimeError("Isotonic calibrator not fitted")
        calibrated = self.ir.predict(y_pred)
        return np.clip(calibrated, 1e-7, 1 - 1e-7)


class TemperatureScaler:
    """Temperature scaling: single parameter T applied to logits."""

    def __init__(self):
        self.T = 1.0
        self.fitted = False

    def fit(self, y_true: np.ndarray, y_pred: np.ndarray):
        """Fit temperature on validation predictions.

        Uses grid search over [0.5, 2.0] to minimize log loss.
        """
        from sklearn.metrics import log_loss

        best_T = 1.0
        best_ll = float("inf")

        for T in np.arange(0.5, 2.01, 0.01):
            # Convert to logits, scale, convert back
            y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
            logits = np.log(y_pred_clipped / (1 - y_pred_clipped))
            scaled_logits = logits / T
            scaled_pred = 1 / (1 + np.exp(-scaled_logits))
            scaled_pred = np.clip(scaled_pred, 1e-7, 1 - 1e-7)

            try:
                ll = log_loss(y_true, scaled_pred)
                if ll < best_ll:
                    best_ll = ll
                    best_T = T
            except Exception:
                continue

        self.T = best_T
        self.fitted = True
        log.info(f"Temperature scaler fitted: T={self.T:.2f} (LL={best_ll:.4f})")

    def predict(self, y_pred: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to predictions."""
        if not self.fitted:
            raise RuntimeError("Temperature scaler not fitted")
        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)
        logits = np.log(y_pred_clipped / (1 - y_pred_clipped))
        scaled_logits = logits / self.T
        calibrated = 1 / (1 + np.exp(-scaled_logits))
        return np.clip(calibrated, 1e-7, 1 - 1e-7)


# ─── Calibration Pipeline ───────────────────────────────────────────────

def fit_calibrator(
    method: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> PlattScaler | IsotonicCalibrator | TemperatureScaler:
    """Fit a calibrator of the specified type.

    Args:
        method: "platt" | "isotonic" | "temperature"
        y_true: binary labels
        y_pred: predicted probabilities

    Returns:
        Fitted calibrator object
    """
    if method == "platt":
        cal = PlattScaler()
    elif method == "isotonic":
        cal = IsotonicCalibrator()
    elif method == "temperature":
        cal = TemperatureScaler()
    else:
        raise ValueError(f"Unknown calibration method: {method}")

    cal.fit(y_true, y_pred)
    return cal


def run_calibration(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Run calibration on OOF predictions.

    Returns:
        Dict with calibrated metrics and calibrator path
    """
    dirs = ensure_dirs(cfg)
    meta = make_meta(cfg, "calibrate")
    meta.save(dirs["base"] / "calibrate_meta.json")

    cal_cfg = cfg.get("calibration", {})
    method = cal_cfg.get("method", "isotonic")
    clip_min = cal_cfg.get("clip_min", 0.005)
    clip_max = cal_cfg.get("clip_max", 0.995)

    # Load OOF predictions
    oof_path = dirs["oof"] / "gbm_oof_predictions.csv"
    if not oof_path.exists():
        log.error(f"OOF predictions not found: {oof_path}")
        return {}

    oof_df = pd.read_csv(oof_path)
    y_true = oof_df["y_true"].values
    y_pred_raw = np.clip(oof_df["y_pred"].values, clip_min, clip_max)

    # Fit calibrator
    log.info(f"Fitting {method} calibrator on {len(y_true)} OOF predictions")
    calibrator = fit_calibrator(method, y_true, y_pred_raw)

    # Apply calibration
    y_pred_calibrated = calibrator.predict(y_pred_raw)

    # Compare metrics
    from sklearn.metrics import log_loss, roc_auc_score

    raw_ll = log_loss(y_true, y_pred_raw)
    cal_ll = log_loss(y_true, y_pred_calibrated)
    raw_auc = roc_auc_score(y_true, y_pred_raw)
    cal_auc = roc_auc_score(y_true, y_pred_calibrated)

    log.info(f"Raw OOF:   LL={raw_ll:.4f} AUC={raw_auc:.4f}")
    log.info(f"Calibrated: LL={cal_ll:.4f} AUC={cal_auc:.4f}")
    log.info(f"LL improvement: {raw_ll - cal_ll:.4f}")

    # Save calibrator
    cal_path = dirs["models"] / f"calibrator_{method}.pkl"
    with open(cal_path, "wb") as f:
        pickle.dump(calibrator, f)
    log.info(f"Calibrator saved: {cal_path}")

    # Save calibrated predictions
    oof_df["y_pred_calibrated"] = y_pred_calibrated
    cal_oof_path = dirs["oof"] / "gbm_oof_calibrated.csv"
    oof_df.to_csv(cal_oof_path, index=False)

    results = {
        "method": method,
        "raw_logloss": raw_ll,
        "calibrated_logloss": cal_ll,
        "raw_auc": raw_auc,
        "calibrated_auc": cal_auc,
        "improvement": raw_ll - cal_ll,
        "calibrator_path": str(cal_path),
    }

    results_path = dirs["base"] / "calibration_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    return results


# ─── Apply Calibration to Test Predictions ──────────────────────────────

def apply_calibration(
    calibrator_path: str | Path,
    y_pred: np.ndarray,
    clip_min: float = 0.005,
    clip_max: float = 0.995,
) -> np.ndarray:
    """Apply a fitted calibrator to test predictions.

    Args:
        calibrator_path: path to saved calibrator pickle
        y_pred: raw predictions
        clip_min/max: clipping bounds

    Returns:
        Calibrated predictions
    """
    with open(calibrator_path, "rb") as f:
        calibrator = pickle.load(f)

    y_pred_clipped = np.clip(y_pred, clip_min, clip_max)
    calibrated = calibrator.predict(y_pred_clipped)
    return np.clip(calibrated, clip_min, clip_max)


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 5: Calibrate OOF predictions")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Path to config YAML")
    parser.add_argument("--method", default=None, help="Override calibration method")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(Path(cfg["paths"]["output_dir"]) / "logs", "calibrate")

    if args.method:
        cfg["calibration"]["method"] = args.method

    run_calibration(cfg)


if __name__ == "__main__":
    main()

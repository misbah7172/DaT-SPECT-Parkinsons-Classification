"""Stage 4: Tabular Classifier (LightGBM / XGBoost).

Trains on concatenated SBR features + CNN embeddings with stratified group k-fold CV.
Produces out-of-fold predictions for calibration and fold-averaged predictions for inference.

Usage:
    python -m pipeline.train_gbm --config pipeline/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .utils import ensure_dirs, load_config, load_labels_and_groups, make_meta, setup_logging

log = logging.getLogger(__name__)


# ─── Feature Matrix Assembly ────────────────────────────────────────────

def build_feature_matrix(
    cfg: Dict[str, Any],
    cnn_embeddings: Optional[pd.DataFrame] = None,
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """Assemble feature matrix from SBR features + CNN embeddings.

    Returns:
        (feature_df, feature_columns, uid_list)
    """
    output_dir = Path(cfg["paths"]["output_dir"])

    # Load SBR features
    sbr_path = output_dir / cfg.get("sbr", {}).get("output_csv", "sbr_features.csv")
    if not sbr_path.exists():
        raise FileNotFoundError(f"SBR features not found: {sbr_path}. Run sbr_features first.")
    sbr_df = pd.read_csv(sbr_path)
    log.info(f"Loaded SBR features: {sbr_df.shape}")

    # Load CNN embeddings if provided
    if cnn_embeddings is not None:
        emb_df = cnn_embeddings
    else:
        emb_path = output_dir / "oof" / "cnn_embeddings.csv"
        if emb_path.exists():
            emb_df = pd.read_csv(emb_path)
            log.info(f"Loaded CNN embeddings: {emb_df.shape}")
        else:
            log.warning("No CNN embeddings found. Using SBR features only.")
            emb_df = None

    # Merge
    if emb_df is not None:
        # Average embeddings across folds/seeds
        emb_cols = [c for c in emb_df.columns if c.startswith("emb_")]
        emb_avg = emb_df.groupby("uid")[emb_cols].mean().reset_index()
        merged = sbr_df.merge(emb_avg, on="uid", how="inner")
    else:
        merged = sbr_df

    # Feature columns (exclude uid, label, group metadata)
    exclude = {"uid", "is_pathologic", "scanner_group", "label", "fold", "seed"}
    feature_cols = [c for c in merged.columns if c not in exclude]

    log.info(f"Final feature matrix: {merged.shape[0]} samples, {len(feature_cols)} features")
    return merged, feature_cols, merged["uid"].tolist()


# ─── Model Training ─────────────────────────────────────────────────────

def train_fold_gbm(
    cfg: Dict[str, Any],
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold: int,
) -> Tuple[Any, Dict[str, float]]:
    """Train GBM model for one fold.

    Returns:
        (trained_model, metrics_dict)
    """
    gbm_cfg = cfg["gbm"]
    model_type = gbm_cfg.get("model", "lightgbm")

    if model_type == "lightgbm":
        import lightgbm as lgb

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "num_leaves": gbm_cfg.get("num_leaves", 31),
            "learning_rate": gbm_cfg.get("learning_rate", 0.05),
            "n_estimators": gbm_cfg.get("n_estimators", 1000),
            "min_child_samples": gbm_cfg.get("min_child_samples", 20),
            "feature_fraction": gbm_cfg.get("feature_fraction", 0.8),
            "bagging_fraction": gbm_cfg.get("bagging_fraction", 0.8),
            "bagging_freq": gbm_cfg.get("bagging_freq", 5),
            "reg_alpha": gbm_cfg.get("reg_alpha", 0.1),
            "reg_lambda": gbm_cfg.get("reg_lambda", 0.1),
            "verbose": -1,
            "n_jobs": cfg.get("env", {}).get("num_threads", 4),
            "seed": 42 + fold,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(gbm_cfg.get("early_stop_rounds", 50)),
                lgb.log_evaluation(100),
            ],
        )

    elif model_type == "xgboost":
        import xgboost as xgb

        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": gbm_cfg.get("num_leaves", 31).bit_length(),
            "learning_rate": gbm_cfg.get("learning_rate", 0.05),
            "n_estimators": gbm_cfg.get("n_estimators", 1000),
            "min_child_samples": gbm_cfg.get("min_child_samples", 20),
            "subsample": gbm_cfg.get("bagging_fraction", 0.8),
            "colsample_bytree": gbm_cfg.get("feature_fraction", 0.8),
            "reg_alpha": gbm_cfg.get("reg_alpha", 0.1),
            "reg_lambda": gbm_cfg.get("reg_lambda", 0.1),
            "verbosity": 0,
            "n_jobs": cfg.get("env", {}).get("num_threads", 4),
            "seed": 42 + fold,
        }

        model = xgb.XGBClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100,
        )

    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Compute metrics on validation set
    val_pred_proba = model.predict_proba(X_val)[:, 1]
    val_pred = model.predict(X_val)

    metrics = compute_metrics(y_val, val_pred_proba, val_pred)
    log.info(f"Fold {fold} {model_type}: LL={metrics['logloss']:.4f} AUC={metrics['auc']:.4f}")

    return model, metrics


def compute_metrics(y_true, y_pred_proba, y_pred) -> Dict[str, float]:
    """Compute evaluation metrics."""
    from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss, f1_score

    metrics = {}
    try:
        metrics["logloss"] = log_loss(y_true, y_pred_proba)
    except Exception:
        metrics["logloss"] = float("inf")

    try:
        metrics["auc"] = roc_auc_score(y_true, y_pred_proba)
    except Exception:
        metrics["auc"] = 0.5

    metrics["brier"] = brier_score_loss(y_true, y_pred_proba)
    metrics["f1"] = f1_score(y_true, y_pred)
    metrics["n_pos"] = int(y_true.sum())
    metrics["n_neg"] = int(len(y_true) - y_true.sum())

    return metrics


# ─── Cross-Validation Runner ────────────────────────────────────────────

def run_gbm_cv(
    cfg: Dict[str, Any],
    cnn_embeddings: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Run full GBM training with cross-validation.

    Returns:
        Dict with oof_predictions, models, metrics
    """
    dirs = ensure_dirs(cfg)
    meta = make_meta(cfg, "train_gbm")
    meta.save(dirs["base"] / "gbm_meta.json")

    # Build feature matrix
    merged, feature_cols, uids = build_feature_matrix(cfg, cnn_embeddings)

    # Load labels
    data_dir = Path(cfg["paths"]["data_dir"])
    df_labels, _ = load_labels_and_groups(data_dir)
    label_map = dict(zip(df_labels["uid"], df_labels["is_pathologic"]))

    y = np.array([label_map.get(u, 0) for u in uids])
    X = merged[feature_cols].values

    # Handle NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # Cross-validation
    cv_cfg = cfg["cv"]
    n_folds = cv_cfg.get("n_folds", 5)
    group_by = cv_cfg.get("group_by", "scanner_group")

    # Get groups for StratifiedGroupKFold
    groups = None
    if group_by and group_by in merged.columns:
        groups = merged[group_by].values
    elif group_by == "scanner_group" and "scanner_group" in merged.columns:
        groups = merged["scanner_group"].values

    if groups is not None:
        # Map groups to integer codes
        unique_groups = list(set(groups))
        group_map = {g: i for i, g in enumerate(unique_groups)}
        group_codes = np.array([group_map[g] for g in groups])
        skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=cv_cfg.get("random_state", 42))
        split_args = (X, y, group_codes)
    else:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=cv_cfg.get("random_state", 42))
        split_args = (X, y)

    # OOF predictions
    oof_preds = np.zeros(len(y))
    oof_labels = y.copy()
    models = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(*split_args)):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold}/{n_folds-1}")
        log.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model, metrics = train_fold_gbm(cfg, X_train, y_train, X_val, y_val, fold)

        # OOF predictions
        oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
        models.append(model)
        fold_metrics.append(metrics)

        # Save model
        model_path = dirs["models"] / f"gbm_fold{fold}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        log.info(f"  Model saved: {model_path}")

    # Overall OOF metrics
    overall_metrics = compute_metrics(oof_labels, oof_preds, (oof_preds > 0.5).astype(int))
    log.info(f"\n{'='*60}")
    log.info(f"OVERALL OOF: LL={overall_metrics['logloss']:.4f} AUC={overall_metrics['auc']:.4f}")

    # Save OOF predictions
    oof_df = pd.DataFrame({
        "uid": uids,
        "y_true": oof_labels,
        "y_pred": oof_preds,
        "fold": [-1] * len(uids),  # placeholder
    })
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(*split_args)):
        oof_df.loc[val_idx, "fold"] = fold_idx

    oof_path = dirs["oof"] / "gbm_oof_predictions.csv"
    oof_df.to_csv(oof_path, index=False)
    log.info(f"OOF predictions saved: {oof_path}")

    # Save feature importances
    importances = _compute_feature_importance(models, feature_cols)
    imp_path = dirs["base"] / "feature_importances.csv"
    importances.to_csv(imp_path, index=False)

    # Save config + metrics
    results = {
        "config": cfg,
        "overall_metrics": overall_metrics,
        "fold_metrics": fold_metrics,
        "feature_cols": feature_cols,
        "n_models": len(models),
    }
    results_path = dirs["base"] / "gbm_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results


def _compute_feature_importance(models: List, feature_cols: List[str]) -> pd.DataFrame:
    """Compute average feature importance across folds."""
    importances = np.zeros(len(feature_cols))
    for model in models:
        if hasattr(model, "feature_importances_"):
            importances += model.feature_importances_
    importances /= len(models)

    df = pd.DataFrame({
        "feature": feature_cols,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    return df


# ─── Inference ──────────────────────────────────────────────────────────

def predict_gbm(
    cfg: Dict[str, Any],
    feature_matrix: pd.DataFrame,
    feature_cols: List[str],
    model_dir: Optional[Path] = None,
) -> np.ndarray:
    """Predict using fold-averaged GBM models.

    Returns:
        Array of averaged predictions
    """
    if model_dir is None:
        model_dir = Path(cfg["paths"]["output_dir"]) / "models"

    X = feature_matrix[feature_cols].values
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    preds = []
    for fold in range(cfg["cv"].get("n_folds", 5)):
        model_path = model_dir / f"gbm_fold{fold}.pkl"
        if not model_path.exists():
            log.warning(f"Missing model: {model_path}")
            continue
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        preds.append(model.predict_proba(X)[:, 1])

    if not preds:
        raise FileNotFoundError("No models found for prediction")

    return np.mean(preds, axis=0)


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 4: Train GBM classifier on SBR + CNN features")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Path to config YAML")
    parser.add_argument("--no-cnn", action="store_true", help="Skip CNN embeddings, use SBR only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(Path(cfg["paths"]["output_dir"]) / "logs", "train_gbm")

    cnn_embeddings = None
    if not args.no_cnn:
        emb_path = Path(cfg["paths"]["output_dir"]) / "oof" / "cnn_embeddings.csv"
        if emb_path.exists():
            cnn_embeddings = pd.read_csv(emb_path)

    run_gbm_cv(cfg, cnn_embeddings)


if __name__ == "__main__":
    main()

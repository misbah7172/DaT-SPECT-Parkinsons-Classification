"""V24 Joint blend weight + temperature optimization.

Replaces v23's two-step sequential optimization with a joint grid search
over blend weight AND temperature, minimizing log loss directly on OOF predictions.

Also includes Platt scaling (2-parameter logistic regression) as an alternative
to temperature-only scaling.
"""
import numpy as np
from sklearn.metrics import log_loss
from sklearn.linear_model import LogisticRegression


def joint_blend_temp_search(
    p_deep: np.ndarray,
    p_sbr: np.ndarray,
    y_true: np.ndarray,
    w_range=None,
    t_range=None,
    n_w=50,
    n_t=50,
):
    """Joint grid search over blend weight and temperature.

    Args:
        p_deep: OOF predictions from deep stream
        p_sbr: OOF predictions from SBR stream
        y_true: binary labels
        w_range: (w_min, w_max) for blend weight search
        t_range: (t_min, t_max) for temperature search
        n_w: number of blend weight points
        n_t: number of temperature points

    Returns:
        dict with best_w, best_t, best_ll, grid_results
    """
    if w_range is None:
        w_range = (0.70, 0.95)
    if t_range is None:
        t_range = (0.50, 1.20)

    weights = np.linspace(w_range[0], w_range[1], n_w)
    temps = np.linspace(t_range[0], t_range[1], n_t)

    best_ll = float("inf")
    best_w = 0.89
    best_t = 0.77
    grid_results = []

    for w in weights:
        for T in temps:
            p_blend = w * p_deep + (1 - w) * p_sbr
            p_blend = np.clip(p_blend, 1e-6, 1 - 1e-6)
            lg = np.log(p_blend / (1.0 - p_blend))
            p_cal = 1.0 / (1.0 + np.exp(-lg / T))
            p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)

            try:
                ll = log_loss(y_true, p_cal)
            except Exception:
                ll = float("inf")

            grid_results.append({"w": w, "T": T, "ll": ll})

            if ll < best_ll:
                best_ll = ll
                best_w = w
                best_t = T

    return {
        "best_w": float(best_w),
        "best_t": float(best_t),
        "best_ll": float(best_ll),
        "grid_results": grid_results,
    }


def fit_platt_scaler(
    p_blend: np.ndarray,
    y_true: np.ndarray,
) -> dict:
    """Fit 2-parameter Platt scaling (a·logit + b).

    Args:
        p_blend: blended predictions (before calibration)
        y_true: binary labels

    Returns:
        dict with a, b, calibrated predictions, and log loss
    """
    p = np.clip(p_blend, 1e-7, 1 - 1e-7)
    logits = np.log(p / (1 - p)).reshape(-1, 1)

    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(logits, y_true)

    a = float(lr.coef_[0][0])
    b = float(lr.intercept_[0])

    calibrated = lr.predict_proba(logits)[:, 1]
    calibrated = np.clip(calibrated, 1e-7, 1 - 1e-7)

    ll = log_loss(y_true, calibrated)

    return {
        "a": a,
        "b": b,
        "calibrated": calibrated,
        "logloss": float(ll),
    }


def fit_per_group_calibration(
    p_blend: np.ndarray,
    y_true: np.ndarray,
    scanner_groups: np.ndarray,
) -> dict:
    """Fit per-scanner-group Platt calibration.

    Groups scans by resolution tier and fits separate Platt scalers.

    Args:
        p_blend: blended predictions
        y_true: binary labels
        scanner_groups: scanner group strings

    Returns:
        dict with group_calibrators, group_metrics, calibrated predictions
    """
    # Cluster scanner groups into resolution tiers
    def _resolution_tier(group_str):
        try:
            parts = group_str.split("x")
            sx = float(parts[0])
            if sx <= 2.5:
                return "high_res"    # ≤2.5mm
            elif sx <= 3.5:
                return "mid_res"     # 2.5-3.5mm
            else:
                return "low_res"     # ≥3.5mm
        except Exception:
            return "unknown"

    tiers = np.array([_resolution_tier(g) for g in scanner_groups])
    unique_tiers = np.unique(tiers)

    group_calibrators = {}
    group_metrics = {}
    calibrated = np.zeros_like(p_blend)

    for tier in unique_tiers:
        mask = tiers == tier
        if mask.sum() < 10:
            # Not enough data for group-level calibration
            calibrated[mask] = p_blend[mask]
            group_metrics[tier] = {"n": int(mask.sum()), "method": "passthrough"}
            continue

        result = fit_platt_scaler(p_blend[mask], y_true[mask])
        calibrated[mask] = result["calibrated"]
        group_calibrators[tier] = {"a": result["a"], "b": result["b"]}
        group_metrics[tier] = {
            "n": int(mask.sum()),
            "logloss": result["logloss"],
            "a": result["a"],
            "b": result["b"],
        }

    overall_ll = log_loss(y_true, np.clip(calibrated, 1e-7, 1 - 1e-7))

    return {
        "group_calibrators": group_calibrators,
        "group_metrics": group_metrics,
        "calibrated": calibrated,
        "overall_logloss": float(overall_ll),
    }


def fit_meta_stacker(
    oof_preds: dict,
    y_true: np.ndarray,
    n_folds: int = 5,
) -> dict:
    """Train a cross-validated meta-learner on OOF predictions.

    Replaces fixed linear blend with a learned stacker.

    Args:
        oof_preds: dict of {model_name: oof_predictions_array}
        y_true: binary labels
        n_folds: CV folds for stacking

    Returns:
        dict with stacker, oof_stacked predictions, log loss
    """
    from sklearn.model_selection import StratifiedKFold

    model_names = sorted(oof_preds.keys())
    X = np.column_stack([oof_preds[name] for name in model_names])
    y = y_true.copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_stacked = np.zeros(len(y))

    # Train stacker on each fold
    stackers = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Use logistic regression as stacker
        lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
        lr.fit(X_train, y_train)

        oof_stacked[val_idx] = lr.predict_proba(X_val)[:, 1]
        stackers.append(lr)

        fold_ll = log_loss(y_val, np.clip(oof_stacked[val_idx], 1e-7, 1 - 1e-7))
        fold_metrics.append({"fold": fold, "logloss": float(fold_ll)})

    overall_ll = log_loss(y, np.clip(oof_stacked, 1e-7, 1 - 1e-7))

    # Also fit a final stacker on all data
    final_stacker = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    final_stacker.fit(X, y)

    # Get feature weights
    weights = {}
    for i, name in enumerate(model_names):
        weights[name] = float(final_stacker.coef_[0][i])

    return {
        "stacker": final_stacker,
        "fold_stackers": stackers,
        "oof_stacked": oof_stacked,
        "logloss": float(overall_ll),
        "fold_metrics": fold_metrics,
        "model_names": model_names,
        "weights": weights,
    }


def verify_probability_space_aggregation():
    """Verify that aggregation happens in probability space, not logit space.

    This is a sanity check — probability-space averaging is correct for log loss.
    """
    # Demonstration with synthetic data
    p1, p2 = 0.8, 0.3
    w = 0.7

    # Probability space (correct for log loss)
    p_prob = w * p1 + (1 - w) * p2

    # Logit space (miscalibrated by construction)
    l1, l2 = np.log(p1 / (1 - p1)), np.log(p2 / (1 - p2))
    l_blend = w * l1 + (1 - w) * l2
    p_logit = 1 / (1 + np.exp(-l_blend))

    # They're different — probability space is correct
    assert abs(p_prob - p_logit) > 0.01, "Probability and logit aggregation should differ"

    return {
        "probability_space": p_prob,
        "logit_space": p_logit,
        "difference": abs(p_prob - p_logit),
        "note": "Probability-space averaging is correct for log loss minimization",
    }

"""Calibration module. All calibration must use inner validation fold only."""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from scipy.optimize import minimize_scalar


def platt_scale(y_trn_prob, y_trn, y_val_prob):
    """Platt (sigmoid) calibration. Fit on train, apply to val."""
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(y_trn_prob.reshape(-1, 1), y_trn)
    return lr.predict_proba(y_val_prob.reshape(-1, 1))[:, 1]


def isotonic_calibrate(y_trn_prob, y_trn, y_val_prob):
    """Isotonic regression calibration. Fit on train, apply to val."""
    ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
    ir.fit(y_trn_prob, y_trn)
    return np.clip(ir.predict(y_val_prob), 1e-7, 1 - 1e-7)


def temperature_scale(y_trn_prob, y_trn, y_val_prob):
    """Temperature scaling. Fit T on train, apply to val."""
    def neg_ll(T):
        logits = np.log(np.clip(y_trn_prob, 1e-7, 1 - 1e-7) /
                        (1 - np.clip(y_trn_prob, 1e-7, 1 - 1e-7)))
        scaled = 1 / (1 + np.exp(-logits / T))
        from sklearn.metrics import log_loss
        return log_loss(y_trn, np.clip(scaled, 1e-7, 1 - 1e-7))

    res = minimize_scalar(neg_ll, bounds=(0.3, 3.0), method='bounded')
    T_opt = res.x

    logits = np.log(np.clip(y_val_prob, 1e-7, 1 - 1e-7) /
                    (1 - np.clip(y_val_prob, 1e-7, 1 - 1e-7)))
    return 1 / (1 + np.exp(-logits / T_opt))


def power_sharpen(y_prob, gamma):
    """Power sharpening: p^gamma / (p^gamma + (1-p)^gamma)."""
    p = np.clip(y_prob, 1e-7, 1 - 1e-7)
    num = p ** gamma
    denom = num + (1 - p) ** gamma
    return np.clip(num / denom, 1e-7, 1 - 1e-7)


def fit_power_gamma(y_trn_prob, y_trn):
    """Find optimal gamma on training fold predictions."""
    from sklearn.metrics import log_loss

    def neg_ll(g):
        sharpened = power_sharpen(y_trn_prob, g)
        return log_loss(y_trn, sharpened)

    res = minimize_scalar(neg_ll, bounds=(0.5, 2.0), method='bounded')
    return res.x

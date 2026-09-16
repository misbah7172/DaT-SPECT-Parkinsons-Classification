"""Metrics module. All evaluation metrics for the competition."""
import numpy as np
from sklearn.metrics import (
    roc_auc_score, log_loss, brier_score_loss,
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix
)


def compute_all_metrics(y_true, y_prob, prefix=''):
    """Compute all required metrics. Returns dict."""
    y_prob = np.clip(y_prob, 1e-7, 1 - 1e-7)
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = {
        f'{prefix}auc': roc_auc_score(y_true, y_prob),
        f'{prefix}logloss': log_loss(y_true, y_prob),
        f'{prefix}brier': brier_score_loss(y_true, y_prob),
        f'{prefix}accuracy': accuracy_score(y_true, y_pred),
        f'{prefix}f1': f1_score(y_true, y_pred),
        f'{prefix}precision': precision_score(y_true, y_pred, zero_division=0),
        f'{prefix}recall': recall_score(y_true, y_pred, zero_division=0),
    }

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    metrics[f'{prefix}tn'] = int(tn)
    metrics[f'{prefix}fp'] = int(fp)
    metrics[f'{prefix}fn'] = int(fn)
    metrics[f'{prefix}tp'] = int(tp)

    return metrics


def compute_site_metrics(y_true, y_prob, groups):
    """Per-site metrics. Returns dict of site -> metrics."""
    sites = np.unique(groups)
    site_metrics = {}
    for site in sites:
        mask = groups == site
        if mask.sum() < 5:
            continue
        yt, yp = y_true[mask], y_prob[mask]
        site_metrics[site] = {
            'n': int(mask.sum()),
            'pos_count': int(yt.sum()),
            'neg_count': int((1 - yt).sum()),
            'pos_rate': float(yt.mean()),
            'mean_pred': float(yp.mean()),
            'std_pred': float(yp.std()),
        }
        if len(np.unique(yt)) > 1:
            site_metrics[site]['auc'] = roc_auc_score(yt, yp)
            site_metrics[site]['logloss'] = log_loss(yt, np.clip(yp, 1e-7, 1 - 1e-7))
            site_metrics[site]['brier'] = brier_score_loss(yt, yp)
        else:
            site_metrics[site]['auc'] = float('nan')
            site_metrics[site]['logloss'] = float('nan')
            site_metrics[site]['brier'] = float('nan')
    return site_metrics


def bootstrap_ci(y_true, y_prob, metric_fn, n_bootstrap=1000, ci=0.95, seed=42):
    """Bootstrap confidence interval for any metric."""
    rng = np.random.RandomState(seed)
    scores = []
    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        yt, yp = y_true[idx], y_prob[idx]
        if len(np.unique(yt)) < 2:
            continue
        scores.append(metric_fn(yt, np.clip(yp, 1e-7, 1 - 1e-7)))
    scores = np.array(scores)
    lo = np.percentile(scores, (1 - ci) / 2 * 100)
    hi = np.percentile(scores, (1 + ci) / 2 * 100)
    return float(scores.mean()), float(lo), float(hi)

"""Validation module. Site-aware and random CV splits."""
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def site_aware_cv(n_splits=5, shuffle=True, random_state=42):
    """StratifiedGroupKFold: groups = scanner site. No group leaks across folds."""
    return StratifiedGroupKFold(n_splits=n_splits, shuffle=shuffle,
                                random_state=random_state)


def random_cv(n_splits=5, shuffle=True, random_state=42):
    """Standard StratifiedKFold (for comparison only)."""
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle,
                           random_state=random_state)


def get_fold_splits(X, y, groups, cv_type='site_aware', n_splits=5, random_state=42):
    """Return list of (train_idx, val_idx) tuples."""
    if cv_type == 'site_aware':
        cv = site_aware_cv(n_splits=n_splits, random_state=random_state)
        return list(cv.split(X, y, groups))
    elif cv_type == 'random':
        cv = random_cv(n_splits=n_splits, random_state=random_state)
        return list(cv.split(X, y))
    else:
        raise ValueError(f"Unknown cv_type: {cv_type}")


def print_fold_stats(splits, y, groups, cv_type):
    """Print per-fold class balance and group info."""
    print(f"\n  {cv_type} CV: {len(splits)} folds")
    for i, (trn_idx, val_idx) in enumerate(splits):
        y_trn, y_val = y[trn_idx], y[val_idx]
        g_val = groups[val_idx]
        print(f"    Fold {i}: trn={len(trn_idx)} val={len(val_idx)} "
              f"val_pos={y_val.mean():.3f} "
              f"val_groups={np.unique(g_val).tolist()}")

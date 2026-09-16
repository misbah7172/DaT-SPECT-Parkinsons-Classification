"""Restore original SBR models (without spacing features)."""
import os, time, pickle, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import log_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb

OUT_W = r'E:\DaT\submission_v26\weights'
LOG = r'E:\DaT\submission_v26\train_sbr_v2.log'

def log(m):
    with open(LOG, 'a') as f:
        f.write(f'[{time.strftime("%H:%M:%S")}] {m}\n')
    print(m, flush=True)

def get_features():
    labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv').set_index('uid')
    site = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv').set_index('uid')
    geom = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv').set_index('uid')
    sbr = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv').drop(columns=['label'], errors='ignore').set_index('uid')
    merged = labels.join(site).join(geom).loc[sbr.index]
    y = merged['is_pathologic'].values
    groups = merged['pseudo_site'].astype(str).values
    PVE_REF = 15.625
    fac = (PVE_REF / merged['voxel_vol'].values) ** (1.0 / 3.0)
    sbr_cols = [c for c in sbr.columns if c.startswith('sbr_')]
    for c in sbr_cols:
        sbr[c] = sbr[c] * fac
    eps = 1e-6
    if 'sbr_left_putamen' in sbr.columns and 'sbr_right_putamen' in sbr.columns:
        L, R = sbr['sbr_left_putamen'].values, sbr['sbr_right_putamen'].values
        sbr['lr_put_asym'] = (L - R) / (L + R + eps)
        sbr['lr_put_ratio'] = np.minimum(L, R) / (np.maximum(L, R) + eps)
        sbr['lr_put_sum'] = L + R
        sbr['lr_put_diff'] = L - R
    if 'sbr_left_caudate' in sbr.columns and 'sbr_right_caudate' in sbr.columns:
        L, R = sbr['sbr_left_caudate'].values, sbr['sbr_right_caudate'].values
        sbr['lr_cau_asym'] = (L - R) / (L + R + eps)
        sbr['lr_cau_ratio'] = np.minimum(L, R) / (np.maximum(L, R) + eps)
        sbr['lr_cau_sum'] = L + R
    if 'sbr_left_putamen_post' in sbr.columns and 'sbr_left_putamen_ant' in sbr.columns:
        sbr['lp_post_ant'] = sbr['sbr_left_putamen_post'].values / (sbr['sbr_left_putamen_ant'].values + eps)
    if 'sbr_right_putamen_post' in sbr.columns and 'sbr_right_putamen_ant' in sbr.columns:
        sbr['rp_post_ant'] = sbr['sbr_right_putamen_post'].values / (sbr['sbr_right_putamen_ant'].values + eps)
    if 'sbr_left_putamen' in sbr.columns and 'sbr_left_caudate' in sbr.columns:
        sbr['lp_cr'] = sbr['sbr_left_putamen'].values / (sbr['sbr_left_caudate'].values + eps)
    if 'sbr_right_putamen' in sbr.columns and 'sbr_right_caudate' in sbr.columns:
        sbr['rp_cr'] = sbr['sbr_right_putamen'].values / (sbr['sbr_right_caudate'].values + eps)
    if 'sbr_left_total' in sbr.columns and 'sbr_right_total' in sbr.columns:
        L, R = sbr['sbr_left_total'].values, sbr['sbr_right_total'].values
        sbr['total_str'] = L + R
        sbr['lr_total_asym'] = (L - R) / (L + R + eps)
    cols = list(sbr.columns)
    a = sbr.values.astype(np.float64)
    X = np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1), nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, groups, cols

def train_oof(X, y, groups, name, model_factory):
    skf = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for fold, (tr, va) in enumerate(skf.split(X, y, groups)):
        sc = StandardScaler()
        Xtr = sc.fit_transform(X[tr])
        Xva_s = sc.transform(X[va])
        m = model_factory()
        if hasattr(m, 'predict_proba'):
            m.fit(Xtr, y[tr])
            oof[va] = m.predict_proba(Xva_s)[:, 1]
        else:
            m.fit(Xtr, y[tr])
            oof[va] = m.decision_function(Xva_s)
            oof[va] = 1 / (1 + np.exp(-oof[va]))
    ll = log_loss(y, np.clip(oof, 1e-6, 1-1e-6))
    auc = roc_auc_score(y, oof)
    log(f'  {name}: AUC={auc:.4f} LL={ll:.4f}')
    return oof, ll, auc

if __name__ == '__main__':
    log('=== SBR restore (original features) ===')
    X, y, groups, cols = get_features()
    log(f'Features: {X.shape[1]} cols')
    models = {
        'sbr_lr': lambda: LogisticRegression(C=0.1, max_iter=1000, solver='lbfgs'),
        'sbr_xgb': lambda: xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', verbosity=0),
        'sbr_lgb': lambda: lgb.LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbose=-1),
        'sbr_et': lambda: ExtraTreesClassifier(n_estimators=300, max_depth=None, min_samples_leaf=5, n_jobs=-1),
        'sbr_ridge': lambda: RidgeClassifier(alpha=1.0),
    }
    oof_arr = {}
    for name, factory in models.items():
        oof_arr[name], _, _ = train_oof(X, y, groups, name, factory)
    mean_oof = np.mean([oof_arr[k] for k in oof_arr], axis=0)
    ll = log_loss(y, np.clip(mean_oof, 1e-6, 1-1e-6))
    auc = roc_auc_score(y, mean_oof)
    log(f'MEAN OOF: AUC={auc:.4f} LL={ll:.4f}')
    sc = StandardScaler()
    Xall = sc.fit_transform(X)
    for name, factory in models.items():
        m = factory()
        if hasattr(m, 'predict_proba'):
            m.fit(Xall, y)
        else:
            m.fit(Xall, y)
        pickle.dump(m, open(os.path.join(OUT_W, f'{name}.pkl'), 'wb'))
        log(f'  Saved {name}.pkl')
    pickle.dump(sc, open(os.path.join(OUT_W, 'sbr_full_scaler.pkl'), 'wb'))
    json.dump(cols, open(os.path.join(OUT_W, 'sbr_full_cols.json'), 'w'))
    log(f'FINAL: AUC={auc:.4f} LL={ll:.4f}')
    log('=== Done ===')

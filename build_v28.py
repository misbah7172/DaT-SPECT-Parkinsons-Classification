"""Build v28: v23 CNN weights + improved SBR (physical features + more models + more seeds).
Uses v23's exact CNN inference pipeline. Only improves the SBR tabular stream."""
import os, json, pickle, shutil
import numpy as np
import pandas as pd
from scipy.special import expit, logit
from scipy.optimize import minimize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import lightgbm as lgb

# Load data
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
groups = LabelEncoder().fit_transform(groups_raw.astype(str))

# Load SBR features
sbr_df = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv')
phys_df = pd.read_csv(r'E:\DaT\Dataset\phys_features_train.csv') if os.path.exists(r'E:\DaT\Dataset\phys_features_train.csv') else None
atlas_df = pd.read_csv(r'E:\DaT\Dataset\atlas_features_train.csv') if os.path.exists(r'E:\DaT\Dataset\atlas_features_train.csv') else None
voxel_df = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
striatal_df = pd.read_csv(r'E:\DaT\striatal_features_from_crop.csv') if os.path.exists(r'E:\DaT\striatal_features_from_crop.csv') else None

print(f"SBR features: {sbr_df.shape}")
if phys_df is not None: print(f"Physical features: {phys_df.shape}")
if atlas_df is not None: print(f"Atlas features: {atlas_df.shape}")
if striatal_df is not None: print(f"Striatal features: {striatal_df.shape}")

# ============ FEATURE ENGINEERING ============
# Merge all features
uid_col = 'uid'
label_col = 'label'

# Drop label and uid from features
sbr_drop = [uid_col, label_col] if label_col in sbr_df.columns else [uid_col]
sbr_cols = [c for c in sbr_df.columns if c not in sbr_drop and sbr_df[c].dtype != object]
X_sbr = sbr_df[sbr_cols].fillna(0).values.astype(np.float64)

# PVE correction on SBR features
PVE_REF = 15.625
voxel_vols = voxel_df['voxel_vol'].values
fac = (PVE_REF / np.clip(voxel_vols, 1, None)) ** (1/3)
X_sbr_pve = X_sbr.copy()
for i in range(min(89, X_sbr.shape[1])):
    X_sbr_pve[:, i] *= fac

# Derived features (same as add_bio in main.py)
eps = 1e-6
if 'sbr_left_putamen' in sbr_df.columns:
    L = sbr_df['sbr_left_putamen'].values
    R = sbr_df['sbr_right_putamen'].values
    derived = np.column_stack([
        (L - R) / (L + R + eps),  # lr_put_asym
        np.minimum(L, R) / (np.maximum(L, R) + eps),  # lr_put_ratio
        L + R,  # lr_put_sum
        L - R,  # lr_put_diff
    ])
    Lc = sbr_df['sbr_left_caudate'].values
    Rc = sbr_df['sbr_right_caudate'].values
    derived = np.column_stack([derived,
        (Lc - Rc) / (Lc + Rc + eps),  # lr_cau_asym
        np.minimum(Lc, Rc) / (np.maximum(Lc, Rc) + eps),  # lr_cau_ratio
        Lc + Rc,  # lr_cau_sum
    ])
    if 'sbr_left_putamen_post' in sbr_df.columns:
        derived = np.column_stack([derived,
            sbr_df['sbr_left_putamen_post'].values / (sbr_df['sbr_left_putamen_ant'].values + eps),
            sbr_df['sbr_right_putamen_post'].values / (sbr_df['sbr_right_putamen_ant'].values + eps),
            L / (Lc + eps),
            R / (Rc + eps),
            (sbr_df.get('sbr_left_total', pd.Series(np.zeros(len(y)))).values if 'sbr_left_total' not in sbr_df.columns else sbr_df['sbr_left_total'].values) +
            (sbr_df.get('sbr_right_total', pd.Series(np.zeros(len(y)))).values if 'sbr_right_total' not in sbr_df.columns else sbr_df['sbr_right_total'].values),
        ])
    X_bio = derived
else:
    X_bio = np.zeros((len(y), 1))

# log1p augmentation (same as v23)
X_full = np.nan_to_num(np.concatenate([X_sbr_pve, X_bio, np.log1p(np.clip(np.concatenate([X_sbr_pve, X_bio], axis=1), 0, None))], axis=1), nan=0.0, posinf=0.0, neginf=0.0)

print(f"Full feature matrix: {X_full.shape}")

# Physical features (if available)
if phys_df is not None:
    phys_uid = phys_df[uid_col].values
    phys_cols = [c for c in phys_df.columns if c not in [uid_col, label_col] and phys_df[c].dtype != object]
    # Align by UID
    uid_to_idx = {uid: i for i, uid in enumerate(sbr_df[uid_col].values)}
    X_phys = np.zeros((len(sbr_df), len(phys_cols)))
    for i, uid in enumerate(phys_uid):
        if uid in uid_to_idx:
            X_phys[uid_to_idx[uid]] = phys_df.iloc[i][phys_cols].fillna(0).values
    X_phys_pve = X_phys.copy()
    for i in range(min(89, X_phys.shape[1])):
        X_phys_pve[:, i] *= fac
    X_combined = np.concatenate([X_full, X_phys_pve, np.log1p(np.clip(X_phys_pve, 0, None))], axis=1)
    print(f"With physical features: {X_combined.shape}")
else:
    X_combined = X_full

# ============ TRAIN IMPROVED SBR MODELS ============
print(f"\n{'='*60}")
print(f"TRAINING IMPROVED SBR MODELS")
print(f"{'='*60}")

SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

model_configs = [
    ('lr', lambda: LogisticRegression(C=1.31, max_iter=1000)),
    ('ridge', lambda: RidgeClassifier(alpha=0.1)),
    ('et', lambda: ExtraTreesClassifier(n_estimators=600, max_depth=9, min_samples_leaf=5, random_state=42)),
    ('xgb', lambda: xgb.XGBClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, verbosity=0, random_state=42)),
    ('lgb', lambda: lgb.LGBMClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8, min_child_weight=5, verbose=-1, random_state=42)),
    ('hgb', lambda: GradientBoostingClassifier(n_estimators=400, max_depth=4, learning_rate=0.03, min_samples_leaf=5, random_state=42)),
]

# MI selection
mi = mutual_info_classif(X_combined, y, random_state=42)
TOP_K = 50
top_idx = np.argsort(mi)[-TOP_K:]
X_selected = X_combined[:, top_idx]
print(f"MI top-{TOP_K} features selected from {X_combined.shape[1]}")

# Train all models with 5 seeds
all_oof_streams = {}  # stream_name -> oof predictions

for model_name, model_factory in model_configs:
    for seed in SEEDS:
        stream_name = f"{model_name}_s{seed}"
        oof = np.zeros(len(y))
        
        for trn_idx, val_idx in sgkf.split(np.zeros(len(y)), y, groups):
            sc = StandardScaler()
            
            if model_name in ('lr', 'ridge'):
                X_trn = sc.fit_transform(X_combined[trn_idx])
                X_val = sc.transform(X_combined[val_idx])
            else:
                X_trn = sc.fit_transform(X_selected[trn_idx])
                X_val = sc.transform(X_selected[val_idx])
            
            clf = model_factory()
            
            if model_name == 'ridge':
                clf.fit(X_trn, y[trn_idx])
                oof[val_idx] = expit(clf.decision_function(X_val))
            else:
                clf.fit(X_trn, y[trn_idx])
                oof[val_idx] = clf.predict_proba(X_val)[:, 1]
        
        auc = roc_auc_score(y, oof)
        ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
        all_oof_streams[stream_name] = oof
        
print(f"\nStream OOF results:")
for name, oof in all_oof_streams.items():
    print(f"  {name:20s}: AUC={roc_auc_score(y, oof):.4f}, LL={log_loss(y, np.clip(oof,1e-7,1-1e-7)):.4f}")

# ============ OPTIMIZE BLEND WEIGHTS ============
print(f"\n{'='*60}")
print(f"OPTIMIZING BLEND WEIGHTS (all SBR streams)")
print(f"{'='*60}")

stream_names = list(all_oof_streams.keys())
stream_oofs = np.column_stack([all_oof_streams[n] for n in stream_names])
n_streams = len(stream_names)

# Try equal weight first
equal_oof = stream_oofs.mean(axis=1)
print(f"Equal weight ({n_streams} streams): AUC={roc_auc_score(y, equal_oof):.4f}, LL={log_loss(y, np.clip(equal_oof,1e-7,1-1e-7)):.4f}")

# Optimize weights
def stream_blend_loss(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()  # normalize
    blend = stream_oofs @ weights
    blend = np.clip(blend, 1e-7, 1-1e-7)
    return log_loss(y, blend)

from scipy.optimize import minimize as sp_minimize

x0 = np.ones(n_streams) / n_streams
result = sp_minimize(stream_blend_loss, x0, method='Nelder-Mead', options={'maxiter': 5000})
opt_weights = result.x / result.x.sum()
opt_oof = stream_oofs @ opt_weights
print(f"Optimized: AUC={roc_auc_score(y, opt_oof):.4f}, LL={log_loss(y, np.clip(opt_oof,1e-7,1-1e-7)):.4f}")

# Show top weights
sorted_idx = np.argsort(opt_weights)[::-1]
print(f"\nTop stream weights:")
for i in sorted_idx[:10]:
    print(f"  {stream_names[i]:20s}: {opt_weights[i]:.4f}")

# ============ REFIT ON FULL DATA ============
print(f"\n{'='*60}")
print(f"REFITTING ON FULL DATA FOR SUBMISSION")
print(f"{'='*60}")

v28_dir = r'E:\DaT\submission_v28'
os.makedirs(os.path.join(v28_dir, 'weights'), exist_ok=True)

# Train each model on full data
for model_name, model_factory in model_configs:
    sc = StandardScaler()
    if model_name in ('lr', 'ridge'):
        X_all = sc.fit_transform(X_combined)
    else:
        X_all = sc.fit_transform(X_selected)
    
    clf = model_factory()
    if model_name == 'ridge':
        clf.fit(X_all, y)
    else:
        clf.fit(X_all, y)
    
    # Save
    with open(os.path.join(v28_dir, 'weights', f'sbr_full_{model_name}.pkl'), 'wb') as f:
        pickle.dump(clf, f)
    print(f"  Saved sbr_full_{model_name}.pkl")

# Save scaler and selection
sc_full = StandardScaler()
sc_full.fit_transform(X_combined)
with open(os.path.join(v28_dir, 'weights', 'sbr_full_scaler.pkl'), 'wb') as f:
    pickle.dump(sc_full, f)

with open(os.path.join(v28_dir, 'weights', 'sbr_full_sel.pkl'), 'wb') as f:
    pickle.dump(top_idx, f)

# Save column names
all_cols = sbr_cols + [f'bio_{i}' for i in range(X_bio.shape[1])]
if phys_df is not None:
    all_cols += [f'phys_{c}' for c in phys_cols]
with open(os.path.join(v28_dir, 'weights', 'sbr_full_cols.json'), 'w') as f:
    json.dump(all_cols, f)

print(f"  Saved scaler, selection, columns")

# Save v23's original weights for reference
print(f"\nNote: v28 uses v23's CNN weights (already in submission_v23/weights/)")
print(f"You need to copy v23's deep_*.pt files to submission_v28/weights/")

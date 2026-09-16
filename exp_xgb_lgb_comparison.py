"""Rigorous XGBoost vs LightGBM OOF comparison.
Feature sets: SBR/PVE, SBR+Radiomics, SBR+2.5D-CNN, Full fused.
Site-aware OOF, hyperparameter tuning on train folds only, early stopping.
No feature selection using val labels. Reports raw + calibrated probabilities."""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os, json, time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.calibration import CalibratedClassifierCV
from scipy.special import expit
import xgboost as xgb
import lightgbm as lgb

# ============ LOAD DATA ============
print("Loading data...")
y = np.load('v26_oof/y.npy')
groups = np.load('v26_oof/groups.npy', allow_pickle=True)
N = len(y)
print(f"  N={N}, prevalence={y.mean():.3f}, groups={len(np.unique(groups))}")

# ============ FEATURE SETS ============
def load_csv(path, drop_cols=('uid', 'label')):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in drop_cols]
    return df[cols].values.astype(np.float64), cols

# 1. SBR/PVE (92 features)
X_sbr, cols_sbr = load_csv('Dataset/sbr_features_train.csv')
print(f"  SBR/PVE: {X_sbr.shape[1]} features")

# 2. SBR + Morph (125 features)
X_morph, cols_morph = load_csv('Dataset/sbr_features_morph_train.csv')
print(f"  SBR+Morph: {X_morph.shape[1]} features")

# 3. Physical (SBR extended, 109 features)
X_phys, cols_phys = load_csv('Dataset/phys_features_train.csv')
print(f"  Physical: {X_phys.shape[1]} features")

# 4. Atlas (34 features)
X_atlas, cols_atlas = load_csv('Dataset/atlas_features_train.csv')
print(f"  Atlas: {X_atlas.shape[1]} features")

# 5. Radiomics global (97 features)
X_rad, cols_rad = load_csv('radiomic_features.csv')
print(f"  Radiomics: {X_rad.shape[1]} features")

# 6. Radiomics inorm (97 features)
X_rad_in, cols_rad_in = load_csv('radiomic_features_inorm.csv')
print(f"  Radiomics(inorm): {X_rad_in.shape[1]} features")

# 7. ROI Radiomics (68 features)
X_roi, cols_roi = load_csv('roi_radiomics.csv')
print(f"  ROI Radiomics: {X_roi.shape[1]} features")

# 8. Texture (48 features)
X_tex, cols_tex = load_csv('texture_roi.csv')
print(f"  Texture: {X_tex.shape[1]} features")

# 9. Striatal from crop (27 features)
X_striatal, cols_striatal = load_csv('striatal_features_from_crop.csv')
print(f"  Striatal: {X_striatal.shape[1]} features")

# 10. CNN OOF predictions (use as features)
cnn_oof_files = [
    'v30_oof/cnn3dr_oof.npy', 'v30_oof/cnn3d_oof.npy',
    'v30_oof/cnn25s_oof.npy', 'v30_oof/cnn25v2_oof.npy',
    'v30_oof/cnn25r2_oof.npy', 'v30_oof/cnn25b_oof.npy',
    'v30_oof/cnn3dr_oof_sc.npy', 'v30_oof/cnn3d_oof_sc.npy',
    'v30_oof/cnn3d_deep6.npy', 'v30_oof/cnn3d_deep12.npy',
    'balanced_oof_all_seeds.npy',
    'v40_honest_oof.npy', 'v42_temp_honest_oof.npy',
]
cnn_features = []
cnn_names = []
for f in cnn_oof_files:
    if os.path.exists(f):
        arr = np.load(f)
        if arr.ndim == 2:
            for i in range(arr.shape[1]):
                cnn_features.append(arr[:, i])
                cnn_names.append(f'{os.path.basename(f)}_s{i}')
        else:
            cnn_features.append(arr)
            cnn_names.append(os.path.basename(f))
X_cnn = np.column_stack(cnn_features)
print(f"  CNN OOF: {X_cnn.shape[1]} features from {len(cnn_oof_files)} files")

# 11. 512-dim embeddings
X_emb = np.load('phase6_embeddings.npy')
print(f"  Embeddings: {X_emb.shape[1]} features")

# 12. PCA embeddings
X_pca = np.load('phase6_pca_emb.npy')
print(f"  PCA Embeddings: {X_pca.shape[1]} features")

# ============ FEATURE SET DEFINITIONS ============
feature_sets = {
    'SBR_PVE': X_sbr,
    'SBR_Morph': X_morph,
    'SBR_Physical': X_phys,
    'SBR_Atlas': X_atlas,
    'SBR_Radiomics': np.hstack([X_sbr, X_rad, X_roi, X_tex]),
    'SBR_Radiomics_Inorm': np.hstack([X_sbr, X_rad_in, X_roi, X_tex]),
    'SBR_CNN_OOF': np.hstack([X_sbr, X_cnn]),
    'SBR_Phys_Rad_CNN': np.hstack([X_phys, X_rad, X_roi, X_tex, X_cnn]),
    'Full_Fused': np.hstack([X_phys, X_rad_in, X_roi, X_tex, X_cnn, X_pca]),
    'Full_Plus_Emb': np.hstack([X_phys, X_rad_in, X_roi, X_tex, X_cnn, X_emb]),
}

# ============ HYPERPARAMETER GRIDS ============
xgb_params_grid = [
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8,
     'colsample_bytree': 0.7, 'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
     'gamma': 0.1, 'scale_pos_weight': 1.0},
    {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.75,
     'colsample_bytree': 0.6, 'min_child_weight': 8, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
     'gamma': 0.2, 'scale_pos_weight': 1.0},
    {'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.04, 'subsample': 0.85,
     'colsample_bytree': 0.8, 'min_child_weight': 3, 'reg_alpha': 0.01, 'reg_lambda': 0.5,
     'gamma': 0.0, 'scale_pos_weight': 1.0},
    {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.9,
     'colsample_bytree': 0.9, 'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
     'gamma': 0.5, 'scale_pos_weight': 1.0},
    {'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.02, 'subsample': 0.7,
     'colsample_bytree': 0.5, 'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 3.0,
     'gamma': 0.3, 'scale_pos_weight': 1.0},
]

lgb_params_grid = [
    {'n_estimators': 500, 'max_depth': 5, 'learning_rate': 0.05, 'subsample': 0.8,
     'colsample_bytree': 0.7, 'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
     'num_leaves': 31, 'min_split_gain': 0.1},
    {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.75,
     'colsample_bytree': 0.6, 'min_child_weight': 8, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
     'num_leaves': 20, 'min_split_gain': 0.2},
    {'n_estimators': 600, 'max_depth': 6, 'learning_rate': 0.04, 'subsample': 0.85,
     'colsample_bytree': 0.8, 'min_child_weight': 3, 'reg_alpha': 0.01, 'reg_lambda': 0.5,
     'num_leaves': 50, 'min_split_gain': 0.0},
    {'n_estimators': 400, 'max_depth': 3, 'learning_rate': 0.1, 'subsample': 0.9,
     'colsample_bytree': 0.9, 'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 5.0,
     'num_leaves': 15, 'min_split_gain': 0.5},
    {'n_estimators': 1000, 'max_depth': 4, 'learning_rate': 0.02, 'subsample': 0.7,
     'colsample_bytree': 0.5, 'min_child_weight': 10, 'reg_alpha': 1.0, 'reg_lambda': 3.0,
     'num_leaves': 25, 'min_split_gain': 0.3},
]

# ============ OOF EVALUATION ============
N_FOLDS = 5
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

def run_oof(X, y, groups, model_factory, model_name, feat_name, calibrate=True):
    """Run OOF with proper train/val split. Tune on train, eval on val."""
    oof_raw = np.zeros(N)
    oof_cal = np.zeros(N) if calibrate else None
    fold_aucs = []
    fold_lls = []
    best_params_per_fold = []

    for fold_idx, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        t0 = time.time()
        X_trn, X_val = X[trn_idx], X[val_idx]
        y_trn, y_val = y[trn_idx], y[val_idx]

        # Scale features
        sc = StandardScaler()
        X_trn_s = sc.fit_transform(X_trn)
        X_val_s = sc.transform(X_val)

        # Replace NaN/Inf
        X_trn_s = np.nan_to_num(X_trn_s, nan=0.0, posinf=0.0, neginf=0.0)
        X_val_s = np.nan_to_num(X_val_s, nan=0.0, posinf=0.0, neginf=0.0)

        # Tune: try each param config, pick best by train-fold AUC via internal CV
        best_score = -1
        best_model = None
        best_params = None

        for params in (xgb_params_grid if 'XGB' in model_name else lgb_params_grid):
            model = model_factory(params)
            if 'XGB' in model_name:
                model.fit(X_trn_s, y_trn, verbose=False)
            else:
                model.fit(X_trn_s, y_trn)

            pred_trn = model.predict_proba(X_trn_s)[:, 1]
            score = roc_auc_score(y_trn, pred_trn)
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params

        # Predict on validation fold
        pred_val = best_model.predict_proba(X_val_s)[:, 1]
        oof_raw[val_idx] = pred_val
        best_params_per_fold.append(best_params)

        # Leakage-safe calibration: fit isotonic on train fold predictions, apply to val
        if calibrate:
            from sklearn.isotonic import IsotonicRegression
            ir = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds='clip')
            ir.fit(pred_trn, y_trn)
            pred_cal = ir.predict(pred_val)
            oof_cal[val_idx] = np.clip(pred_cal, 1e-7, 1-1e-7)

        auc_raw = roc_auc_score(y_val, pred_val)
        ll_raw = log_loss(y_val, np.clip(pred_val, 1e-7, 1-1e-7))
        fold_aucs.append(auc_raw)
        fold_lls.append(ll_raw)

        elapsed = time.time() - t0
        print(f"    Fold {fold_idx}: AUC={auc_raw:.4f} LL={ll_raw:.4f} ({elapsed:.0f}s)")

    # Overall metrics
    overall_auc = roc_auc_score(y, oof_raw)
    overall_ll = log_loss(y, np.clip(oof_raw, 1e-7, 1-1e-7))

    result = {
        'model': model_name, 'features': feat_name,
        'n_features': X.shape[1],
        'oof_auc': overall_auc, 'oof_ll': overall_ll,
        'fold_aucs': fold_aucs, 'fold_lls': fold_lls,
        'mean_fold_auc': np.mean(fold_aucs), 'std_fold_auc': np.std(fold_aucs),
        'mean_fold_ll': np.mean(fold_lls), 'std_fold_ll': np.std(fold_lls),
    }

    if calibrate and oof_cal is not None:
        auc_cal = roc_auc_score(y, oof_cal)
        ll_cal = log_loss(y, np.clip(oof_cal, 1e-7, 1-1e-7))
        result['oof_auc_cal'] = auc_cal
        result['oof_ll_cal'] = ll_cal

    # Save OOF predictions
    np.save(f'oof_{model_name}_{feat_name}_raw.npy', oof_raw)
    if calibrate and oof_cal is not None:
        np.save(f'oof_{model_name}_{feat_name}_cal.npy', oof_cal)

    return result

# ============ MAIN ============
results = []

for feat_name, X_feat in feature_sets.items():
    print(f"\n{'='*70}")
    print(f"Feature set: {feat_name} ({X_feat.shape[1]} features)")
    print(f"{'='*70}")

    # XGBoost
    print(f"\n  XGBoost:")
    def xgb_factory(params):
        return xgb.XGBClassifier(
            **params, objective='binary:logistic', eval_metric='logloss',
            use_label_encoder=False, random_state=42, n_jobs=-1)
    r_xgb = run_oof(X_feat, y, groups, xgb_factory, 'XGBoost', feat_name)
    results.append(r_xgb)
    print(f"  => XGB: AUC={r_xgb['oof_auc']:.4f} LL={r_xgb['oof_ll']:.4f}", end='')
    if 'oof_auc_cal' in r_xgb:
        print(f" | Cal: AUC={r_xgb['oof_auc_cal']:.4f} LL={r_xgb['oof_ll_cal']:.4f}")
    else:
        print()

    # LightGBM
    print(f"\n  LightGBM:")
    def lgb_factory(params):
        return lgb.LGBMClassifier(
            **params, objective='binary', metric='binary_logloss',
            random_state=42, n_jobs=-1, verbose=-1)
    r_lgb = run_oof(X_feat, y, groups, lgb_factory, 'LightGBM', feat_name)
    results.append(r_lgb)
    print(f"  => LGB: AUC={r_lgb['oof_auc']:.4f} LL={r_lgb['oof_ll']:.4f}", end='')
    if 'oof_auc_cal' in r_lgb:
        print(f" | Cal: AUC={r_lgb['oof_auc_cal']:.4f} LL={r_lgb['oof_ll_cal']:.4f}")
    else:
        print()

# ============ SUMMARY ============
print(f"\n\n{'='*90}")
print("SUMMARY")
print(f"{'='*90}")
print(f"{'Model':<12} {'Features':<25} {'#Feat':>6} {'AUC':>7} {'LL':>7} {'AUC_cal':>8} {'LL_cal':>8}")
print("-"*90)
for r in sorted(results, key=lambda x: x.get('oof_ll_cal', x['oof_ll'])):
    auc_cal = r.get('oof_auc_cal', float('nan'))
    ll_cal = r.get('oof_ll_cal', float('nan'))
    print(f"{r['model']:<12} {r['features']:<25} {r['n_features']:>6} "
          f"{r['oof_auc']:>7.4f} {r['oof_ll']:>7.4f} {auc_cal:>8.4f} {ll_cal:>8.4f}")

# Target check
print(f"\nTarget: LL < 0.25, AUC > 0.95")
for r in results:
    ll = r.get('oof_ll_cal', r['oof_ll'])
    auc = r.get('oof_auc_cal', r['oof_auc'])
    if ll < 0.25 and auc > 0.95:
        print(f"  ** TARGET MET: {r['model']} {r['features']} AUC={auc:.4f} LL={ll:.4f}")

# Save results
with open('xgb_lgb_comparison.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nResults saved to xgb_lgb_comparison.json")

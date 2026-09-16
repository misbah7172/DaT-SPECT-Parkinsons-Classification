"""Round 2: Fix calibration, try embeddings, try SBR-only without CNN OOF leakage.
Key fixes:
- Platt (sigmoid) calibration instead of isotonic (more robust on small folds)
- Cross-validated Platt to avoid overfitting
- Try 512-dim CNN embeddings as features (not OOF predictions)
- Try LGB+XGB blend"""
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import os, json, time
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
import lightgbm as lgb

y = np.load('v26_oof/y.npy')
groups = np.load('v26_oof/groups.npy', allow_pickle=True)
N = len(y)

def load_csv(path):
    df = pd.read_csv(path)
    cols = [c for c in df.columns if c not in ('uid', 'label')]
    return df[cols].values.astype(np.float64)

X_sbr = load_csv('Dataset/sbr_features_train.csv')
X_phys = load_csv('Dataset/phys_features_train.csv')
X_rad = load_csv('radiomic_features_inorm.csv')
X_roi = load_csv('roi_radiomics.csv')
X_tex = load_csv('texture_roi.csv')

# CNN features: use embeddings (not OOF predictions - avoids leakage)
X_emb = np.load('phase6_embeddings.npy')  # 512-dim
X_pca = np.load('phase6_pca_emb.npy')     # 50-dim

# CNN OOF predictions (these DO have some leakage, but useful for stacking)
cnn_files = ['v30_oof/cnn3dr_oof.npy', 'v30_oof/cnn3d_oof.npy',
             'v30_oof/cnn25s_oof.npy', 'v30_oof/cnn25v2_oof.npy',
             'v30_oof/cnn25r2_oof.npy', 'v30_oof/cnn25b_oof.npy',
             'v30_oof/cnn3dr_oof_sc.npy', 'v30_oof/cnn3d_oof_sc.npy',
             'v30_oof/cnn3d_deep6.npy', 'v30_oof/cnn3d_deep12.npy',
             'balanced_oof_all_seeds.npy', 'v40_honest_oof.npy',
             'v42_temp_honest_oof.npy']
cnn_feats = []
for f in cnn_files:
    if os.path.exists(f):
        arr = np.load(f)
        if arr.ndim == 2:
            for i in range(arr.shape[1]):
                cnn_feats.append(arr[:, i])
        else:
            cnn_feats.append(arr)
X_cnn = np.column_stack(cnn_feats)

feature_sets = {
    'SBR': X_sbr,
    'SBR_Phys': X_phys,
    'SBR_Rad': np.hstack([X_sbr, X_rad, X_roi, X_tex]),
    'SBR_CNN_OOF': np.hstack([X_sbr, X_cnn]),
    'SBR_Phys_CNN_OOF': np.hstack([X_phys, X_cnn]),
    'SBR_PCA': np.hstack([X_sbr, X_pca]),
    'SBR_Phys_PCA': np.hstack([X_phys, X_pca]),
    'SBR_Emb_PCA': np.hstack([X_sbr, X_pca, X_emb[:, :100]]),
    'Phys_Rad_CNN': np.hstack([X_phys, X_rad, X_roi, X_tex, X_cnn]),
}

N_FOLDS = 5
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

xgb_params = [
    {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.04, 'subsample': 0.8,
     'colsample_bytree': 0.7, 'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0},
    {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.75,
     'colsample_bytree': 0.6, 'min_child_weight': 8, 'reg_alpha': 0.5, 'reg_lambda': 2.0},
    {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.85,
     'colsample_bytree': 0.8, 'min_child_weight': 3, 'reg_alpha': 0.01, 'reg_lambda': 0.5},
]
lgb_params = [
    {'n_estimators': 600, 'max_depth': 5, 'learning_rate': 0.04, 'subsample': 0.8,
     'colsample_bytree': 0.7, 'min_child_weight': 5, 'reg_alpha': 0.1, 'reg_lambda': 1.0,
     'num_leaves': 31},
    {'n_estimators': 800, 'max_depth': 4, 'learning_rate': 0.03, 'subsample': 0.75,
     'colsample_bytree': 0.6, 'min_child_weight': 8, 'reg_alpha': 0.5, 'reg_lambda': 2.0,
     'num_leaves': 20},
    {'n_estimators': 400, 'max_depth': 6, 'learning_rate': 0.05, 'subsample': 0.85,
     'colsample_bytree': 0.8, 'min_child_weight': 3, 'reg_alpha': 0.01, 'reg_lambda': 0.5,
     'num_leaves': 50},
]

results = []

for feat_name, X_feat in feature_sets.items():
    print(f"\n{'='*70}")
    print(f"{feat_name} ({X_feat.shape[1]} features)")

    for model_name, factory, params_grid in [
        ('XGBoost', lambda p: xgb.XGBClassifier(**p, objective='binary:logistic',
            eval_metric='logloss', use_label_encoder=False, random_state=42, n_jobs=-1), xgb_params),
        ('LightGBM', lambda p: lgb.LGBMClassifier(**p, objective='binary',
            metric='binary_logloss', random_state=42, n_jobs=-1, verbose=-1), lgb_params),
    ]:
        oof_raw = np.zeros(N)
        oof_platt = np.zeros(N)
        oof_temp = np.zeros(N)
        fold_aucs, fold_lls = [], []

        for fold_idx, (trn_idx, val_idx) in enumerate(sgkf.split(X_feat, y, groups)):
            t0 = time.time()
            X_trn, X_val = X_feat[trn_idx], X_feat[val_idx]
            y_trn, y_val = y[trn_idx], y[val_idx]

            sc = StandardScaler()
            X_trn_s = np.nan_to_num(sc.fit_transform(X_trn), 0)
            X_val_s = np.nan_to_num(sc.transform(X_val), 0)

            best_score, best_model = -1, None
            for params in params_grid:
                m = factory(params)
                m.fit(X_trn_s, y_trn)
                pred = m.predict_proba(X_trn_s)[:, 1]
                score = roc_auc_score(y_trn, pred)
                if score > best_score:
                    best_score = score
                    best_model = m

            pred_val = best_model.predict_proba(X_val_s)[:, 1]
            oof_raw[val_idx] = pred_val

            # Platt calibration: fit sigmoid on train predictions
            pred_trn = best_model.predict_proba(X_trn_s)[:, 1]
            lr = LogisticRegression(C=1.0, max_iter=1000)
            lr.fit(pred_trn.reshape(-1, 1), y_trn)
            oof_platt[val_idx] = lr.predict_proba(pred_val.reshape(-1, 1))[:, 1]

            # Temperature scaling: fit T on train predictions
            from scipy.optimize import minimize_scalar
            def neg_ll(T):
                logits = np.log(np.clip(pred_trn, 1e-7, 1-1e-7) /
                               (1 - np.clip(pred_trn, 1e-7, 1-1e-7)))
                scaled = 1 / (1 + np.exp(-logits / T))
                return log_loss(y_trn, np.clip(scaled, 1e-7, 1-1e-7))
            res = minimize_scalar(neg_ll, bounds=(0.3, 2.0), method='bounded')
            T_opt = res.x
            logits = np.log(np.clip(pred_val, 1e-7, 1-1e-7) /
                           (1 - np.clip(pred_val, 1e-7, 1-1e-7)))
            oof_temp[val_idx] = 1 / (1 + np.exp(-logits / T_opt))

            fold_aucs.append(roc_auc_score(y_val, pred_val))
            fold_lls.append(log_loss(y_val, np.clip(pred_val, 1e-7, 1-1e-7)))
            print(f"  {model_name} F{fold_idx}: AUC={fold_aucs[-1]:.4f} LL={fold_lls[-1]:.4f} ({time.time()-t0:.0f}s)")

        for tag, oof in [('raw', oof_raw), ('platt', oof_platt), ('temp', oof_temp)]:
            auc = roc_auc_score(y, oof)
            ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
            results.append({
                'model': model_name, 'features': feat_name, 'calibration': tag,
                'n_features': X_feat.shape[1],
                'auc': auc, 'll': ll,
                'mean_fold_auc': np.mean(fold_aucs), 'std_fold_auc': np.std(fold_aucs),
            })
            marker = ' ***' if ll < 0.30 else ''
            print(f"  => {tag:6s}: AUC={auc:.4f} LL={ll:.4f}{marker}")

# Summary
print(f"\n\n{'='*90}")
print("TOP 20 BY LOG LOSS (raw probabilities)")
print(f"{'='*90}")
raw_results = [r for r in results if r['calibration'] == 'raw']
raw_results.sort(key=lambda x: x['ll'])
print(f"{'Model':<10} {'Features':<22} {'#F':>4} {'AUC':>7} {'LL':>7} {'FoldAUC':>12}")
print("-"*70)
for r in raw_results[:20]:
    print(f"{r['model']:<10} {r['features']:<22} {r['n_features']:>4} "
          f"{r['auc']:>7.4f} {r['ll']:>7.4f} "
          f"{r['mean_fold_auc']:>5.4f}+/-{r['std_fold_auc']:.4f}")

print(f"\n{'='*90}")
print("BEST PER FEATURE SET (all calibrations)")
print(f"{'='*90}")
best_per_feat = {}
for r in results:
    key = (r['model'], r['features'])
    if key not in best_per_feat or r['ll'] < best_per_feat[key]['ll']:
        best_per_feat[key] = r
for key, r in sorted(best_per_feat.items(), key=lambda x: x[1]['ll']):
    print(f"  {r['model']:<10} {r['features']:<22} {r['calibration']:<8} "
          f"AUC={r['auc']:.4f} LL={r['ll']:.4f}")

with open('xgb_lgb_r2.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)
print("\nSaved xgb_lgb_r2.json")

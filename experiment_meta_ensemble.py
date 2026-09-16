"""Experiment 2: Feature-level ensemble - stack OOF predictions → meta-learner."""
import json, pickle, sys, os
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

W = r'E:\DaT\submission_v26\weights'
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()

deep_raw = np.load(W + r'\fold_oof.npy')

with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names, oofs = A['names'], np.column_stack(A['oof'])
idx = [i for i, n in enumerate(names) if n in ('sbr_lr', 'sbr_xgb', 'sbr_lgb', 'sbr_et', 'sbr_ridge')]
sbr_models_oof = oofs[:, idx]
sbr_mean = sbr_models_oof.mean(axis=1)

with open(W + r'\ship_final.json', 'r') as f:
    ship = json.load(f)

eps = 1e-6

print("=" * 60)
print("EXPERIMENT 2: Meta-Learner on Stacked OOF Predictions")
print("=" * 60)

# Meta-features from OOF predictions
meta_features = np.column_stack([
    deep_raw,
    sbr_mean,
    sbr_models_oof[:, 0],
    sbr_models_oof[:, 1],
    sbr_models_oof[:, 2],
    sbr_models_oof[:, 3],
    sbr_models_oof[:, 4],
    deep_raw * sbr_mean,
    np.abs(deep_raw - sbr_mean),
    deep_raw ** 2,
    sbr_mean ** 2,
])
feature_names = ['deep', 'sbr_mean', 'sbr_lr', 'sbr_xgb', 'sbr_lgb', 'sbr_et', 'sbr_ridge', 'deep*sbr', '|deep-sbr|', 'deep^2', 'sbr^2']
print(f"Meta-features shape: {meta_features.shape}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def evaluate_oof(oof_pred, name):
    auc = roc_auc_score(y, oof_pred)
    ll = log_loss(y, np.clip(oof_pred, eps, 1-eps))
    brier = brier_score_loss(y, oof_pred)
    print(f"   {name:30s} AUC={auc:.4f}  LL={ll:.4f}  Brier={brier:.4f}")
    return auc, ll

print(f"\n   Current blend:                AUC={ship['final_auc']:.4f}  LL={ship['final_ll']:.4f}")
print(f"   Deep only (raw):              ", end="")
evaluate_oof(deep_raw, "")

# 1. Simple stacking: mean of deep + sbr
simple_mean = (deep_raw + sbr_mean) / 2
print(f"   Simple mean (deep+sbr):       ", end="")
evaluate_oof(simple_mean, "")

# 2. Logistic Regression meta-learner
oof_lr = np.zeros(len(y))
for fold, (trn_idx, val_idx) in enumerate(sgkf.split(meta_features, y, groups)):
    X_trn, X_val = meta_features[trn_idx], meta_features[val_idx]
    y_trn = y[trn_idx]
    scaler = StandardScaler()
    X_trn_s = scaler.fit_transform(X_trn)
    X_val_s = scaler.transform(X_val)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_trn_s, y_trn)
    oof_lr[val_idx] = lr.predict_proba(X_val_s)[:, 1]
print(f"   Meta-LogReg (11 features):     ", end="")
evaluate_oof(oof_lr, "")

# 3. XGBoost meta-learner
oof_xgb = np.zeros(len(y))
for fold, (trn_idx, val_idx) in enumerate(sgkf.split(meta_features, y, groups)):
    X_trn, X_val = meta_features[trn_idx], meta_features[val_idx]
    y_trn, y_val = y[trn_idx], y[val_idx]
    dtrain = xgb.DMatrix(X_trn, label=y_trn, feature_names=feature_names)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
    params = {
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'max_depth': 3, 'learning_rate': 0.05, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'min_child_weight': 5,
        'seed': 42, 'verbosity': 0,
    }
    model = xgb.train(params, dtrain, num_boost_round=200,
                      evals=[(dval, 'val')], early_stopping_rounds=20, verbose_eval=False)
    oof_xgb[val_idx] = model.predict(dval)
print(f"   Meta-XGBoost (11 features):    ", end="")
evaluate_oof(oof_xgb, "")

# 4. ExtraTrees meta-learner
oof_et = np.zeros(len(y))
for fold, (trn_idx, val_idx) in enumerate(sgkf.split(meta_features, y, groups)):
    X_trn, X_val = meta_features[trn_idx], meta_features[val_idx]
    y_trn = y[trn_idx]
    et = ExtraTreesClassifier(n_estimators=200, max_depth=5, min_samples_leaf=5, random_state=42)
    et.fit(X_trn, y_trn)
    oof_et[val_idx] = et.predict_proba(X_val)[:, 1]
print(f"   Meta-ExtraTrees (11 features): ", end="")
evaluate_oof(oof_et, "")

# 5. Minimal meta: just deep + sbr + interaction
meta_minimal = meta_features[:, [0, 1, 7, 8]]  # deep, sbr, deep*sbr, |deep-sbr|
oof_lr_min = np.zeros(len(y))
for fold, (trn_idx, val_idx) in enumerate(sgkf.split(meta_minimal, y, groups)):
    X_trn, X_val = meta_minimal[trn_idx], meta_minimal[val_idx]
    y_trn = y[trn_idx]
    scaler = StandardScaler()
    X_trn_s = scaler.fit_transform(X_trn)
    X_val_s = scaler.transform(X_val)
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr.fit(X_trn_s, y_trn)
    oof_lr_min[val_idx] = lr.predict_proba(X_val_s)[:, 1]
print(f"   Meta-LogReg (4 features):      ", end="")
evaluate_oof(oof_lr_min, "")

# 6. Temperature scale the best meta-learner and compare
def temp_loss(params, y, p):
    T = params[0]
    p_cal = np.clip(expit(logit(np.clip(p, eps, 1-eps)) / T), eps, 1-eps)
    return log_loss(y, p_cal)

best_T = 1.0
best_ll = temp_loss([1.0], y, oof_lr)
for T_init in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.5]:
    r = minimize(temp_loss, [T_init], args=(y, oof_lr), method='Nelder-Mead',
                 options={'maxiter': 200})
    if r.fun < best_ll:
        best_ll = r.fun
        best_T = r.x[0]
oof_lr_cal = np.clip(expit(logit(np.clip(oof_lr, eps, 1-eps)) / best_T), eps, 1-eps)
print(f"   Meta-LogReg + temp scale:      ", end="")
evaluate_oof(oof_lr_cal, f"(T={best_T:.4f})")

# 7. Try stacking only the deep individual combo OOFs
# Load individual combo OOFs
import glob
combo_files = sorted(glob.glob(W + r'\combo_*_oof.npy'))
if combo_files:
    print(f"\n2. Individual Combo Analysis ({len(combo_files)} combos):")
    combo_oofs = []
    for cf in combo_files:
        combo_oof = np.load(cf)
        combo_oofs.append(combo_oof)
        auc = roc_auc_score(y, combo_oof)
        ll = log_loss(y, np.clip(combo_oof, eps, 1-eps))
        fname = os.path.basename(cf).replace('_oof.npy', '')
        print(f"   {fname:20s} AUC={auc:.4f}  LL={ll:.4f}")
    
    # Try optimal combo weighting
    combo_stack = np.column_stack(combo_oofs)
    
    def combo_loss(params):
        w = np.array(params)
        w = w / w.sum()
        blend = np.clip(combo_stack @ w, eps, 1-eps)
        return log_loss(y, blend)
    
    n_combos = len(combo_oofs)
    init_w = [1.0/n_combos] * n_combos
    bounds = [(0, 1)] * n_combos
    from scipy.optimize import minimize as sp_minimize
    result = sp_minimize(combo_loss, init_w, method='Nelder-Mead',
                        options={'maxiter': 5000, 'xatol': 1e-6})
    opt_w = np.array(result.x)
    opt_w = opt_w / opt_w.sum()
    combo_blend = np.clip(combo_stack @ opt_w, eps, 1-eps)
    print(f"\n   Optimal combo weights: {np.round(opt_w, 4)}")
    print(f"   Combo blend:           AUC={roc_auc_score(y, combo_blend):.4f}  LL={log_loss(y, combo_blend):.4f}")
    
    # Equal weight combo blend
    eq_w = np.ones(n_combos) / n_combos
    eq_blend = np.clip(combo_stack @ eq_w, eps, 1-eps)
    print(f"   Equal weight blend:    AUC={roc_auc_score(y, eq_blend):.4f}  LL={log_loss(y, eq_blend):.4f}")
else:
    print("\n2. No combo OOF files found")

print("\n" + "=" * 60)
print("CONCLUSION")
print("=" * 60)
print(f"""
Current best:  AUC={ship['final_auc']:.4f}, LL={ship['final_ll']:.4f}
Best meta:     AUC={roc_auc_score(y, oof_lr):.4f}, LL={log_loss(y, oof_lr):.4f}

Recommendation: {'Meta-learner helps!' if log_loss(y, oof_lr) < ship['final_ll'] else 'Current blend is already good.'}
""")

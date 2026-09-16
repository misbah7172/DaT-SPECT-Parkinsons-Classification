"""Phase 4: Fixed SBR analysis (label column excluded)."""
import pandas as pd, numpy as np, pickle, json
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
import xgboost as xgb, lightgbm as lgb

y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
deep_raw = np.load(r'E:\DaT\submission_v26\weights\fold_oof.npy')

sbr = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv')
vg = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
m = sbr.merge(vg, left_index=True, right_index=True, how='left')

# DROP uid, label
drop = ['uid', 'label']
cols = [c for c in m.columns if c not in drop and m[c].dtype != object]
X = m[cols].fillna(0).values.astype(float)
print(f'Features: {X.shape[1]} (label excluded)')

PVE_REF = 15.625
vv = m['voxel_vol'].values
fac = (PVE_REF / np.clip(vv, 1, None)) ** (1/3)
Xp = X.copy()
for i in range(min(89, X.shape[1])):
    Xp[:, i] *= fac

eps = 1e-8
lp = m['sbr_left_putamen'].values; rp = m['sbr_right_putamen'].values
lc = m['sbr_left_caudate'].values; rc = m['sbr_right_caudate'].values
derived = np.column_stack([
    (lp-rp)/(lp+rp+eps), (lc-rc)/(lc+rc+eps),
    lp+rp+lc+rc, lp+lc, rp+rc,
    (lp+lc)/(lp+rp+lc+rc+eps), (rp+rc)/(lp+rp+lc+rc+eps),
    (lp+rp)/(lc+rc+eps),
])
Xf = np.column_stack([Xp, derived])
print(f'Final features: {Xf.shape[1]}')

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

def run_sbr(name, fac_fn):
    oof = np.zeros(len(y))
    for tri, vai in sgkf.split(Xf, y, groups):
        Xt, Xv = Xf[tri], Xf[vai]
        yt = y[tri]
        mi = mutual_info_classif(Xt, yt, random_state=42)
        top = np.argsort(mi)[-40:]
        sc = StandardScaler()
        Xt_s = sc.fit_transform(Xt[:, top])
        Xv_s = sc.transform(Xv[:, top])
        clf = fac_fn()
        clf.fit(Xt_s, yt)
        oof[vai] = clf.predict_proba(Xv_s)[:, 1]
    return roc_auc_score(y, oof), log_loss(y, np.clip(oof, 1e-7, 1-1e-7)), oof

models = [
    ('LogReg', lambda: LogisticRegression(C=0.5, max_iter=1000)),
    ('XGBoost', lambda: xgb.XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbosity=0)),
    ('LightGBM', lambda: lgb.LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, verbose=-1)),
    ('ExtraTrees', lambda: ExtraTreesClassifier(n_estimators=300, min_samples_leaf=5)),
]

sbr_oofs = {}
print()
for name, fac in models:
    auc, ll, oof = run_sbr(name, fac)
    sbr_oofs[name] = oof
    print(f'  {name:15s}: AUC={auc:.4f}, LL={ll:.4f}')

sbr_new = np.mean(np.column_stack(list(sbr_oofs.values())), axis=1)
new_auc = roc_auc_score(y, sbr_new)
new_ll = log_loss(y, np.clip(sbr_new, 1e-7, 1-1e-7))
print(f'  New SBR Mean: AUC={new_auc:.4f}, LL={new_ll:.4f}')

with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
    A = pickle.load(f)
names, oofs_arr = A['names'], np.column_stack(A['oof'])
idx = [i for i, n in enumerate(names) if n in ('sbr_lr', 'sbr_xgb', 'sbr_lgb', 'sbr_et', 'sbr_ridge')]
sbr_old = oofs_arr[:, idx].mean(axis=1)
old_auc = roc_auc_score(y, sbr_old)
old_ll = log_loss(y, np.clip(sbr_old, 1e-7, 1-1e-7))
print(f'  Old SBR Mean: AUC={old_auc:.4f}, LL={old_ll:.4f}')

def blend_loss(params):
    wd, T = params
    b = np.clip(wd*deep_raw + (1-wd)*sbr_new, 1e-7, 1-1e-7)
    p = np.clip(expit(logit(b)/T), 1e-7, 1-1e-7)
    return log_loss(y, p)

best = None
for init in ([0.8,0.7],[0.85,0.8],[0.75,0.6],[0.9,0.5],[0.82,0.65]):
    r = minimize(blend_loss, init, method='Nelder-Mead', options={'maxiter':500})
    if best is None or r.fun < best.fun: best = r
wd, T = best.x
bp = np.clip(expit(logit(np.clip(wd*deep_raw+(1-wd)*sbr_new,1e-7,1-1e-7))/T), 1e-7, 1-1e-7)
blend_auc = roc_auc_score(y, bp)
blend_ll = log_loss(y, bp)
print(f'  New blend: w_deep={wd:.4f}, w_sbr={1-wd:.4f}, T={T:.4f}')
print(f'             AUC={blend_auc:.4f}, LL={blend_ll:.4f}')

with open(r'E:\DaT\submission_v26\weights\ship_final.json') as f:
    ship = json.load(f)
print(f'  Current:    AUC={ship["final_auc"]:.4f}, LL={ship["final_ll"]:.4f}')

print(f'\nSBR improvement: {old_ll:.4f} -> {new_ll:.4f}')
print(f'Blend improvement: {ship["final_ll"]:.4f} -> {blend_ll:.4f}')

if blend_ll < ship['final_ll']:
    # Save new blend parameters
    result = {
        'w_deep': round(float(wd), 4),
        'w_sbr': round(float(1-wd), 4),
        'T': round(float(T), 4),
        'deep_auc': float(roc_auc_score(y, deep_raw)),
        'deep_ll': float(log_loss(y, np.clip(deep_raw, 1e-7, 1-1e-7))),
        'sbr_auc': float(new_auc),
        'sbr_ll': float(new_ll),
        'final_auc': float(blend_auc),
        'final_ll': float(blend_ll),
    }
    with open(r'E:\DaT\submission_v26\weights\ship_final.json', 'w') as f:
        json.dump(result, f, indent=2)
    print('\nUpdated ship_final.json with new blend!')
else:
    print('\nCurrent blend is better. No update.')

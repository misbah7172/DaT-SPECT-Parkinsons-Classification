"""Train the 6 full-data sbr models for submission (mirrors v26 sbr stream).
Saves scaler, MI selection, and models to E:/DaT/submission_v23/weights/
as sbr_full_{lr,xgb,lgb,cb,et,ridge}.pkl and sbr_full_scaler.pkl, sbr_full_sel.pkl.
Also saves the feature column list used at test time."""
import os, pickle, json
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss, roc_auc_score
import xgboost as xgb, lightgbm as lgb, catboost as cb
from sklearn.ensemble import ExtraTreesClassifier

PVE_REF = 15.625
W = r'E:\DaT\submission_v23\weights'
os.makedirs(W, exist_ok=True)
SEED = 42

# ---- load raw sbr features (same as v26, PVE-corrected) ----
labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
geom = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
sbr_raw = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv').drop(columns=['label'], errors='ignore')
merged = labels.merge(geom, on='uid').set_index('uid')
fac = (PVE_REF / merged['voxel_vol'].values) ** (1.0 / 3.0)
sbr = sbr_raw.set_index('uid').loc[merged.index].copy()
for c in [c for c in sbr.columns if c.startswith('sbr_')]:
    sbr[c] = sbr[c] * fac
y = merged['is_pathologic'].values


def add_bio(df):
    eps = 1e-6
    L = df["sbr_left_putamen"].values.astype(float)
    R = df["sbr_right_putamen"].values.astype(float)
    df["lr_put_asym"] = (L - R) / (L + R + eps)
    df["lr_put_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
    df["lr_put_sum"] = L + R
    df["lr_put_diff"] = L - R
    L, R = df["sbr_left_caudate"].values.astype(float), df["sbr_right_caudate"].values.astype(float)
    df["lr_cau_asym"] = (L - R) / (L + R + eps)
    df["lr_cau_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
    df["lr_cau_sum"] = L + R
    df["lp_post_ant"] = df["sbr_left_putamen_post"].values.astype(float) / (df["sbr_left_putamen_ant"].values.astype(float) + eps)
    df["rp_post_ant"] = df["sbr_right_putamen_post"].values.astype(float) / (df["sbr_right_putamen_ant"].values.astype(float) + eps)
    df["lp_cr"] = df["sbr_left_putamen"].values.astype(float) / (df["sbr_left_caudate"].values.astype(float) + eps)
    df["rp_cr"] = df["sbr_right_putamen"].values.astype(float) / (df["sbr_right_caudate"].values.astype(float) + eps)
    df["total_str"] = df["sbr_left_total"].values.astype(float) + df["sbr_right_total"].values.astype(float)
    L, R = df["sbr_left_total"].values.astype(float), df["sbr_right_total"].values.astype(float)
    df["lr_total_asym"] = (L - R) / (L + R + eps)
    return df


def build_matrix(raw):
    cols = [c for c in raw.columns if c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


df = add_bio(sbr.copy())
COLS = list(df.columns)
X = build_matrix(df)
sc = StandardScaler().fit(X)
Xs = sc.transform(X)
mt = min(44, X.shape[1])
sel = np.argsort(mutual_info_classif(Xs, y, random_state=SEED))[::-1][:mt]
Xt = Xs[:, sel]

models = {}
models['lr'] = LogisticRegression(C=1.31, max_iter=3000, random_state=SEED).fit(Xs, y)
models['ridge'] = RidgeClassifier(alpha=0.1, random_state=SEED).fit(Xs, y)
models['xgb'] = xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False,
    eval_metric="logloss", random_state=SEED, n_jobs=-1, verbosity=0).fit(Xt, y)
models['lgb'] = lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10,
    random_state=SEED, n_jobs=-1, verbose=-1).fit(Xt, y)
models['cb'] = cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05,
    l2_leaf_reg=3.0, random_seed=SEED, verbose=0).fit(Xt, y)
models['et'] = ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=SEED).fit(Xt, y)

pred = np.zeros(len(y))
for name, m in models.items():
    if name in ('lr', 'ridge'):
        p = m.predict_proba(Xs)[:, 1] if name == 'lr' else 1/(1+np.exp(-m.decision_function(Xs)))
    else:
        p = m.predict_proba(Xt)[:, 1]
    pred += p
    print(f'{name}: AUC={roc_auc_score(y, p):.4f} LL={log_loss(y, np.clip(p, 1e-6, 1-1e-6)):.4f}', flush=True)
pred /= len(models)
print(f'sbr-mean stream: AUC={roc_auc_score(y, pred):.4f} LL={log_loss(y, np.clip(pred, 1e-6, 1-1e-6)):.4f}', flush=True)

# save assets
for name, m in models.items():
    with open(os.path.join(W, f'sbr_full_{name}.pkl'), 'wb') as f:
        pickle.dump(m, f)
with open(os.path.join(W, 'sbr_full_scaler.pkl'), 'wb') as f:
    pickle.dump(sc, f)
with open(os.path.join(W, 'sbr_full_sel.pkl'), 'wb') as f:
    pickle.dump(sel, f)
json.dump(COLS, open(os.path.join(W, 'sbr_full_cols.json'), 'w'))
print('saved sbr full-data models + scaler + sel + cols')
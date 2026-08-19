"""Generic: train a 6-model global OOF stream on a given feature CSV.
Usage: python train_stream.py <csv_path> <tag> <out_token>
"""
import os, sys, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from scipy.special import expit
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
import xgboost as xgb, lightgbm as lgb, catboost as cb

CSV, TAG, TOKEN = sys.argv[1], sys.argv[2], sys.argv[3]
SEEDS = [42, 777, 2024, 12345, 999]
N_FOLDS = 5
OUT = os.path.join(r'E:\DaT\v30_oof', f'oof_{TOKEN}.pkl')

labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
site = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv')
feat = pd.read_csv(CSV)
merged = labels.merge(site, on='uid').set_index('uid')
feat = feat.set_index('uid').loc[merged.index]
y = merged['is_pathologic'].values
groups = merged['pseudo_site'].astype(str).values
X = feat.values.astype(np.float64)
X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
X = np.hstack([X, np.log1p(np.clip(X, 0, None))])
print(CSV, '->', X.shape, 'pos', y.sum(), '/', len(y), flush=True)

n = len(y)
n_m = 6
oof = [np.zeros(n) for _ in range(n_m)]
cnt = [np.zeros(n) for _ in range(n_m)]
for seed in SEEDS:
    for tr, va in StratifiedGroupKFold(N_FOLDS, shuffle=True, random_state=seed).split(X, y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xva = sc.transform(X[tr]), sc.transform(X[va])
        mt = min(44, Xtr.shape[1])
        sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:mt]
        Xt, Xv = Xtr[:, sel], Xva[:, sel]
        oof[0][va] += LogisticRegression(C=1.31, max_iter=3000, random_state=seed).fit(Xtr, y[tr]).predict_proba(Xva)[:, 1]; cnt[0][va] += 1
        oof[1][va] += xgb.XGBClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, use_label_encoder=False, eval_metric="logloss", random_state=seed, n_jobs=-1, verbosity=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[1][va] += 1
        oof[2][va] += lgb.LGBMClassifier(n_estimators=500, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0, min_child_samples=10, random_state=seed, n_jobs=-1, verbose=-1).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[2][va] += 1
        oof[3][va] += cb.CatBoostClassifier(iterations=500, depth=4, learning_rate=0.05, l2_leaf_reg=3.0, random_seed=seed, verbose=0).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[3][va] += 1
        oof[4][va] += ExtraTreesClassifier(700, max_depth=9, n_jobs=-1, random_state=seed).fit(Xt, y[tr]).predict_proba(Xv)[:, 1]; cnt[4][va] += 1
        oof[5][va] += expit(RidgeClassifier(alpha=0.1, random_state=seed).fit(Xtr, y[tr]).decision_function(Xva)); cnt[5][va] += 1
    print('seed', seed, flush=True)
for i in range(n_m):
    oof[i] = np.where(cnt[i] > 0, oof[i] / cnt[i], 0.5)

names = [f'{TAG}_{m}' for m in ['lr','xgb','lgb','cb','et','ridge']]
M = np.column_stack(oof)
for i, name in enumerate(names):
    print(f'  {name}: AUC={roc_auc_score(y, M[:,i]):.4f} LL={log_loss(y, np.clip(M[:,i],1e-6,1-1e-6)):.4f}', flush=True)
with open(OUT, 'wb') as f:
    pickle.dump({'oof': oof, 'names': names}, f)
np.save(r'E:\DaT\v30_oof\y.npy', y)
np.save(r'E:\DaT\v30_oof\groups.npy', groups)
print('saved', OUT)
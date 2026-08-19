"""Consistent comparison within dominant site: existing features vs aligned-image features, same pipeline."""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold

CROPS = r'E:\DaT\aligned_cache\crops'
labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv').merge(pd.read_csv(r'E:\DaT\Dataset\site_labels.csv'), on='uid')
sbr = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv').set_index('uid').drop(columns=['label'], errors='ignore')
phys = pd.read_csv(r'E:\DaT\Dataset\phys_features_train.csv')
phys = phys.rename(columns={c: f'ph_{c}' for c in phys.columns if c != 'uid'}).set_index('uid').drop(columns=['ph_label'], errors='ignore')
morph = pd.read_csv(r'E:\DaT\Dataset\sbr_features_morph_train.csv').set_index('uid').drop(columns=['label'], errors='ignore')
atlas = pd.read_csv(r'E:\DaT\Dataset\atlas_features_train.csv').set_index('uid').drop(columns=['label'], errors='ignore')
geom = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv').set_index('uid')

fac = (15.625 / geom.loc[labels['uid'], 'voxel_vol'].values) ** (1/3)
sbrc = sbr.loc[labels['uid']].copy()
for c in sbrc.columns:
    if c.startswith('sbr_'):
        sbrc[c] = sbrc[c] * fac

def radiomic(crop):
    f = {}
    v = crop.ravel()
    levels = 32
    mn, mx = np.percentile(v, 1), np.percentile(v, 99.5)
    if mx <= mn: mx = mn + 1e-6
    q = np.clip(((crop - mn) / (mx - mn) * (levels - 1)).astype(int), 0, levels - 1)
    hist, _ = np.histogram(v, bins=24, range=(mn, mx))
    h = hist / max(hist.sum(), 1)
    f['hist_ent'] = -(h * np.log(h + 1e-9)).sum(); f['hist_energy'] = (h ** 2).sum()
    for p in (50, 75, 90, 95, 97, 99): f[f'p{p}'] = np.percentile(v, p)
    f['range'] = np.percentile(v, 99) - np.percentile(v, 50)
    f['max'] = v.max(); f['mean'] = v.mean(); f['std'] = v.std()
    for ax, name in ((0,'x'),(1,'y'),(2,'z')):
        prof = crop.max(axis=ax); vv = prof.ravel()
        if vv.max() > vv.min():
            pid = (vv - vv.min())/(vv.max() - vv.min())
            c = np.array(__import__('scipy').ndimage.center_of_mass(pid))
            if len(c) == 2: f[f'com_{name}1'] = c[0]; f[f'com_{name}2'] = c[1]
            f[f'std_{name}_sf'] = np.std(pid)
    return f

uids = labels[labels['pseudo_site'] == '2.5x2.5x2.5']['uid'].tolist()
n = len(uids)
y = labels.set_index('uid').loc[uids, 'is_pathologic'].values

Xs = {
    'sbr': sbrc.loc[uids].values,
    'morph': morph.loc[uids].values,
    'phys': phys.loc[uids].values,
    'atlas': atlas.loc[uids].values,
    'radiom': np.array([list(radiomic(np.load(os.path.join(CROPS, f'{u}.npy'))).values()) for u in uids], dtype=np.float64),
}
# concatenations
all_feat = np.hstack([Xs[k] for k in ['sbr','morph','phys','atlas','radiom']])

def eval_mat(Xmat, name, seeds=(42,)):
    aucs, lls = [], []
    for seed in seeds:
        for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(Xmat, y):
            Xtr = np.log1p(np.abs(Xmat[tr].astype(np.float64))); Xva = np.log1p(np.abs(Xmat[va].astype(np.float64)))
            sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:44]
            m = LogisticRegression(C=1.31, max_iter=30000).fit(Xtr[:, sel], y[tr])
            p = m.predict_proba(Xva[:, sel])[:, 1]
            aucs.append(roc_auc_score(y[va], p)); lls.append(log_loss(y[va], np.clip(p, 1e-6, 1-1e-6)))
    print(f'{name:28s} AUC {np.mean(aucs):.4f}  LL {np.mean(lls):.4f}')

for k, X in Xs.items():
    eval_mat(X, k)
eval_mat(all_feat, 'ALL combined')
eval_mat(np.hstack([Xs['sbr'], Xs['atlas']]), 'sbr+atlas (ceiling ref)')
eval_mat(np.hstack([Xs['sbr'], Xs['atlas'], Xs['radiom']]), 'sbr+atlas+radiom')
eval_mat(np.hstack([Xs['sbr'], Xs['atlas'], Xs['morph'], Xs['phys']]), 'sbr+atlas+morph+phys')
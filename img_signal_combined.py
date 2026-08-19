"""Combined signal test within dominant site: atlas features + radiomics."""
import os, sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedKFold
import importlib.util, types

spec = importlib.util.spec_from_file_location('img_signal_test', r'E:\DaT\img_signal_test.py')
mod = types.ModuleType('img_signal_test')
mod.__dict__['__file__'] = r'E:\DaT\img_signal_test.py'

CROPS = r'E:\DaT\aligned_cache\crops'
labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv').merge(pd.read_csv(r'E:\DaT\Dataset\site_labels.csv'), on='uid')
atlas = pd.read_csv(r'E:\DaT\Dataset\atlas_features_train.csv')

dom_ids = labels[labels['pseudo_site'] == '2.5x2.5x2.5']['uid'].tolist()
print('n=', len(dom_ids))

# radiomic features
def radiomic(crop):
    f = {}
    v = crop.ravel()
    levels = 32
    mn, mx = np.percentile(v, 1), np.percentile(v, 99.5)
    if mx <= mn: mx = mn + 1e-6
    q = np.clip(((crop - mn) / (mx - mn) * (levels - 1)).astype(int), 0, levels - 1)
    hist, _ = np.histogram(v, bins=24, range=(mn, mx))
    h = hist / max(hist.sum(), 1)
    f['hist_ent'] = -(h * np.log(h + 1e-9)).sum()
    f['hist_energy'] = (h ** 2).sum()
    for p in (50, 75, 90, 95, 97, 99):
        f[f'p{p}'] = np.percentile(v, p)
    f['range'] = np.percentile(v, 99) - np.percentile(v, 50)
    f['max'] = v.max(); f['mean'] = v.mean(); f['std'] = v.std()
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        prof = crop.max(axis=ax)
        vv = prof.ravel()
        if vv.max() > vv.min():
            pid = (vv - vv.min()) / (vv.max() - vv.min())
            c = np.array(__import__('scipy').ndimage.center_of_mass(pid))
            if len(c) == 2:
                f[f'com_{name}_surface'] = c[0]; f[f'com_{name}_surface2'] = c[1]
            f[f'std_{name}_surface'] = np.std(pid)
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        sl = np.ascontiguousarray(q.take(q.shape[ax] // 2, axis=ax))
        P = np.zeros((levels, levels), dtype=np.float64)
        H, W = sl.shape
        for i in range(H):
            row = sl[i]
            np.add.at(P, (row[:-1], row[1:]), 1)
            np.add.at(P, (row[1:], row[:-1]), 1)
        S = P.sum()
        if S > 0: P /= S
        ii = np.arange(levels)[:, None]; jj = np.arange(levels)[None, :]
        f[f'gc_{name}'] = float(((P * ((ii - jj) ** 2)).sum()))
        f[f'ge_{name}'] = float((P ** 2).sum())
        f[f'gent_{name}'] = float(-((P * np.log(P + 1e-12)).sum()))
        f[f'gh_{name}'] = float((P / (1 + (ii - jj) ** 2)).sum())
    return f

rows = {}
for uid in dom_ids:
    rows[uid] = radiomic(np.load(os.path.join(CROPS, f'{uid}.npy')))
Xr = pd.DataFrame.from_dict(rows, orient='index')

atlas = atlas.set_index('uid')
atlas_c = atlas.loc[dom_ids]
y = labels.set_index('uid').loc[dom_ids, 'is_pathologic'].values

formulas = {
    'atlas': atlas_c,
    'radiomic': Xr,
    'atlas+radiomic': pd.concat([atlas_c, Xr], axis=1),
    'atlas+logradiomic': pd.concat([atlas_c, np.log1p(np.abs(Xr))], axis=1),
}
for name, Xdf in formulas.items():
    X = Xdf.values
    aucs, lls = [], []
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(X, y):
        m = LogisticRegression(C=1.0, max_iter=3000).fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        aucs.append(roc_auc_score(y[va], p)); lls.append(log_loss(y[va], np.clip(p, 1e-6, 1 - 1e-6)))
    print(f'{name:20s} AUC {np.mean(aucs):.4f}+/-{np.std(aucs):.4f}  LL {np.mean(lls):.4f}')
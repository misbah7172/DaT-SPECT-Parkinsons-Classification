"""Quick signal test: radiomic features from aligned striatal crops vs within-site ceiling (0.902)."""
import os, sys, time
import numpy as np
import pandas as pd
from scipy import stats
import scipy.ndimage as ndi
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

CROPS = r'E:\DaT\aligned_cache\crops'
labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv') .merge(pd.read_csv(r'E:\DaT\Dataset\site_labels.csv'), on='uid')

dom = labels[labels['pseudo_site'] == '2.5x2.5x2.5'].copy()
print('dominant site n =', len(dom))

def radiomic(crop):
    f = {}
    v = crop.ravel()
    levels = 32
    mn, mx = np.percentile(v, 1), np.percentile(v, 99.5)
    if mx <= mn: mx = mn + 1e-6
    q = np.clip(((crop - mn) / (mx - mn) * (levels - 1)).astype(int), 0, levels - 1)

    # intensity histogram features
    hist, _ = np.histogram(v, bins=24, range=(mn, mx))
    f['hist_ent'] = -(((hist / max(hist.sum(),1)) * np.log((hist / max(hist.sum(),1)) + 1e-9)).sum())
    f['hist_energy'] = ((hist / max(hist.sum(),1)) ** 2).sum()
    for qn in range(1, 25):
        pass
    # percentile band radiance
    for p in (50, 75, 90, 95, 97, 99):
        f[f'p{p}'] = np.percentile(v, p)
    f['range'] = np.percentile(v, 99) - np.percentile(v, 50)
    f['max'] = v.max(); f['mean'] = v.mean(); f['std'] = v.std()

    # spatial: for each axis-slice profile, quantile of centroid etc.
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        prof = crop.max(axis=ax)  # intensity max along axis -> 2D map
        vv = prof.ravel()
        if vv.max() > vv.min():
            pid = (vv - vv.min()) / (vv.max() - vv.min())
            c = ndi.center_of_mass(pid)
            if len(c) == 2:
                f[f'com_{name}_surface'] = c[0]
                f[f'com_{name}_surface2'] = c[1]
            f[f'std_{name}_surface'] = np.std(pid)

    # GLCM (2D cooccurrence) on middle slice for each axis
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        sl = q.take(q.shape[ax] // 2, axis=ax)
        P = np.zeros((levels, levels), dtype=np.float64)
        sl = np.ascontiguousarray(sl)
        # horizontal pairs
        H, W = sl.shape
        for i in range(H):
            row = sl[i]
            np.add.at(P, (row[:-1], row[1:]), 1)
            np.add.at(P, (row[1:], row[:-1]), 1)
        S = P.sum()
        if S > 0: P /= S
        p_x = P.sum(1)
        p_y = P.sum(0)
        # contrast, energy
        ii = np.arange(levels)[:, None]; jj = np.arange(levels)[None, :]
        f[f'glcm_contrast_{name}'] = float(((P * ((ii - jj) ** 2)).sum()))
        f[f'glcm_energy_{name}'] = float((P ** 2).sum())
        f[f'glcm_entropy_{name}'] = float(-((P * np.log(P + 1e-12)).sum()))
        # homogeneity
        f[f'glcm_homog_{name}'] = float((P / (1 + (ii - jj) ** 2)).sum())
    return f

rows = {}
t0 = time.time()
for _, r in dom.iterrows():
    uid = r['uid']
    crop = np.load(os.path.join(CROPS, f'{uid}.npy'))
    rows[uid] = radiomic(crop)
    if len(rows) % 150 == 0:
        print(f'{len(rows)} done in {time.time()-t0:.0f}s', flush=True)

X = pd.DataFrame.from_dict(rows, orient='index')
y = dom.set_index('uid')['is_pathologic']
print('features:', X.shape)

from sklearn.model_selection import StratifiedKFold
Xarr = np.log1p(np.abs(X.values))
y = y.values  # numpy
aucs, lls = [], []
for tr, va in StratifiedKFold(5, shuffle=True, random_state=42).split(Xarr, y):
    m = LogisticRegression(C=1.0, max_iter=3000).fit(Xarr[tr], y[tr])
    p = m.predict_proba(Xarr[va])[:, 1]
    aucs.append(roc_auc_score(y[va], p)); lls.append(log_loss(y[va], p))
print('Radiomic-only within dominant site: AUC %.4f +/- %.4f  LL %.4f +/- %.4f'
      % (np.mean(aucs), np.std(aucs), np.mean(lls), np.std(lls)))
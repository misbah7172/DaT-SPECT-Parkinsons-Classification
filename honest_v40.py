"""V40: HONEST nested eval including deep-3D per-seed streams (cnn3dr2_seed*).
No validation-fold leakage: per-site calibration and blend weights fit on outer-train only.
"""
import os, pickle
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

OUT = r'E:\DaT\v26_oof'
y = np.load(os.path.join(OUT, 'y.npy'))
groups = np.load(os.path.join(OUT, 'groups.npy'), allow_pickle=True)
with open(os.path.join(OUT, 'oof_a.pkl'), 'rb') as f:
    A = pickle.load(f)
with open(os.path.join(OUT, 'oof_b.pkl'), 'rb') as f:
    B = pickle.load(f)
with open(os.path.join(OUT, 'oof_c.pkl'), 'rb') as f:
    C = pickle.load(f)
oof_a = np.column_stack(A['oof'])
oof_b = np.column_stack(B['oof'])
oof_c = np.column_stack(C['oof'])
blend_d = np.load(os.path.join(OUT, 'blend_d.npy'))
with open(r'E:\DaT\v30_oof\oof_roi.pkl', 'rb') as f:
    ROI = pickle.load(f)
oof_roi = np.column_stack(ROI['oof'])

streams = [oof_a, oof_b, oof_c, blend_d.reshape(-1, 1), oof_roi]
cnn_files = ['cnn3d', 'cnn25b', 'cnn25s', 'cnn25r2', 'cnn3dr', 'cnn25v2']
for name in cnn_files:
    streams.append(np.load(rf'E:\DaT\v30_oof\{name}_oof.npy').reshape(-1, 1))
deep_seeds = sorted(f for f in os.listdir(r'E:\DaT\v30_oof')
                    if f.startswith('cnn3dr2_seed') and f.endswith('.npy'))
deep_oof = None
if deep_seeds:
    deep_oof = np.mean([np.load(os.path.join(r'E:\DaT\v30_oof', f)) for f in deep_seeds], axis=0)
    streams.append(deep_oof.reshape(-1, 1))
    print('deep-3D seeds:', deep_seeds, 'stream added')

S = np.hstack(streams)
n, p = S.shape
print('streams:', p, 'n:', n)

eps = 1e-6


def platt_fit(pp, yy):
    l = logit(np.clip(pp, eps, 1 - eps))

    def loss(q):
        return log_loss(yy, np.clip(expit(q[0] * l + q[1]), eps, 1 - eps))
    r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
    return r.x[0], r.x[1]


def opt_w(Mt, yt):
    def loss(w):
        return log_loss(yt, np.clip(Mt @ w, eps, 1 - eps))
    r = minimize(loss, np.ones(Mt.shape[1]) / Mt.shape[1], method="L-BFGS-B",
                 bounds=[(0, 1)] * Mt.shape[1])
    return r.x / r.x.sum()


def run(K, seed, calibrator):
    oof = np.zeros(n)
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(S, y, groups):
        St, Sa = S[tr], S[va]
        aucs = np.array([roc_auc_score(y[tr], St[:, j]) for j in range(p)])
        cand = np.argsort(aucs)[::-1][:30]
        sel = []
        for _ in range(K):
            best = 1e9; bi = None
            for j in cand:
                if j in sel:
                    continue
                Mt = St[:, sel + [j]]
                w = opt_w(Mt, y[tr])
                s = log_loss(y[tr], np.clip(Mt @ w, eps, 1 - eps))
                if s < best:
                    best = s; bi = j
            if bi is None:
                break
            sel.append(bi)
        Mt = St[:, sel]
        w = opt_w(Mt, y[tr])
        pt = np.clip(Mt @ w, eps, 1 - eps)
        pv = np.clip(Sa[:, sel] @ w, eps, 1 - eps)
        oof[va] += calibrator(pt, pv, y[tr], y[va], groups[tr], groups[va])
    return oof


def cal_shift(pt, pv, yt, yv, gt, gv):
    res = pv.copy()
    for site in np.unique(gt):
        mt = gt == site
        if mt.sum() > 10 and len(np.unique(yt[mt])) > 1:
            lt = logit(pt[mt])
            lv = logit(np.clip(pv[gv == site], eps, 1 - eps))
            lo = np.log((yt[mt].mean() + 1e-6) / (1 - yt[mt].mean() + 1e-6))
            shift = lo - lt.mean()
            res[gv == site] = expit(lv + shift)
    return res


for name, cal in [('none', lambda pt, pv, yt, yv, gt, gv: pv), ('shift', cal_shift)]:
    o = np.zeros(n)
    for seed in [42, 777, 2024]:
        o += run(16, seed, cal)
    o /= 3
    print(f'{name:6s}: AUC={roc_auc_score(y, o):.4f} LL={log_loss(y, np.clip(o, eps, 1 - eps)):.4f}', flush=True)
    for site in np.unique(groups):
        m = groups == site
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        print(f'   {site}: n={m.sum()} AUC={roc_auc_score(y[m], o[m]):.4f} LL={log_loss(y[m], np.clip(o[m], eps, 1 - eps)):.4f}')

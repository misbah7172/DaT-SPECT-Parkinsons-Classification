"""V38: HONEST nested with per-site calibration fit ONLY on outer-train folds.
Two calibrators: Platt and per-site mean-shift on logit. No validation-fold leakage.
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
cnn3d = np.load(r'E:\DaT\v30_oof\cnn3d_oof.npy')
cnn25b = np.load(r'E:\DaT\v30_oof\cnn25b_oof.npy')
cnn25s = np.load(r'E:\DaT\v30_oof\cnn25s_oof.npy')
cnn25r2 = np.load(r'E:\DaT\v30_oof\cnn25r2_oof.npy')
cnn3dr = np.load(r'E:\DaT\v30_oof\cnn3dr_oof.npy')
cnn25v2 = np.load(r'E:\DaT\v30_oof\cnn25v2_oof.npy')

eps = 1e-6
S = np.hstack([oof_a, oof_b, oof_c, blend_d.reshape(-1,1), oof_roi,
               cnn3d.reshape(-1,1), cnn25b.reshape(-1,1), cnn25s.reshape(-1,1),
               cnn25r2.reshape(-1,1), cnn3dr.reshape(-1,1), cnn25v2.reshape(-1,1)])
n, p = S.shape
print('streams:', p, 'n:', n)

def platt_fit(pp, yy):
    l = logit(np.clip(pp, eps, 1-eps))
    def loss(q):
        return log_loss(yy, np.clip(expit(q[0]*l + q[1]), eps, 1-eps))
    r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
    return r.x[0], r.x[1]

def opt_w(Mt, yt):
    def loss(w):
        return log_loss(yt, np.clip(Mt @ w, eps, 1-eps))
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
                if j in sel: continue
                Mt = St[:, sel + [j]]
                w = opt_w(Mt, y[tr])
                s = log_loss(y[tr], np.clip(Mt @ w, eps, 1-eps))
                if s < best:
                    best = s; bi = j
            if bi is None: break
            sel.append(bi)
        Mt = St[:, sel]
        w = opt_w(Mt, y[tr])
        pt = np.clip(Mt @ w, eps, 1-eps)
        pv = np.clip(Sa[:, sel] @ w, eps, 1-eps)
        oof[va] += calibrator(pt, pv, y[tr], y[va], groups[tr], groups[va])
    return oof

def cal_platt(pt, pv, yt, yv, gt, gv):
    res = pv.copy()
    for site in np.unique(gt):
        mt = gt == site
        if mt.sum() > 10 and len(np.unique(yt[mt])) > 1:
            a, b = platt_fit(pt[mt], yt[mt])
            mv = gv == site
            if mv.sum():
                res[mv] = expit(a * logit(np.clip(pv[mv], eps, 1-eps)) + b)
    return res

def cal_shift(pt, pv, yt, yv, gt, gv):
    """per-site additive shift in logit space toward site-optimal threshold."""
    res = pv.copy()
    for site in np.unique(gt):
        mt = gt == site
        if mt.sum() > 10 and len(np.unique(yt[mt])) > 1:
            lt = logit(pt[mt])
            lv = logit(np.clip(pv[gv == site], eps, 1-eps))
            # shift so mean logit matches empirical log-odds of site
            lo = np.log((yt[mt].mean() + 1e-6) / (1 - yt[mt].mean() + 1e-6))
            shift = lo - lt.mean()
            res[gv == site] = expit(lv + shift)
    return res

for name, cal in [('shift', cal_shift), ('none', lambda pt,pv,yt,yv,gt,gv: pv)]:
    o = np.zeros(n)
    for seed in [42]:
        o += run(16, seed, cal)
    o /= 1
    print(f'{name:6s}: AUC={roc_auc_score(y, o):.4f} LL={log_loss(y, np.clip(o,eps,1-eps)):.4f}', flush=True)
    for site in np.unique(groups):
        m = groups == site
        if m.sum() < 20 or len(np.unique(y[m])) < 2: continue
        print(f'   {site}: n={m.sum()} AUC={roc_auc_score(y[m], o[m]):.4f} LL={log_loss(y[m], np.clip(o[m],eps,1-eps)):.4f}')

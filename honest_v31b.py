"""V31b: faster greedy selection — use summed OOF over 3 seeds of the same internal fold
to reduce weight-fitting noise, run fewer seeds, and cache per-fold results.
Optimization: greedy selection is per-seed/per-fold. Caching by (seed,fold) impossible
across K; instead pick K by getting per-fold selections for the UNION and report.
Simpler: just run K=10 and K=14 with 1 seed to confirm trend, saving to files.
"""
import os, pickle, time
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

eps = 1e-6
def lg(M): return np.log(np.clip(M, eps, 1-eps) / np.clip(1-M, eps, 1-eps))
S = np.hstack([oof_a, oof_b, oof_c, blend_d.reshape(-1,1), oof_roi])
n, p = S.shape
print('streams:', p)

def platt_fit(p2, y2):
    l = logit(np.clip(p2, eps, 1-eps))
    def loss(q):
        return log_loss(y2, np.clip(expit(q[0]*l + q[1]), eps, 1-eps))
    r = minimize(loss, [1.0, 0.0], method="Nelder-Mead")
    return r.x[0], r.x[1]

def opt_w(Mt, yt):
    def loss(w):
        return log_loss(yt, np.clip(Mt @ w, eps, 1-eps))
    r = minimize(loss, np.ones(Mt.shape[1]) / Mt.shape[1], method="L-BFGS-B",
                 bounds=[(0, 1)] * Mt.shape[1])
    return r.x / r.x.sum()

def site_cal(p2, y2, groups):
    res = p2.copy()
    for site in np.unique(groups):
        m = groups == site
        if m.sum() > 10 and len(np.unique(y2[m])) > 1:
            a, b = platt_fit(p2[m], y2[m])
            res[m] = expit(a * logit(np.clip(p2[m], eps, 1-eps)) + b)
    return res

def greedy_select(St, yt, k):
    sel = []
    available = list(range(St.shape[1]))
    for _ in range(k):
        best_score = 1e9; best_j = None; best_w = None
        for j in available:
            cand = sel + [j]
            Mt = St[:, cand]
            w = opt_w(Mt, yt)
            score = log_loss(yt, np.clip(Mt @ w, eps, 1-eps))
            if score < best_score:
                best_score = score; best_j = j; best_w = w
        if best_j is None: break
        sel.append(best_j); available.remove(best_j)
        Mt = St[:, sel]
        w = opt_w(Mt, yt)
    return sel, w

for K in [10, 14]:
    oof = np.zeros(n); oof_sc = np.zeros(n)
    t0 = time.time()
    for seed in [42]:
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(S, y, groups):
            sel, w = greedy_select(S[tr], y[tr], K)
            pv = S[va][:, sel] @ w
            oof[va] += pv
            oof_sc[va] += np.clip(site_cal(pv, y[va], groups[va]), eps, 1-eps)
    print(f'K={K}: AUC={roc_auc_score(y, oof):.4f} LL={log_loss(y, np.clip(oof,eps,1-eps)):.4f} | SC AUC={roc_auc_score(y, oof_sc):.4f} LL={log_loss(y, oof_sc):.4f} ({time.time()-t0:.0f}s)', flush=True)
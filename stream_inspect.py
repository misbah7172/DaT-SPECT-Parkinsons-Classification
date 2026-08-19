"""Inspect which streams greedy selection actually picks (diagnostic)."""
import os, pickle
import numpy as np
from scipy.optimize import minimize
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
eps = 1e-6

names = (A['names'] + B['names'] + C['names'] + ['mlp_ens'] + ROI['names'] + ['cnn3d'] + ['cnn25b'])
S = np.hstack([oof_a, oof_b, oof_c, blend_d.reshape(-1,1), oof_roi, cnn3d.reshape(-1,1), cnn25b.reshape(-1,1)])

def opt_w(Mt, yt):
    def loss(w):
        return log_loss(yt, np.clip(Mt @ w, eps, 1-eps))
    r = minimize(loss, np.ones(Mt.shape[1]) / Mt.shape[1], method="L-BFGS-B",
                 bounds=[(0, 1)] * Mt.shape[1])
    return r.x / r.x.sum()

for seed in [42, 777]:
    tr, va = next(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(S, y, groups))
    St = S[tr]
    aucs = np.array([roc_auc_score(y[tr], St[:, j]) for j in range(S.shape[1])])
    cand = np.argsort(aucs)[::-1][:30]
    sel = []
    for _ in range(12):
        best = 1e9; bi = None
        for j in cand:
            if j in sel: continue
            Mt = St[:, sel + [j]]
            w = opt_w(Mt, y[tr])
            s = log_loss(y[tr], np.clip(Mt @ w, eps, 1-eps))
            if s < best:
                best = s; bi = j
        sel.append(bi)
    print(f'seed {seed} selected {len(sel)}:')
    for j in sel:
        print(f'   {names[j]:35s} AUC={aucs[j]:.4f}')
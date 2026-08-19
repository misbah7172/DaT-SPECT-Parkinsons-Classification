"""Diagnostic: meta-learner robustness in honest nested CV.
Compare exact weight fit vs regularized LR on logits vs top-k equal averages."""
import os, pickle
import numpy as np
from scipy.optimize import minimize
from scipy.special import logit, expit
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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

# streams: 5 groups
streams = [oof_a, oof_b, oof_c, blend_d.reshape(-1,1), oof_roi]
S = np.hstack(streams)  # probabilities
L = lg(S)               # logits
n = len(y)

def opt_w(Mt, yt):
    def loss(w):
        return log_loss(yt, np.clip(Mt @ w, eps, 1-eps))
    r = minimize(loss, np.ones(Mt.shape[1]) / Mt.shape[1], method="L-BFGS-B",
                 bounds=[(0, 1)] * Mt.shape[1])
    return r.x / r.x.sum()

results = {}
seeds = [42, 777, 2024]
for strategy in ['exact', 'lr_c001', 'lr_c01', 'lr_c1', 'top5', 'top10']:
    oof = np.zeros(n)
    for seed in seeds:
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(S, y, groups):
            St = S[tr]; Lt = L[tr]; yt = y[tr]
            if strategy == 'exact':
                w = opt_w(St, yt)
                pv = S[va] @ w
            elif strategy.startswith('lr'):
                C = float(strategy.split('c')[1]) / 100 if False else float(strategy.split('_c')[1])
                m = LogisticRegression(C=C, max_iter=5000).fit(StandardScaler().fit_transform(Lt), yt)
                pv = m.predict_proba(StandardScaler().fit_transform(L[va]))[:, 1]
            elif strategy.startswith('top'):
                k = int(strategy[3:])
                temps = np.full(S.shape[1], np.nan)
                for j in range(S.shape[1]):
                    temps[j] = roc_auc_score(yt, St[:, j])
                top = np.argsort(temps)[::-1][:k]
                pv = S[va][:, top].mean(axis=1)
            oof[va] += pv
    oof /= len(seeds)
    results[strategy] = (roc_auc_score(y, oof), log_loss(y, np.clip(oof, eps, 1-eps)))
    print(f'{strategy:10s}: AUC={results[strategy][0]:.4f} LL={results[strategy][1]:.4f}')
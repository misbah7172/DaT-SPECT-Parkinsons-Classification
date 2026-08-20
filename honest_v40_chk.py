"""V40 checkpointed: greedy K=16, 3 outer seeds, saves per-seed OOFs and final."""
import os, pickle, sys
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
oof_a = np.column_stack(A['oof']); oof_b = np.column_stack(B['oof']); oof_c = np.column_stack(C['oof'])
blend_d = np.load(os.path.join(OUT, 'blend_d.npy'))
with open(r'E:\DaT\v30_oof\oof_roi.pkl', 'rb') as f:
    ROI = pickle.load(f)
oof_roi = np.column_stack(ROI['oof'])
streams = [oof_a, oof_b, oof_c, blend_d.reshape(-1, 1), oof_roi]
for name in ['cnn3d', 'cnn25b', 'cnn25s', 'cnn25r2', 'cnn3dr', 'cnn25v2']:
    streams.append(np.load(rf'E:\DaT\v30_oof\{name}_oof.npy').reshape(-1, 1))
deep_oof = np.mean([np.load(os.path.join(r'E:\DaT\v30_oof', f))
                    for f in sorted(f for f in os.listdir(r'E:\DaT\v30_oof')
                    if f.startswith('cnn3dr2_seed') and f.endswith('.npy'))], axis=0)
deep6 = np.load(r'E:\DaT\v30_oof\cnn3d_deep6.npy')
streams.append(deep_oof.reshape(-1, 1))
streams.append(deep6.reshape(-1, 1))
S = np.hstack(streams)
n, p = S.shape
eps = 1e-6


def log(m):
    print(m, flush=True)


def opt_w(Mt, yt):
    def loss(w):
        return log_loss(yt, np.clip(Mt @ w, eps, 1 - eps))
    r = minimize(loss, np.ones(Mt.shape[1]) / Mt.shape[1], method='L-BFGS-B',
                 bounds=[(0, 1)] * Mt.shape[1])
    return r.x / r.x.sum()


def cal_temp(pt, pv, yt, yv, gt, gv):
    lp = logit(np.clip(pt, eps, 1 - eps))

    def loss(T):
        return log_loss(yt, np.clip(expit(lp / max(T, 1e-3)), eps, 1 - eps))
    r = minimize(loss, np.array([1.0]), method='Nelder-Mead')
    return expit(logit(np.clip(pv, eps, 1 - eps)) / r.x[0])


def run(K, seed, cal):
    oof = np.zeros(n)
    for fold, (tr, va) in enumerate(StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(S, y, groups)):
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
        oof[va] += cal(pt, pv, y[tr], y[va], groups[tr], groups[va])
    return oof


if __name__ == '__main__':
    K = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    SEEDS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [42, 777, 2024]
    cname = sys.argv[3] if len(sys.argv) > 3 else 'none'
    cal = {'none': lambda pt, pv, yt, yv, gt, gv: pv, 'temp': cal_temp}[cname]
    tag = f'v40t_{cname}'
    for s in SEEDS:
        f = rf'E:\DaT\v30_oof\{tag}_seed{s}.npy'
        if os.path.exists(f):
            log(f'skip seed {s} (exists)')
            continue
        log(f'== {cname} seed {s} start')
        o = run(K, s, cal)
        np.save(f, o)
        log(f'== {cname} seed {s}: AUC={roc_auc_score(y, o):.4f} LL={log_loss(y, np.clip(o, eps, 1 - eps)):.4f} saved')
    o = np.zeros(n)
    for s in SEEDS:
        o += np.load(rf'E:\DaT\v30_oof\{tag}_seed{s}.npy')
    o /= len(SEEDS)
    log(f'FINAL {cname}: AUC={roc_auc_score(y, o):.4f} LL={log_loss(y, np.clip(o, eps, 1 - eps)):.4f}')
    for site in np.unique(groups):
        m = groups == site
        if m.sum() < 20 or len(np.unique(y[m])) < 2:
            continue
        log(f'  {site}: n={m.sum()} AUC={roc_auc_score(y[m], o[m]):.4f} LL={log_loss(y[m], np.clip(o[m], eps, 1 - eps)):.4f}')
    np.save(rf'E:\DaT\{tag}_honest_oof.npy', o)
    np.save(r'C:\Users\misba\AppData\Local\Temp\opencode\best_honest_oof.npy', o)
    log(f'saved {tag}_honest_oof.npy')

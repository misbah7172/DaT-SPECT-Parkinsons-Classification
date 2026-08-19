import pandas as pd, numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
site = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv')
geom = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
sbr = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv')
atlas = pd.read_csv(r'E:\DaT\Dataset\atlas_features_train.csv')

merged = labels.merge(site, on='uid').merge(geom, on='uid')
merged = merged.merge(sbr, on='uid')
atlas2 = atlas.rename(columns={c: f'at_{c}' for c in atlas.columns if c != 'uid'})
merged = merged.merge(atlas2, on='uid')
merged = merged.dropna()

print('Per-site stats:')
for s, g in merged.groupby('pseudo_site'):
    print(f'  {s}: n={len(g)}, pos={g["is_pathologic"].mean():.2f}')

print()
print('=== Within-site AUC (sbr + atlas, LR, per-site models) ===')
skip = {'uid','is_pathologic','pseudo_site','site_id','sx','sy','sz','voxel_vol','voxel_side','label'}
feat_cols = [c for c in merged.columns if c not in skip]
X_all = merged[feat_cols].values.astype(float)
X_all = np.nan_to_num(np.log1p(np.clip(X_all,0,None)))
y_all = merged['is_pathologic'].values
grp = merged['pseudo_site'].astype(str).values

oof = np.zeros(len(y_all))
covered = np.zeros(len(y_all))
for s in np.unique(grp):
    idx = np.where(grp==s)[0]
    if len(idx) < 30:
        continue
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    for tr, va in skf.split(idx, y_all[idx]):
        sc = StandardScaler().fit(X_all[idx[tr]])
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X_all[idx[tr]]), y_all[idx[tr]])
        oof[idx[va]] = lr.predict_proba(sc.transform(X_all[idx[va]]))[:,1]
        covered[idx[va]] = 1
mask = covered > 0
print(f'  Per-site models OOF (n={mask.sum()}): AUC={roc_auc_score(y_all[mask],oof[mask]):.4f} LL={log_loss(y_all[mask],oof[mask]):.4f}')

# Per-site AUC breakdown
print()
print('=== Per-site AUC breakdown ===')
for s in np.unique(grp):
    idx = np.where(grp==s)[0]
    mask_s = mask[idx]
    if mask_s.sum() < 20 or len(np.unique(y_all[idx][mask_s])) < 2:
        print(f'  {s}: n={len(idx)} - insufficient')
        continue
    auc = roc_auc_score(y_all[idx][mask_s], oof[idx][mask_s])
    ll = log_loss(y_all[idx][mask_s], oof[idx][mask_s])
    print(f'  {s}: n={len(idx)}, AUC={auc:.4f}, LL={ll:.4f}')

# Also try global model (all data)
print()
print('=== Global model (sbr + atlas) ===')
for seed in [42, 777, 2024]:
    oofg = np.zeros(len(y_all))
    for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X_all, y_all, grp):
        sc = StandardScaler().fit(X_all[tr])
        lr = LogisticRegression(C=1.0, max_iter=3000).fit(sc.transform(X_all[tr]), y_all[tr])
        oofg[va] = lr.predict_proba(sc.transform(X_all[va]))[:,1]
    print(f'  Global seed {seed}: AUC={roc_auc_score(y_all,oofg):.4f} LL={log_loss(y_all,oofg):.4f}')

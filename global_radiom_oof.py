"""Global OOF: existing 4 datasets + radiomics, group-stratified CV."""
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import StratifiedGroupKFold

labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv').merge(pd.read_csv(r'E:\DaT\Dataset\site_labels.csv'), on='uid').merge(pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv'), on='uid')
sbr = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv').set_index('uid').drop(columns=['label'], errors='ignore')
phys_raw = pd.read_csv(r'E:\DaT\Dataset\phys_features_train.csv').set_index('uid')
phys = phys_raw.rename(columns={c: f'ph_{c}' for c in phys_raw.columns})
morph = pd.read_csv(r'E:\DaT\Dataset\sbr_features_morph_train.csv').set_index('uid').drop(columns=['label'], errors='ignore')
atlas = pd.read_csv(r'E:\DaT\Dataset\atlas_features_train.csv').set_index('uid').drop(columns=['label'], errors='ignore')
radiom = pd.read_csv(r'E:\DaT\radiomic_features.csv').set_index('uid')

merged = labels.set_index('uid')
fac = (15.625 / merged['voxel_vol'].values) ** (1/3)
sbrc = sbr.loc[merged.index].copy()
for c in sbrc.columns:
    if c.startswith('sbr_'):
        sbrc[c] = sbrc[c] * fac
merged = pd.concat([merged, sbrc, phys.loc[merged.index], morph.loc[merged.index], atlas.loc[merged.index], radiom.loc[merged.index]], axis=1)
merged = merged.dropna()
y = merged['is_pathologic'].values
grp = merged['pseudo_site'].astype(str).values

groups = {
    'sbr': [c for c in sbr.columns],
    'morph': [c for c in morph.columns],
    'phys': [c for c in phys.columns],
    'atlas': [c for c in atlas.columns],
    'radiom': [c for c in radiom.columns],
    'sbr+morph+phys+atlas': list(sbr.columns) + list(morph.columns) + list(phys.columns) + list(atlas.columns),
    'sbr+atlas+radiom': list(sbr.columns) + list(atlas.columns) + list(radiom.columns),
    'ALL': list(sbr.columns) + list(morph.columns) + list(phys.columns) + list(atlas.columns) + list(radiom.columns),
}

def run(cols, seeds=(42, 777, 2024)):
    X = merged[cols].values.astype(np.float64)
    X = np.nan_to_num(X)
    aucs, lls = [], []
    for seed in seeds:
        for tr, va in StratifiedGroupKFold(5, shuffle=True, random_state=seed).split(X, y, grp):
            Xtr = np.log1p(np.abs(X[tr])); Xva = np.log1p(np.abs(X[va]))
            sel = np.argsort(mutual_info_classif(Xtr, y[tr], random_state=seed))[::-1][:60]
            m = LogisticRegression(C=1.31, max_iter=30000).fit(Xtr[:, sel], y[tr])
            p = m.predict_proba(Xva[:, sel])[:, 1]
            aucs.append(roc_auc_score(y[va], p)); lls.append(log_loss(y[va], np.clip(p, 1e-6, 1-1e-6)))
    return np.mean(aucs), np.mean(lls)

for name, cols in groups.items():
    a, ll = run(cols)
    print(f'{name:26s} global OOF AUC {a:.4f}  LL {ll:.4f}')
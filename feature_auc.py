import pandas as pd, numpy as np
from sklearn.metrics import roc_auc_score

labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
sbr = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv')
merged = labels.merge(sbr.drop(columns=['label']), on='uid').dropna()

# Focus on dominant site
site = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv')
merged = merged.merge(site, on='uid')
big = merged[merged['pseudo_site']=='2.5x2.5x2.5']
print(f'Big site n={len(big)}')
y = big['is_pathologic'].values

cols = [c for c in big.columns if c not in ('uid','is_pathologic','pseudo_site','site_id','label')]
results = []
for c in cols:
    x = big[c].values
    if np.std(x)==0 or np.isnan(x).any():
        continue
    auc = roc_auc_score(y, x) if len(np.unique(y))>1 else np.nan
    if auc > 0.5: auc = auc
    else: auc = 1-auc
    results.append((auc, c))
results.sort(reverse=True)
print('\nTop 30 individual feature AUC (big site):')
for auc, c in results[:30]:
    print(f'  {auc:.4f}  {c}')

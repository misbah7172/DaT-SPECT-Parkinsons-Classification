import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv').merge(pd.read_csv(r'E:\DaT\Dataset\site_labels.csv'), on='uid')
sbr = pd.read_csv(r'E:\DaT\Dataset\sbr_features_train.csv').set_index('uid')
uids = labels[labels['pseudo_site']=='2.5x2.5x2.5']['uid'].tolist()
y = labels.set_index('uid').loc[uids,'is_pathologic'].values
X = sbr.loc[uids].values.astype(float)
print('X finite:', np.isfinite(X).all(), 'y shape', y.shape)
from sklearn.feature_selection import mutual_info_classif
for seed in (42,):
    aucs=[]
    for tr,va in StratifiedKFold(5,shuffle=True,random_state=seed).split(X,y):
        Xtr=np.log1p(np.abs(X[tr])); Xva=np.log1p(np.abs(X[va]))
        sel=np.argsort(mutual_info_classif(Xtr,y[tr],random_state=seed))[::-1][:44]
        m=LogisticRegression(C=1.31,max_iter=30000).fit(Xtr[:,sel],y[tr])
        p=m.predict_proba(Xva[:,sel])[:,1]
        aucs.append(roc_auc_score(y[va],p))
    print('seed',seed,'aucs',[round(a,4) for a in aucs],'mean',round(np.mean(aucs),4))
"""Quick local test: run submission_v27 on 20 training scans."""
import os, sys, time
import numpy as np
import pandas as pd
import torch
sys.path.insert(0, r'E:\DaT\submission_v27')

from cnn_infer import DeepEnsemble, TemplateCache, load_aligned, to_model_input
from sbr_extractor import extract_features

W = r'E:\DaT\submission_v27\weights'
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load ensemble
ens = DeepEnsemble(W, device=device)
print(f"Loaded {len(ens.models)} models")

# Load template
template = np.load(r'E:\DaT\Dataset\atlas_template.npy', allow_pickle=True)
cache = TemplateCache(template)

# Load labels
labels_df = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
nifti_dir = r'E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip'

# Test first 20 scans
test_uids = labels_df['uid'].tolist()[:20]
test_y = labels_df['is_pathologic'].values[:20]

preds = []
t0 = time.time()
for i, uid in enumerate(test_uids):
    path = os.path.join(nifti_dir, f"{uid}.nii.gz")
    try:
        crop = to_model_input(path, template, cache)
        p_deep = ens.predict_one(crop)
        preds.append(p_deep)
        print(f"  {i+1}/20 {uid}: p={p_deep:.4f}, true={test_y[i]}")
    except Exception as e:
        print(f"  {i+1}/20 {uid}: ERROR {e}")
        preds.append(0.5)

elapsed = time.time() - t0
preds = np.array(preds)

from sklearn.metrics import roc_auc_score, log_loss
auc = roc_auc_score(test_y, preds)
ll = log_loss(test_y, np.clip(preds, 1e-7, 1-1e-7))

print(f"\n20-scan test ({elapsed:.0f}s):")
print(f"  AUC: {auc:.4f}")
print(f"  LL:  {ll:.4f}")
print(f"  Mean: {preds.mean():.4f}")
print(f"  Time per scan: {elapsed/20:.1f}s")

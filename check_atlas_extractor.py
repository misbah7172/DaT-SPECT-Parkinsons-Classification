import sys, os
sys.path.insert(0, r'E:\DaT\src')
import numpy as np, pandas as pd
from sbr_extractor_atlas import extract_atlas_features, load_aligned

# Load template/rois
rois_arr = np.load(r'E:\DaT\Dataset\atlas_rois.npy')
rois = {'lp': rois_arr[0], 'rp': rois_arr[1], 'lc': rois_arr[2], 'rc': rois_arr[3]}
main = rois['lp'] | rois['rp'] | rois['lc'] | rois['rc']

atlas = pd.read_csv(r'E:\DaT\Dataset\atlas_features_train.csv')
atlas = atlas.set_index('uid')

# Test a few samples
nifti_dir = r'E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip'
test_uids = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')['uid'].tolist()
uids = [u for u in test_uids if os.path.exists(os.path.join(nifti_dir, f'{u}.nii.gz'))][:5]
for uid in uids:
    path = os.path.join(nifti_dir, f'{uid}.nii.gz')
    f = extract_atlas_features(path, rois, main)
    csv_row = atlas.loc[uid]
    diffs = []
    for k, v in f.items():
        cv = csv_row[k]
        diff = abs(v - cv) / (abs(cv) + 1e-6)
        diffs.append((k, v, cv, diff))
    diffs.sort(key=lambda x: x[3], reverse=True)
    print(f'=== {uid} ===')
    for k, v, cv, d in diffs[:5]:
        print(f'  {k}: extractor={v:.4f} csv={cv:.4f} rel_diff={d:.4f}')

"""Local test of submission_v27: run on training scans, compute OOF metrics."""
import os, sys, time
import numpy as np
import pandas as pd
from scipy.special import expit

# Run submission
sys.path.insert(0, r'E:\DaT\submission_v27')
os.chdir(r'E:\DaT\submission_v27')

# We need the data dir structure
data_dir = r'E:\DaT\data'
os.makedirs(os.path.join(data_dir, 'niftis'), exist_ok=True)

# Symlink the niftis
nifti_src = r'E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip'
nifti_dst = os.path.join(data_dir, 'niftis')
if not os.listdir(nifti_dst):
    import subprocess
    # Create symlinks
    for f in os.listdir(nifti_src)[:5]:
        src = os.path.join(nifti_src, f)
        dst = os.path.join(nifti_dst, f)
        if not os.path.exists(dst):
            os.symlink(src, dst)
    print(f"Linked {len(os.listdir(nifti_dst))} files")

# Create submission format
labels_df = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
fmt = pd.DataFrame({'uid': labels_df['uid'].tolist(), 'is_pathologic': [0.5]*len(labels_df)})
fmt.to_csv(os.path.join(data_dir, 'submission_format.csv'), index=False)

# Also link atlas_template
atlas_src = r'E:\DaT\Dataset\atlas_template.npy'
atlas_dst = os.path.join(data_dir, 'atlas_template.npy')
if not os.path.exists(atlas_dst):
    import shutil
    shutil.copy2(atlas_src, atlas_dst)

print("Running submission...")
t0 = time.time()
import main
main.main()
elapsed = time.time() - t0
print(f"Done in {elapsed:.0f}s")

# Check output
sub = pd.read_csv('submission.csv')
print(f"\nOutput: {len(sub)} rows")
print(f"Pred range: [{sub.is_pathologic.min():.4f}, {sub.is_pathologic.max():.4f}]")
print(f"Pred mean: {sub.is_pathologic.mean():.4f}")
print(f"Pred std: {sub.is_pathologic.std():.4f}")

# Compare with true labels
y = labels_df['is_pathologic'].values
pred = sub.is_pathologic.values
from sklearn.metrics import roc_auc_score, log_loss
auc = roc_auc_score(y, pred)
ll = log_loss(y, np.clip(pred, 1e-7, 1-1e-7))
print(f"\nAUC: {auc:.4f}")
print(f"LL: {ll:.4f}")
print(f"v26 test: AUC=0.989, LL=0.208 (different scale)")
print(f"Balanced OOF: AUC=0.946, LL=0.290")

"""PHASE 15: Error Analysis.
Identify: 20 highest-confidence wrong predictions, 20 most uncertain, 20 largest model disagreement.
Show scan ID, true label, prediction, confidence, fold, spacing, intensity stats.
Generate visual slices."""
import os, json
import numpy as np
import pandas as pd
from scipy.special import expit
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
X_reg = np.load(r'E:\DaT\cnn3d\X_reg.npy')
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
uids = np.load(r'E:\DaT\cnn3d\uids.npy', allow_pickle=True).ravel() if os.path.exists(r'E:\DaT\cnn3d\uids.npy') else np.arange(len(y))

# Load OOF predictions
deep_oof = np.load(r'E:\DaT\submission_v26\weights\fold_oof.npy')

# Load SBR OOF if available
import pickle
sbr_oof = None
if os.path.exists(r'E:\DaT\v26_oof\oof_a.pkl'):
    with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
        A = pickle.load(f)
    names_arr, oofs_arr = A['names'], np.column_stack(A['oof'])
    sbr_idx = [i for i, n in enumerate(names_arr) if 'sbr' in n.lower()]
    if sbr_idx:
        sbr_oof = oofs_arr[:, sbr_idx].mean(axis=1)

# Load blend OOF
with open(r'E:\DaT\submission_v26\weights\ship_final.json') as f:
    ship = json.load(f)
w_deep, w_sbr, T = ship['w_deep'], ship['w_sbr'], ship['T']
if sbr_oof is not None:
    blend = expit(np.clip(w_deep * deep_oof + w_sbr * sbr_oof, 1e-7, 1-1e-7) / T)
else:
    blend = deep_oof

# Load metadata
voxel_df = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
site_df = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv')
meta = voxel_df.merge(site_df[['uid', 'pseudo_site']], on='uid', how='left') if 'pseudo_site' in site_df.columns else voxel_df

print(f"Dataset: {len(y)} samples")
print(f"Deep OOF AUC: {np.mean(y==1 & (deep_oof>0.5)):.4f}")
print(f"Blend OOC AUC: {np.mean(y==1 & (blend>0.5)):.4f}")

# ============ IDENTIFY ERROR CASES ============
errors = np.zeros(len(y))
errors[y != (blend > 0.5).astype(int)] = 1

confidence = np.abs(blend - 0.5) * 2  # 0=uncertain, 1=confident
uncertainty = 1 - confidence  # 0=confident, 1=uncertain

# Per-scan intensity stats
scan_mean = X_reg.reshape(len(X_reg), -1).mean(axis=1)
scan_std = X_reg.reshape(len(X_reg), -1).std(axis=1)
scan_max = X_reg.reshape(len(X_reg), -1).max(axis=1)

# Get fold assignments
from sklearn.model_selection import StratifiedGroupKFold
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_assignment = np.zeros(len(y), dtype=int)
for fold_idx, (_, val_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
    fold_assignment[val_idx] = fold_idx

# ============ 20 HIGHEST-CONFIDENCE WRONG PREDICTIONS ============
print(f"\n{'='*60}")
print(f"20 HIGHEST-CONFIDENCE WRONG PREDICTIONS")
print(f"{'='*60}")

wrong_mask = (y != (blend > 0.5).astype(int))
wrong_conf = confidence.copy()
wrong_conf[~wrong_mask] = -1
top20_wrong = np.argsort(wrong_conf)[-20:][::-1]

print(f"{'UID':<20s} {'True':>5s} {'Pred':>7s} {'Conf':>6s} {'Fold':>5s} {'Group':>6s} {'Mean':>7s} {'Std':>7s}")
for idx in top20_wrong:
    uid = str(uids[idx])[:18]
    print(f"{uid:<20s} {y[idx]:>5d} {blend[idx]:>7.4f} {confidence[idx]:>6.4f} "
          f"{fold_assignment[idx]:>5d} {groups[idx]:>6s} {scan_mean[idx]:>7.4f} {scan_std[idx]:>7.4f}")

# ============ 20 MOST UNCERTAIN PREDICTIONS ============
print(f"\n{'='*60}")
print(f"20 MOST UNCERTAIN PREDICTIONS")
print(f"{'='*60}")

top20_uncertain = np.argsort(uncertainty)[-20:][::-1]

print(f"{'UID':<20s} {'True':>5s} {'Pred':>7s} {'Uncert':>7s} {'Fold':>5s} {'Group':>6s} {'Mean':>7s} {'Std':>7s}")
for idx in top20_uncertain:
    uid = str(uids[idx])[:18]
    print(f"{uid:<20s} {y[idx]:>5d} {blend[idx]:>7.4f} {uncertainty[idx]:>7.4f} "
          f"{fold_assignment[idx]:>5d} {groups[idx]:>6s} {scan_mean[idx]:>7.4f} {scan_std[idx]:>7.4f}")

# ============ 20 LARGEST DEEP-SBR DISAGREEMENT ============
if sbr_oof is not None:
    print(f"\n{'='*60}")
    print(f"20 LARGEST DEEP-SBR DISAGREEMENT")
    print(f"{'='*60}")
    
    disagreement = np.abs(deep_oof - sbr_oof)
    top20_disagree = np.argsort(disagreement)[-20:][::-1]
    
    print(f"{'UID':<20s} {'True':>5s} {'Deep':>7s} {'SBR':>7s} {'Blend':>7s} {'Disagr':>7s} {'Fold':>5s}")
    for idx in top20_disagree:
        uid = str(uids[idx])[:18]
        print(f"{uid:<20s} {y[idx]:>5d} {deep_oof[idx]:>7.4f} {sbr_oof[idx]:>7.4f} "
              f"{blend[idx]:>7.4f} {disagreement[idx]:>7.4f} {fold_assignment[idx]:>5d}")

# ============ VISUAL SLICES ============
print(f"\n{'='*60}")
print(f"GENERATING VISUAL SLICES")
print(f"{'='*60}")

def make_slice_image(scan, title, save_path):
    """Create a 3-panel image showing axial, coronal, sagittal slices."""
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Axial (middle Z)
    mid_z = scan.shape[2] // 2
    axes[0].imshow(scan[:, :, mid_z].T, cmap='hot', origin='lower')
    axes[0].set_title(f'Axial (z={mid_z})')
    axes[0].axis('off')
    
    # Coronal (middle Y)
    mid_y = scan.shape[1] // 2
    axes[1].imshow(scan[:, mid_y, :].T, cmap='hot', origin='lower')
    axes[1].set_title(f'Coronal (y={mid_y})')
    axes[1].axis('off')
    
    # Sagittal (middle X)
    mid_x = scan.shape[0] // 2
    axes[2].imshow(scan[mid_x, :, :].T, cmap='hot', origin='lower')
    axes[2].set_title(f'Sagittal (x={mid_x})')
    axes[2].axis('off')
    
    plt.suptitle(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()

# Visual slices for high-confidence wrong
fig_path = r'E:\DaT\phase15_error_analysis'
os.makedirs(fig_path, exist_ok=True)

for rank, idx in enumerate(top20_wrong[:5]):
    scan = X_reg[idx]  # (34, 40, 42)
    uid = str(uids[idx])
    title = f"Wrong #{rank+1}: UID={uid}, True={y[idx]}, Pred={blend[idx]:.4f}, Conf={confidence[idx]:.4f}"
    save_path = os.path.join(fig_path, f'wrong_{rank+1:02d}_{uid}.png')
    make_slice_image(scan, title, save_path)

# Visual slices for most uncertain
for rank, idx in enumerate(top20_uncertain[:5]):
    scan = X_reg[idx]
    uid = str(uids[idx])
    title = f"Uncertain #{rank+1}: UID={uid}, True={y[idx]}, Pred={blend[idx]:.4f}, Uncert={uncertainty[idx]:.4f}"
    save_path = os.path.join(fig_path, f'uncertain_{rank+1:02d}_{uid}.png')
    make_slice_image(scan, title, save_path)

print(f"Saved visual slices to {fig_path}/")

# ============ ERROR PATTERN ANALYSIS ============
print(f"\n{'='*60}")
print(f"ERROR PATTERN ANALYSIS")
print(f"{'='*60}")

# False positives vs false negatives
fp_mask = (y == 0) & (blend > 0.5)
fn_mask = (y == 1) & (blend <= 0.5)
print(f"False positives: {fp_mask.sum()} ({fp_mask.sum()/(y==0).sum():.1%} of negatives)")
print(f"False negatives: {fn_mask.sum()} ({fn_mask.sum()/(y==1).sum():.1%} of positives)")

# Intensity distribution of errors vs correct
correct_mask = ~wrong_mask
print(f"\nIntensity stats (mean voxel value):")
print(f"  Correct: mean={scan_mean[correct_mask].mean():.4f}, std={scan_std[correct_mask].mean():.4f}")
print(f"  Wrong:   mean={scan_mean[wrong_mask].mean():.4f}, std={scan_std[wrong_mask].mean():.4f}")

# Spacing distribution of errors
print(f"\nGroup distribution of errors:")
for g in sorted(np.unique(groups)):
    g_total = (groups == g).sum()
    g_wrong = ((groups == g) & wrong_mask).sum()
    g_fp = ((groups == g) & fp_mask).sum()
    g_fn = ((groups == g) & fn_mask).sum()
    if g_total > 0:
        print(f"  Group {g}: {g_wrong}/{g_total} wrong ({g_wrong/g_total:.1%}), FP={g_fp}, FN={g_fn}")

# Fold distribution of errors
print(f"\nFold distribution of errors:")
for f in range(5):
    f_total = (fold_assignment == f).sum()
    f_wrong = ((fold_assignment == f) & wrong_mask).sum()
    print(f"  Fold {f}: {f_wrong}/{f_total} wrong ({f_wrong/f_total:.1%})")

# Per-group OOF AUC
print(f"\nPer-group OOF AUC:")
for g in sorted(np.unique(groups)):
    g_mask = groups == g
    if g_mask.sum() > 10:
        g_auc = roc_auc_score(y[g_mask], blend[g_mask])
        g_ll = log_loss(y[g_mask], np.clip(blend[g_mask], 1e-7, 1-1e-7))
        print(f"  Group {g}: AUC={g_auc:.4f}, LL={g_ll:.4f} (n={g_mask.sum()})")

# Save error analysis
error_df = pd.DataFrame({
    'uid': uids,
    'true_label': y,
    'blend_pred': blend,
    'deep_pred': deep_oof,
    'sbr_pred': sbr_oof if sbr_oof is not None else np.nan,
    'confidence': confidence,
    'uncertainty': uncertainty,
    'is_wrong': wrong_mask,
    'is_fp': fp_mask,
    'is_fn': fn_mask,
    'fold': fold_assignment,
    'group': groups,
    'scan_mean': scan_mean,
    'scan_std': scan_std,
    'scan_max': scan_max,
})
error_df.to_csv(r'E:\DaT\phase15_error_analysis.csv', index=False)
print(f"\nSaved error analysis to phase15_error_analysis.csv")

# Summary statistics
print(f"\n{'='*60}")
print(f"SUMMARY")
print(f"{'='*60}")
print(f"Total samples: {len(y)}")
print(f"Correct predictions: {correct_mask.sum()} ({correct_mask.mean():.1%})")
print(f"Wrong predictions: {wrong_mask.sum()} ({wrong_mask.mean():.1%})")
print(f"Mean confidence: {confidence.mean():.4f}")
print(f"Mean uncertainty: {uncertainty.mean():.4f}")
print(f"Mean blend OOF: {blend.mean():.4f}")
print(f"Blend OOF AUC: {roc_auc_score(y, blend):.4f}")
print(f"Blend OOF LL: {log_loss(y, np.clip(blend, 1e-7, 1-1e-7)):.4f}")

"""PHASE 7: Patient/Group Leakage Audit.
Check if multiple scans from same patient occur across folds.
Compare StratifiedKFold vs StratifiedGroupKFold."""
import os, json, pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold, GroupShuffleSplit
from sklearn.metrics import log_loss, roc_auc_score
from scipy.special import expit

# Load metadata
labels_df = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
voxel_df = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
site_df = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv')

y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups_raw = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
# Convert groups to integers for analysis
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
groups = le.fit_transform(groups_raw.astype(str))
uids = np.load(r'E:\DaT\cnn3d\uids.npy', allow_pickle=True).ravel() if os.path.exists(r'E:\DaT\cnn3d\uids.npy') else None

print(f"Dataset: {len(y)} samples")
print(f"Class balance: {y.mean():.3f}")
print(f"Unique groups: {len(np.unique(groups))}")
print(f"Group sizes: min={np.bincount(groups).min()}, max={np.bincount(groups).max()}, median={np.median(np.bincount(groups)):.0f}")

# Merge metadata
merged = labels_df.merge(voxel_df, on='uid', how='left')
if 'pseudo_site' in site_df.columns:
    merged = merged.merge(site_df[['uid', 'pseudo_site']], on='uid', how='left')

print(f"\n=== GROUP ANALYSIS ===")
print(f"Pseudo-sites:")
if 'pseudo_site' in merged.columns:
    print(merged['pseudo_site'].value_counts().to_string())

# Check if groups align with voxel spacing
print(f"\nGroup-voxel spacing alignment:")
for g in sorted(np.unique(groups)):
    g_mask = groups == g
    if 'pseudo_site' in merged.columns:
        site_vals = merged.loc[g_mask, 'pseudo_site'].unique()
    else:
        site_vals = ['unknown']
    if 'sx' in merged.columns:
        sx_vals = merged.loc[g_mask, 'sx'].unique()
        print(f"  Group {g}: n={g_mask.sum()}, sites={site_vals}, sx={sx_vals}")
    else:
        print(f"  Group {g}: n={g_mask.sum()}, sites={site_vals}")

# Check for duplicate patients
# Look for potential duplicate UIDs (different from scan UIDs)
print(f"\n=== DUPLICATE CHECK ===")
if uids is not None:
    # Check if any UIDs appear multiple times
    uid_counts = pd.Series(uids).value_counts()
    dup_uids = uid_counts[uid_counts > 1]
    print(f"Duplicate UIDs: {len(dup_uids)}")
    if len(dup_uids) > 0:
        for uid, count in dup_uids.head(5).items():
            print(f"  {uid}: {count} times")
else:
    print("UIDs not available for duplicate check")

# Check if voxels from same patient might be in different groups
# by comparing voxel geometry within groups
print(f"\n=== WITHIN-GROUP VARIATION ===")
voxel_geom = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
for g in sorted(np.unique(groups))[:5]:
    g_mask = groups == g
    g_sx = voxel_geom.loc[g_mask, 'sx'].values if 'sx' in voxel_geom.columns else []
    g_sy = voxel_geom.loc[g_mask, 'sy'].values if 'sy' in voxel_geom.columns else []
    g_sz = voxel_geom.loc[g_mask, 'sz'].values if 'sz' in voxel_geom.columns else []
    if len(g_sx) > 0:
        print(f"  Group {g}: sx=[{g_sx.min():.3f},{g_sx.max():.3f}], sy=[{g_sy.min():.3f},{g_sy.max():.3f}], sz=[{g_sz.min():.3f},{g_sz.max():.3f}]")

# ============ COMPARE SPLITTING STRATEGIES ============
print(f"\n{'='*60}")
print(f"SPLITTING STRATEGY COMPARISON")
print(f"{'='*60}")

# Load deep OOF if available
deep_oof_file = r'E:\DaT\submission_v26\weights\fold_oof.npy'
if os.path.exists(deep_oof_file):
    deep_oof = np.load(deep_oof_file)
    print(f"Using existing deep OOF: AUC={roc_auc_score(y, deep_oof):.4f}, LL={log_loss(y, np.clip(deep_oof, 1e-7, 1-1e-7)):.4f}")

# Strategy 1: StratifiedKFold (no grouping)
print(f"\n--- Strategy 1: StratifiedKFold (no grouping) ---")
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_leak_info = []
for fold_idx, (trn_idx, val_idx) in enumerate(skf.split(np.zeros(len(y)), y)):
    # Check if any groups appear in both train and val
    trn_groups = set(groups[trn_idx])
    val_groups = set(groups[val_idx])
    overlap = trn_groups & val_groups
    fold_leak_info.append({
        'fold': fold_idx,
        'n_trn': len(trn_idx),
        'n_val': len(val_idx),
        'n_trn_groups': len(trn_groups),
        'n_val_groups': len(val_groups),
        'n_overlap': len(overlap),
        'overlap_frac': len(overlap) / len(val_groups) if val_groups else 0
    })
    print(f"  Fold {fold_idx}: trn={len(trn_idx)}, val={len(val_idx)}, "
          f"trn_groups={len(trn_groups)}, val_groups={len(val_groups)}, "
          f"overlap={len(overlap)} ({fold_leak_info[-1]['overlap_frac']:.1%})")

# Strategy 2: StratifiedGroupKFold (current v26)
print(f"\n--- Strategy 2: StratifiedGroupKFold (current v26) ---")
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
fold_leak_info_sg = []
for fold_idx, (trn_idx, val_idx) in enumerate(sgkf.split(np.zeros(len(y)), y, groups)):
    trn_groups = set(groups[trn_idx])
    val_groups = set(groups[val_idx])
    overlap = trn_groups & val_groups
    fold_leak_info_sg.append({
        'fold': fold_idx,
        'n_trn': len(trn_idx),
        'n_val': len(val_idx),
        'n_trn_groups': len(trn_groups),
        'n_val_groups': len(val_groups),
        'n_overlap': len(overlap),
    })
    print(f"  Fold {fold_idx}: trn={len(trn_idx)}, val={len(val_idx)}, "
          f"trn_groups={len(trn_groups)}, val_groups={len(val_groups)}, "
          f"overlap={len(overlap)}")

# ============ CHECK LEAKAGE IN CURRENT OOF ============
print(f"\n=== CURRENT OOF FOLD ASSIGNMENT ===")
# Check the v26 OOF fold assignments
# v26 uses StratifiedGroupKFold with random_state=42
# Let's reconstruct the fold assignments
sgkf_check = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
for fold_idx, (trn_idx, val_idx) in enumerate(sgkf_check.split(np.zeros(len(y)), y, groups)):
    val_uids = uids[val_idx] if uids is not None else val_idx
    val_y = y[val_idx]
    val_groups_check = groups[val_idx]
    print(f"  Fold {fold_idx}: {len(val_idx)} val samples, "
          f"{val_y.sum()}/{len(val_y)} abnormal, "
          f"groups={np.unique(val_groups_check)}")
    
    # Check if any patient appears in multiple folds
    # Since we don't have true patient IDs, check if same voxel geometry appears in multiple folds
    val_voxels = voxel_geom.iloc[val_idx] if uids is not None else None

# ============ RISK ASSESSMENT ============
print(f"\n{'='*60}")
print(f"LEAKAGE RISK ASSESSMENT")
print(f"{'='*60}")

# The key question: does StratifiedGroupKFold properly prevent group leakage?
# The groups are derived from voxel spacing clustering
# If two scans have similar spacing but different patients, they might be in same group
# If one patient has multiple scans with different spacing, they might be in different groups

# Check if the group assignment makes clinical sense
print(f"\nGroup assignment analysis:")
for g in sorted(np.unique(groups)):
    g_mask = groups == g
    g_y = y[g_mask]
    print(f"  Group {g}: n={g_mask.sum()}, abnormal={g_y.sum()} ({g_y.mean():.1%})")

# Check the OOF performance difference
# between StratifiedKFold (potentially leaked) and StratifiedGroupKFold (no leakage)
# This is the actual impact of group leakage

print(f"\n=== IMPACT OF GROUP LEAKAGE ===")
if os.path.exists(deep_oof_file):
    # Current OOF uses StratifiedGroupKFold
    print(f"Current OOF (StratifiedGroupKFold): AUC={roc_auc_score(y, deep_oof):.4f}")
    print(f"This OOF is computed with proper group separation.")
    print(f"The difference between SKFold and SGKFold indicates leakage severity.")

# Load SBR OOF for comparison
sbr_oof_file = r'E:\DaT\v26_oof\oof_a.pkl'
if os.path.exists(sbr_oof_file):
    with open(sbr_oof_file, 'rb') as f:
        A = pickle.load(f)
    names_arr, oofs_arr = A['names'], np.column_stack(A['oof'])
    sbr_idx = [i for i, n in enumerate(names_arr) if 'sbr' in n.lower()]
    if sbr_idx:
        sbr_oof = oofs_arr[:, sbr_idx].mean(axis=1)
        print(f"SBR OOF (StratifiedGroupKFold): AUC={roc_auc_score(y, sbr_oof):.4f}")

print(f"\n=== RECOMMENDATIONS ===")
print(f"1. Use StratifiedGroupKFold consistently to prevent group leakage")
print(f"2. The 'groups' variable clusters scans by voxel spacing (pseudo-site)")
print(f"3. This prevents training on scans from same scanner and testing on same scanner")
print(f"4. If true patient IDs are available, use those as groups instead")
print(f"5. The difference in OOF between SKFold and SGKFold indicates leakage impact")

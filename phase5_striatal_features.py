"""PHASE 5: Explicit Striatal Features vs CNN.
Extract approximate striatal uptake features directly from registered crops.
Compare simple classifier on these features vs CNN OOF.
Also test combining them."""
import os, json, pickle
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

# Load data
X_reg = np.load(r'E:\DaT\cnn3d\X_reg.npy')
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
uids = np.load(r'E:\DaT\cnn3d\uids.npy', allow_pickle=True).ravel()

# Load atlas ROIs
atlas_rois = np.load(r'E:\DaT\Dataset\atlas_rois.npy')  # (4, 100, 100, 100) - lp, rp, lc, rc
print(f"Atlas ROIs shape: {atlas_rois.shape}")

# The registered crops are 34x40x42, while atlas is 100^3
# We need to understand the mapping between them
# Let's check the registration pipeline to understand the crop coordinates

# The atlas template is 100^3
atlas_template = np.load(r'E:\DaT\Dataset\atlas_template.npy')
print(f"Atlas template shape: {atlas_template.shape}")

# The crop is centered on the striatal region
# Let's extract features by applying ROI masks to the registered crops
# Since the registered crops are already aligned to the atlas, we need to
# map the 100^3 ROI masks to the 34x40x42 crop space

# The registration was done as: RAS -> isotropic 2.5mm -> COM-center -> NCC best-shift -> crop
# The crop is extracted from the 100^3 grid
# Let's check the registration code to understand the coordinate mapping

# From register_cnn.py / cnn_infer.py, the crop is:
# taken from the shifted 100^3 grid at specific coordinates
# Let's load the shifts to understand

shifts_file = r'E:\DaT\cnn3d\shifts.json'
if os.path.exists(shifts_file):
    with open(shifts_file) as f:
        shifts = json.load(f)
    print(f"Loaded shifts for {len(shifts)} scans")
else:
    print("shifts.json not found, trying alternative approach")

# The ROIs are in the 100^3 atlas space
# The registered crop X_reg is 34x40x42
# Let's look at how the crop was extracted from the aligned volume

# From build_aligned_cache.py, the crop is:
# x_start=33, y_start=30, z_start=29
# shape: 34, 40, 42
# This means the crop covers: x=[33:67], y=[30:70], z=[29:71] in the 100^3 grid

# Let's extract the ROI masks in the crop space
CROP_ORIGIN = (33, 30, 29)  # x_start, y_start, z_start
CROP_SHAPE = (34, 40, 42)   # dx, dy, dz

lp_mask = atlas_rois[0]  # left putamen
rp_mask = atlas_rois[1]  # right putamen
lc_mask = atlas_rois[2]  # left caudate
rc_mask = atlas_rois[3]  # right caudate

# Extract ROI masks in crop space
lp_crop = lp_mask[CROP_ORIGIN[0]:CROP_ORIGIN[0]+CROP_SHAPE[0],
                  CROP_ORIGIN[1]:CROP_ORIGIN[1]+CROP_SHAPE[1],
                  CROP_ORIGIN[2]:CROP_ORIGIN[2]+CROP_SHAPE[2]]
rp_crop = rp_mask[CROP_ORIGIN[0]:CROP_ORIGIN[0]+CROP_SHAPE[0],
                  CROP_ORIGIN[1]:CROP_ORIGIN[1]+CROP_SHAPE[1],
                  CROP_ORIGIN[2]:CROP_ORIGIN[2]+CROP_SHAPE[2]]
lc_crop = lc_mask[CROP_ORIGIN[0]:CROP_ORIGIN[0]+CROP_SHAPE[0],
                  CROP_ORIGIN[1]:CROP_ORIGIN[1]+CROP_SHAPE[1],
                  CROP_ORIGIN[2]:CROP_ORIGIN[2]+CROP_SHAPE[2]]
rc_crop = rc_mask[CROP_ORIGIN[0]:CROP_ORIGIN[0]+CROP_SHAPE[0],
                  CROP_ORIGIN[1]:CROP_ORIGIN[1]+CROP_SHAPE[1],
                  CROP_ORIGIN[2]:CROP_ORIGIN[2]+CROP_SHAPE[2]]

print(f"ROI masks in crop space: lp={lp_crop.sum()}, rp={rp_crop.sum()}, lc={lc_crop.sum()}, rc={rc_crop.sum()}")

# Define background ROI (non-striatal, non-zero)
striatal_mask = (lp_crop + rp_crop + lc_crop + rc_crop) > 0
all_nonzero = X_reg.reshape(len(X_reg), -1).max(axis=1) > 0  # at least one voxel > 0

# Extract features for each scan
print(f"\nExtracting striatal features for {len(X_reg)} scans...")
features = []

for i in range(len(X_reg)):
    scan = X_reg[i]  # (34, 40, 42)
    
    # Striatal uptake (mean intensity in ROI)
    lp_mean = scan[lp_crop > 0].mean() if lp_crop.sum() > 0 else 0
    rp_mean = scan[rp_crop > 0].mean() if rp_crop.sum() > 0 else 0
    lc_mean = scan[lc_crop > 0].mean() if lc_crop.sum() > 0 else 0
    rc_mean = scan[rc_crop > 0].mean() if rc_crop.sum() > 0 else 0
    
    # Background (non-striatal region)
    bg_mask = ~striatal_mask & (scan > 0)
    bg_mean = scan[bg_mask].mean() if bg_mask.sum() > 0 else 1e-6
    
    # Striatal/Background ratios (SBR)
    lp_sbr = lp_mean / (bg_mean + 1e-6)
    rp_sbr = rp_mean / (bg_mean + 1e-6)
    lc_sbr = lc_mean / (bg_mean + 1e-6)
    rc_sbr = rc_mean / (bg_mean + 1e-6)
    
    # Asymmetry features
    put_asym = (lp_mean - rp_mean) / (lp_mean + rp_mean + 1e-6)
    cau_asym = (lc_mean - rc_mean) / (lc_mean + rc_mean + 1e-6)
    left_total = lp_mean + lc_mean
    right_total = rp_mean + rc_mean
    lr_asym = (left_total - right_total) / (left_total + right_total + 1e-6)
    
    # Total striatal uptake
    total_sbr = lp_sbr + rp_sbr + lc_sbr + rc_sbr
    
    # Left/Right ratios
    lr_ratio = left_total / (right_total + 1e-6)
    rl_ratio = right_total / (left_total + 1e-6)
    
    # Putamen/Caudate ratios
    pc_ratio = (lp_mean + rp_mean) / (lc_mean + rc_mean + 1e-6)
    
    # Absolute values
    abs_put_asym = abs(lp_mean - rp_mean)
    abs_cau_asym = abs(lc_mean - rc_mean)
    
    # Intensity statistics within ROI
    lp_std = scan[lp_crop > 0].std() if lp_crop.sum() > 1 else 0
    rp_std = scan[rp_crop > 0].std() if rp_crop.sum() > 1 else 0
    lc_std = scan[lc_crop > 0].std() if lc_crop.sum() > 1 else 0
    rc_std = scan[rc_crop > 0].std() if rc_crop.sum() > 1 else 0
    
    # Asymmetry of intensities
    put_asym_std = abs(lp_std - rp_std)
    cau_asym_std = abs(lc_std - rc_std)
    
    features.append([
        lp_mean, rp_mean, lc_mean, rc_mean,
        lp_sbr, rp_sbr, lc_sbr, rc_sbr,
        put_asym, cau_asym, lr_asym,
        total_sbr, lr_ratio, rl_ratio, pc_ratio,
        abs_put_asym, abs_cau_asym,
        lp_std, rp_std, lc_std, rc_std,
        put_asym_std, cau_asym_std,
        lp_crop.sum(), rp_crop.sum(), lc_crop.sum(), rc_crop.sum(),
    ])

feature_names = [
    'lp_mean', 'rp_mean', 'lc_mean', 'rc_mean',
    'lp_sbr', 'rp_sbr', 'lc_sbr', 'rc_sbr',
    'put_asym', 'cau_asym', 'lr_asym',
    'total_sbr', 'lr_ratio', 'rl_ratio', 'pc_ratio',
    'abs_put_asym', 'abs_cau_asym',
    'lp_std', 'rp_std', 'lc_std', 'rc_std',
    'put_asym_std', 'cau_asym_std',
    'lp_voxels', 'rp_voxels', 'lc_voxels', 'rc_voxels',
]

X_feat = np.array(features)
print(f"Feature matrix: {X_feat.shape}")
print(f"Feature names: {feature_names}")

# Save features
feat_df = pd.DataFrame(X_feat, columns=feature_names)
feat_df.insert(0, 'uid', uids)
feat_df.to_csv(r'E:\DaT\striatal_features_from_crop.csv', index=False)

# ============ OOF Evaluation ============
print(f"\n{'='*60}")
print(f"PHASE 5: Striatal Features OOF")
print(f"{'='*60}")

sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

models = [
    ('LogReg', lambda: LogisticRegression(C=1.0, max_iter=1000)),
    ('ExtraTrees', lambda: ExtraTreesClassifier(n_estimators=200, min_samples_leaf=5)),
    ('GradBoost', lambda: GradientBoostingClassifier(n_estimators=200, max_depth=4, learning_rate=0.05)),
]

# Test on raw features
print(f"\n--- Raw features ({X_feat.shape[1]} features) ---")
for name, factory in models:
    oof = np.zeros(len(y))
    for trn_idx, val_idx in sgkf.split(X_feat, y, groups):
        sc = StandardScaler()
        X_trn = sc.fit_transform(X_feat[trn_idx])
        X_val = sc.transform(X_feat[val_idx])
        clf = factory()
        clf.fit(X_trn, y[trn_idx])
        oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
    brier = np.mean((y - oof) ** 2)
    print(f"  {name:15s}: AUC={auc:.4f}, LL={ll:.4f}, Brier={brier:.4f}")

# Test with derived features (add asymmetry ratios)
print(f"\n--- Derived features (add asymmetry ratios) ---")
X_derived = np.column_stack([
    X_feat,
    X_feat[:, 0] / (X_feat[:, 1] + 1e-6),  # lp/rp
    X_feat[:, 1] / (X_feat[:, 0] + 1e-6),  # rp/lp
    X_feat[:, 2] / (X_feat[:, 3] + 1e-6),  # lc/rc
    X_feat[:, 3] / (X_feat[:, 2] + 1e-6),  # rc/lc
    X_feat[:, 0] / (X_feat[:, 2] + 1e-6),  # lp/lc
    X_feat[:, 1] / (X_feat[:, 3] + 1e-6),  # rp/rc
])
print(f"Derived features: {X_derived.shape[1]}")

for name, factory in models:
    oof = np.zeros(len(y))
    for trn_idx, val_idx in sgkf.split(X_derived, y, groups):
        sc = StandardScaler()
        X_trn = sc.fit_transform(X_derived[trn_idx])
        X_val = sc.transform(X_derived[val_idx])
        clf = factory()
        clf.fit(X_trn, y[trn_idx])
        oof[val_idx] = clf.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y, oof)
    ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
    brier = np.mean((y - oof) ** 2)
    print(f"  {name:15s}: AUC={auc:.4f}, LL={ll:.4f}, Brier={brier:.4f}")

# Compare with CNN OOF
if os.path.exists(r'E:\DaT\submission_v26\weights\fold_oof.npy'):
    deep_oof = np.load(r'E:\DaT\submission_v26\weights\fold_oof.npy')
    deep_auc = roc_auc_score(y, deep_oof)
    deep_ll = log_loss(y, np.clip(deep_oof, 1e-7, 1-1e-7))
    print(f"\n--- CNN OOF (for reference) ---")
    print(f"  Deep OOF:        AUC={deep_auc:.4f}, LL={deep_ll:.4f}")

# Test combination: striatal features + CNN
if os.path.exists(r'E:\DaT\submission_v26\weights\fold_oof.npy'):
    print(f"\n--- Combination: CNN + Striatal features ---")
    X_combo = np.column_stack([X_feat, deep_oof])
    for name, factory in models:
        oof = np.zeros(len(y))
        for trn_idx, val_idx in sgkf.split(X_combo, y, groups):
            sc = StandardScaler()
            X_trn = sc.fit_transform(X_combo[trn_idx])
            X_val = sc.transform(X_combo[val_idx])
            clf = factory()
            clf.fit(X_trn, y[trn_idx])
            oof[val_idx] = clf.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y, oof)
        ll = log_loss(y, np.clip(oof, 1e-7, 1-1e-7))
        print(f"  {name:15s}: AUC={auc:.4f}, LL={ll:.4f}")

# Load SBR OOF if available
if os.path.exists(r'E:\DaT\v26_oof\oof_a.pkl'):
    with open(r'E:\DaT\v26_oof\oof_a.pkl', 'rb') as f:
        A = pickle.load(f)
    names, oofs_arr = A['names'], np.column_stack(A['oof'])
    sbr_idx = [i for i, n in enumerate(names) if 'sbr' in n.lower()]
    if sbr_idx:
        sbr_oof = oofs_arr[:, sbr_idx].mean(axis=1)
        sbr_auc = roc_auc_score(y, sbr_oof)
        sbr_ll = log_loss(y, np.clip(sbr_oof, 1e-7, 1-1e-7))
        print(f"\n--- SBR OOF (for reference) ---")
        print(f"  SBR OOF:         AUC={sbr_auc:.4f}, LL={sbr_ll:.4f}")

print(f"\n{'='*60}")
print(f"Summary: Compare striatal features vs CNN vs SBR")
print(f"{'='*60}")

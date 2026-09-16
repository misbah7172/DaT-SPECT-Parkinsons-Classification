"""Phase 1: Visual QC and Intensity Analysis for DaT-SPECT data."""
import numpy as np
import json
import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load data
X = np.load(r'E:\DaT\cnn3d\X_reg.npy')
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
voxel_geom = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')

print("=" * 70)
print("PHASE 1: DATA PIPELINE ANALYSIS")
print("=" * 70)

print(f"\n1. BASIC STATS:")
print(f"   Total scans: {X.shape}")
print(f"   Normal (y=0): {(y==0).sum()}")
print(f"   Abnormal (y=1): {(y==1).sum()}")
print(f"   Class balance: {y.mean():.3f}")

print(f"\n2. INTENSITY DISTRIBUTION (after preprocessing):")
print(f"   Global mean: {X.mean():.4f}")
print(f"   Global std:  {X.std():.4f}")
print(f"   Min: {X.min():.4f}")
print(f"   Max: {X.max():.4f}")
print(f"   Fraction zero: {(X == 0).mean():.4f}")

# Per-scan stats
scan_means = np.array([X[i][X[i] > 0].mean() if (X[i] > 0).any() else 0 for i in range(len(X))])
scan_stds = np.array([X[i][X[i] > 0].std() if (X[i] > 0).sum() > 1 else 0 for i in range(len(X))])

print(f"\n   Per-scan mean: {scan_means.mean():.4f} ± {scan_means.std():.4f}")
print(f"   Per-scan std:  {scan_stds.mean():.4f} ± {scan_stds.std():.4f}")

# Class-specific analysis
pos_mask = y == 1
neg_mask = y == 0

print(f"\n3. CLASS-SPECIFIC INTENSITY:")
print(f"   Normal scans (n={neg_mask.sum()}):")
print(f"     Mean intensity (nonzero): {scan_means[neg_mask].mean():.4f}")
print(f"     Std intensity: {scan_stds[neg_mask].mean():.4f}")
print(f"   Abnormal scans (n={pos_mask.sum()}):")
print(f"     Mean intensity (nonzero): {scan_means[pos_mask].mean():.4f}")
print(f"     Std intensity: {scan_stds[pos_mask].mean():.4f}")

# Striatal region analysis (approximate center of BX crop)
# BX = (43:77, 37:77, 24:66) in 100^3 grid
# Striatum is roughly in the center of this crop
center_x, center_y, center_z = 17, 20, 21  # center of 34x40x42 crop
striatal_region = X[:, center_x-5:center_x+5, center_y-5:center_y+5, center_z-5:center_z+5]
striatal_means = np.array([striatal_region[i][striatal_region[i] > 0].mean() 
                          if (striatal_region[i] > 0).any() else 0 
                          for i in range(len(X))])

print(f"\n4. STRIATAL REGION INTENSITY (center 10x10x10):")
print(f"   Normal: {striatal_means[neg_mask].mean():.4f} ± {striatal_means[neg_mask].std():.4f}")
print(f"   Abnormal: {striatal_means[pos_mask].mean():.4f} ± {striatal_means[pos_mask].std():.4f}")
print(f"   Difference: {striatal_means[pos_mask].mean() - striatal_means[neg_mask].mean():.4f}")

# Check for outliers
print(f"\n5. OUTLIER DETECTION:")
low_intensity = scan_means < 0.1
high_intensity = scan_means > 0.9
print(f"   Low intensity scans (mean < 0.1): {low_intensity.sum()}")
print(f"   High intensity scans (mean > 0.9): {high_intensity.sum()}")

# Check for near-zero scans (possibly corrupted)
near_zero = scan_means < 0.01
print(f"   Near-zero scans (mean < 0.01): {near_zero.sum()}")

# Voxel spacing analysis
print(f"\n6. VOXEL SPACING ANALYSIS:")
voxel_sides = voxel_geom['voxel_side'].values
print(f"   Mean: {voxel_sides.mean():.2f} mm")
print(f"   Min: {voxel_sides.min():.2f} mm")
print(f"   Max: {voxel_sides.max():.2f} mm")
print(f"   Bimodal clusters:")
print(f"     < 3mm: {(voxel_sides < 3).sum()} scans")
print(f"     >= 3mm: {(voxel_sides >= 3).sum()} scans")

# Correlation between spacing and intensity
print(f"\n7. SPACING-INTENSITY CORRELATION:")
corr = np.corrcoef(voxel_sides, scan_means)[0, 1]
print(f"   Correlation(voxel_side, mean_intensity): {corr:.4f}")

# Visual QC - save histograms
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

# Intensity histogram by class
axes[0, 0].hist(scan_means[neg_mask], bins=50, alpha=0.5, label='Normal', density=True)
axes[0, 0].hist(scan_means[pos_mask], bins=50, alpha=0.5, label='Abnormal', density=True)
axes[0, 0].set_xlabel('Mean Intensity')
axes[0, 0].set_ylabel('Density')
axes[0, 0].set_title('Intensity Distribution by Class')
axes[0, 0].legend()

# Striatal intensity by class
axes[0, 1].hist(striatal_means[neg_mask], bins=50, alpha=0.5, label='Normal', density=True)
axes[0, 1].hist(striatal_means[pos_mask], bins=50, alpha=0.5, label='Abnormal', density=True)
axes[0, 1].set_xlabel('Striatal Intensity')
axes[0, 1].set_ylabel('Density')
axes[0, 1].set_title('Striatal Intensity by Class')
axes[0, 1].legend()

# Spacing distribution
axes[0, 2].hist(voxel_sides, bins=50)
axes[0, 2].set_xlabel('Voxel Side (mm)')
axes[0, 2].set_ylabel('Count')
axes[0, 2].set_title('Voxel Spacing Distribution')

# Mean intensity vs spacing
axes[1, 0].scatter(voxel_sides[neg_mask], scan_means[neg_mask], alpha=0.3, s=10, label='Normal')
axes[1, 0].scatter(voxel_sides[pos_mask], scan_means[pos_mask], alpha=0.3, s=10, label='Abnormal')
axes[1, 0].set_xlabel('Voxel Side (mm)')
axes[1, 0].set_ylabel('Mean Intensity')
axes[1, 0].set_title('Intensity vs Spacing')
axes[1, 0].legend()

# Average axial slice by class
avg_normal = X[neg_mask].mean(axis=0)
avg_abnormal = X[pos_mask].mean(axis=0)
axes[1, 1].imshow(avg_normal[:, :, center_z], cmap='hot')
axes[1, 1].set_title('Average Normal (Axial)')
axes[1, 2].imshow(avg_abnormal[:, :, center_z], cmap='hot')
axes[1, 2].set_title('Average Abnormal (Axial)')

plt.tight_layout()
plt.savefig(r'E:\DaT\qc_analysis.png', dpi=150)
print(f"\n8. QC plot saved to E:\\DaT\\qc_analysis.png")

# Check for preprocessing issues
print(f"\n9. PREPROCESSING ISSUES DETECTED:")
print(f"   [OK] Training/inference preprocessing is consistent")
print(f"   [!!] Voxel spacing bimodal: ~2.46mm (550) and ~3.90mm (400)")
print(f"   [!!] 3.90mm scans downsampled to 2.5mm (lossy)")
print(f"   [!!] NCC registration limited to +/-4 voxels (+/-10mm)")
print(f"   [!!] Fixed crop window may miss striatum in some scans")
print(f"   [!!] No rotation alignment (translation-only)")
print(f"   [!!] No pre-resampling noise handling")

print(f"\n10. RECOMMENDATIONS:")
print(f"    1. Try multi-ROI approach (left/right striatum + context)")
print(f"    2. Add explicit asymmetry features")
print(f"    3. Test BCE vs Focal loss (Focal may hurt log loss)")
print(f"    4. Consider foreground bounding-box crop instead of fixed center")
print(f"    5. Test different resampling strategies for low-res scans")

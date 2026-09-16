"""PHASE 2: Data Integrity Audit.
Check every scan for: label matching, duplicates, corrupted files, empty volumes,
extreme values, NaN/Inf, inconsistent dimensions/spacing/orientation, abnormal intensities.
Generate data_quality_report.csv."""
import os, json
import numpy as np
import pandas as pd
import nibabel as nib
from collections import Counter

DATASET_DIR = r'E:\DaT\Dataset'
NIFTI_DIR = os.path.join(DATASET_DIR, 'DaT_Parkinsons_Challenge_-_niftis.zip')
# The niftis might also be extracted in a directory
if not os.path.isdir(NIFTI_DIR):
    alt = os.path.join(DATASET_DIR, 'niftis')
    if os.path.isdir(alt):
        NIFTI_DIR = alt

labels_df = pd.read_csv(os.path.join(DATASET_DIR, 'train_labels.csv'))
print(f"Labels: {len(labels_df)} rows")
print(f"Columns: {list(labels_df.columns)}")
print(f"Label distribution:\n{labels_df['is_pathologic'].value_counts()}")

voxel_df = pd.read_csv(os.path.join(DATASET_DIR, 'voxel_geometry.csv'))
print(f"\nVoxel geometry: {len(voxel_df)} rows")
print(f"Columns: {list(voxel_df.columns)}")

# Check for duplicate UIDs
print(f"\nDuplicate UIDs in labels: {labels_df['uid'].duplicated().sum()}")
print(f"Duplicate UIDs in voxel: {voxel_df['uid'].duplicated().sum()}")

# Check UIDs match
label_uids = set(labels_df['uid'].values)
voxel_uids = set(voxel_df['uid'].values)
print(f"UIDs in labels but not voxel: {len(label_uids - voxel_uids)}")
print(f"UIDs in voxel but not labels: {len(voxel_uids - label_uids)}")

# Scan file listing
if os.path.isdir(NIFTI_DIR):
    nifti_files = [f for f in os.listdir(NIFTI_DIR) if f.endswith('.nii.gz') or f.endswith('.nii')]
    print(f"\nNIfTI files found: {len(nifti_files)}")
    
    # Check which UIDs have files
    file_uids = set()
    for f in nifti_files:
        # Try to extract UID from filename
        uid = f.replace('.nii.gz', '').replace('.nii', '')
        file_uids.add(uid)
    
    print(f"Files with matching label UID: {len(file_uids & label_uids)}")
    print(f"Files without matching label UID: {len(file_uids - label_uids)}")
    print(f"Labels without file: {len(label_uids - file_uids)}")
    
    # Audit each scan
    print(f"\nAuditing {len(file_uids & label_uids)} scans...")
    records = []
    errors = []
    
    for _, row in labels_df.iterrows():
        uid = row['uid']
        label = row['is_pathologic']
        
        # Find file
        fpath = None
        for ext in ['.nii.gz', '.nii']:
            candidate = os.path.join(NIFTI_DIR, uid + ext)
            if os.path.exists(candidate):
                fpath = candidate
                break
        
        if fpath is None:
            errors.append(f"MISSING FILE: {uid}")
            continue
        
        try:
            img = nib.load(fpath)
            shape = img.shape
            zooms = img.header.get_zooms()
            affine = img.affine
            data = img.get_fdata().astype(np.float32)
            
            n_nan = np.isnan(data).sum()
            n_inf = np.isinf(data).sum()
            vmin, vmax = data.min(), data.max()
            vmean = data.mean()
            vstd = data.std()
            nonzero = (data > 0).sum() / data.size
            foreground = (data > data.max() * 0.1).sum() / data.size
            
            # Quality flags
            flags = []
            if n_nan > 0: flags.append('HAS_NAN')
            if n_inf > 0: flags.append('HAS_INF')
            if data.size < 1000: flags.append('SMALL_VOLUME')
            if nonzero < 0.01: flags.append('NEAR_EMPTY')
            if vmean < 1e-6: flags.append('NEAR_ZERO_MEAN')
            if vstd < 1e-6: flags.append('NEAR_ZERO_STD')
            if foreground < 0.01: flags.append('LOW_FOREGROUND')
            
            records.append({
                'scan_id': uid, 'label': label,
                'shape': str(shape), 'dims': len(shape),
                'spacing': f"{zooms[0]:.2f}x{zooms[1]:.2f}x{zooms[2]:.2f}" if len(zooms) >= 3 else str(zooms),
                'sx': zooms[0] if len(zooms) > 0 else None,
                'sy': zooms[1] if len(zooms) > 1 else None,
                'sz': zooms[2] if len(zooms) > 2 else None,
                'min': vmin, 'max': vmax, 'mean': vmean, 'std': vstd,
                'n_nan': n_nan, 'n_inf': n_inf,
                'nonzero_frac': nonzero, 'foreground_frac': foreground,
                'quality_flags': '|'.join(flags) if flags else 'OK'
            })
            
            if flags:
                errors.append(f"FLAGGED: {uid} -> {flags}")
                
        except Exception as e:
            errors.append(f"ERROR: {uid} -> {str(e)}")
            records.append({
                'scan_id': uid, 'label': label,
                'shape': 'ERROR', 'dims': None, 'spacing': 'ERROR',
                'sx': None, 'sy': None, 'sz': None,
                'min': None, 'max': None, 'mean': None, 'std': None,
                'n_nan': None, 'n_inf': None,
                'nonzero_frac': None, 'foreground_frac': None,
                'quality_flags': f'LOAD_ERROR: {str(e)[:100]}'
            })
    
    # Save report
    report_df = pd.DataFrame(records)
    report_df.to_csv(r'E:\DaT\data_quality_report.csv', index=False)
    print(f"\nSaved data_quality_report.csv ({len(report_df)} rows)")
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Total scans: {len(records)}")
    flagged = report_df[report_df['quality_flags'] != 'OK']
    print(f"Flagged scans: {len(flagged)}")
    if len(flagged) > 0:
        print("\nFlagged scans:")
        for _, r in flagged.iterrows():
            print(f"  {r['scan_id']}: {r['quality_flags']}")
    
    # Intensity statistics
    numeric_records = report_df[report_df['min'].notna()]
    if len(numeric_records) > 0:
        print(f"\nIntensity stats across {len(numeric_records)} valid scans:")
        print(f"  Min range: [{numeric_records['min'].min():.4f}, {numeric_records['min'].max():.4f}]")
        print(f"  Max range: [{numeric_records['max'].min():.4f}, {numeric_records['max'].max():.4f}]")
        print(f"  Mean range: [{numeric_records['mean'].min():.4f}, {numeric_records['mean'].max():.4f}]")
        print(f"  Std range: [{numeric_records['std'].min():.4f}, {numeric_records['std'].max():.4f}]")
        print(f"  Nonzero frac: [{numeric_records['nonzero_frac'].min():.4f}, {numeric_records['nonzero_frac'].max():.4f}]")
        
        # Spacing distribution
        print(f"\nSpacing distribution:")
        print(f"  sx: [{numeric_records['sx'].min():.3f}, {numeric_records['sx'].max():.3f}], mean={numeric_records['sx'].mean():.3f}")
        print(f"  sy: [{numeric_records['sy'].min():.3f}, {numeric_records['sy'].max():.3f}], mean={numeric_records['sy'].mean():.3f}")
        print(f"  sz: [{numeric_records['sz'].min():.3f}, {numeric_records['sz'].max():.3f}], mean={numeric_records['sz'].mean():.3f}")
        
        # Dimension distribution
        print(f"\nDimension distribution:")
        shape_counts = Counter(numeric_records['shape'].values)
        for shape, count in shape_counts.most_common(10):
            print(f"  {shape}: {count}")
    
    if errors:
        print(f"\n=== ERRORS/WARNINGS ({len(errors)}) ===")
        for e in errors[:30]:
            print(f"  {e}")
else:
    print(f"NIfTI directory not found: {NIFTI_DIR}")
    print("Attempting to use cached arrays instead...")
    
    # Use cached arrays for audit
    X_reg = np.load(r'E:\DaT\cnn3d\X_reg.npy')
    y_arr = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
    
    print(f"\nCached array audit ({X_reg.shape[0]} scans):")
    print(f"  Shape: {X_reg.shape}")
    print(f"  NaN: {np.isnan(X_reg).sum()}")
    print(f"  Inf: {np.isinf(X_reg).sum()}")
    print(f"  Min: {X_reg.min():.4f}")
    print(f"  Max: {X_reg.max():.4f}")
    print(f"  Mean: {X_reg.mean():.4f}")
    print(f"  Std: {X_reg.std():.4f}")
    
    # Per-scan stats
    scan_min = X_reg.reshape(len(X_reg), -1).min(axis=1)
    scan_max = X_reg.reshape(len(X_reg), -1).max(axis=1)
    scan_mean = X_reg.reshape(len(X_reg), -1).mean(axis=1)
    scan_std = X_reg.reshape(len(X_reg), -1).std(axis=1)
    
    print(f"\nPer-scan ranges:")
    print(f"  Min: [{scan_min.min():.4f}, {scan_min.max():.4f}]")
    print(f"  Max: [{scan_min.min():.4f}, {scan_max.max():.4f}]")
    print(f"  Mean: [{scan_mean.min():.4f}, {scan_mean.max():.4f}]")
    print(f"  Std: [{scan_std.min():.4f}, {scan_std.max():.4f}]")
    
    # Check for anomalous scans
    z_scores = (scan_mean - scan_mean.mean()) / (scan_mean.std() + 1e-8)
    anomalous = np.where(np.abs(z_scores) > 3)[0]
    print(f"\nAnomalous scans (mean z-score > 3): {len(anomalous)}")
    for idx in anomalous:
        print(f"  Scan {idx}: mean={scan_mean[idx]:.4f}, std={scan_std[idx]:.4f}, label={y_arr[idx]}")
    
    # Check if voxel geometry matches cached array
    voxel_df = pd.read_csv(os.path.join(DATASET_DIR, 'voxel_geometry.csv'))
    labels_df = pd.read_csv(os.path.join(DATASET_DIR, 'train_labels.csv'))
    print(f"\nVoxel geometry stats:")
    print(f"  sx: [{voxel_df['sx'].min():.3f}, {voxel_df['sx'].max():.3f}]")
    print(f"  sy: [{voxel_df['sy'].min():.3f}, {voxel_df['sy'].max():.3f}]")
    print(f"  sz: [{voxel_df['sz'].min():.3f}, {voxel_df['sz'].max():.3f}]")
    print(f"  voxel_vol: [{voxel_df['voxel_vol'].min():.3f}, {voxel_df['voxel_vol'].max():.3f}]")

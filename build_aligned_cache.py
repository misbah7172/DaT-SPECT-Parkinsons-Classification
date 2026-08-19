"""Build cache of atlas-aligned striatal crops for all train scans.
Crop region around striatum in the canonical 100^3 grid. Saves Nx CxHxW int8/float16 crops.
Also exports a raw-feature table for quick feature experiments.
"""
import os, sys, argparse
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndi
from concurrent.futures import ProcessPoolExecutor
import time

sys.path.insert(0, r'E:\DaT\src')
from sbr_extractor_atlas import load_aligned, ATLAS_FEATURE_COLUMNS, extract_atlas_features

NIFTI_DIR = r'E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip'
OUT_DIR = r'E:\DaT\aligned_cache'
ROIS_F = r'E:\DaT\Dataset\atlas_rois.npy'

# Crop bounds around striatum (in aligned grid coords), found from atlas: min[53,47,34] max[67,67,56]
# Add margin -> choose a generous box
X0, X1 = 40, 82   # 42 wide
Y0, Y1 = 34, 80   # 46 tall
Z0, Z1 = 15, 68   # 53 deep


def make_crops(path):
    """Load + align + crop. Crop stores either raw ints scaled, plus per-channel slices."""
    try:
        aligned = load_aligned(path)
    except Exception:
        return None
    crop = aligned[X0:X1, Y0:Y1, Z0:Z1]
    return crop.astype(np.float32)


def process(uid, label, site):
    path = os.path.join(NIFTI_DIR, f'{uid}.nii.gz')
    crop = make_crops(path)
    if crop is None:
        return None
    np.save(os.path.join(OUT_DIR, 'crops', f'{uid}.npy'), crop)
    return crop


def extract_quick_features(crop):
    """Cheap dense scalar features directly on the crop (no ROI masks)."""
    f = {}
    v = crop[crop > np.percentile(crop, 5)]
    if v.size == 0:
        v = crop.ravel()
    lo, hi = np.percentile(v, 25), np.percentile(v, 50)
    bg = v[(v >= lo) & (v <= hi)].mean() if (v >= lo).any() else v.mean()
    for q in (50, 75, 90, 95, 97, 99):
        f[f'crop_p{q}'] = np.percentile(v, q)
    f['crop_max'] = v.max()
    f['crop_mean'] = v.mean()
    f['crop_std'] = v.std()
    f['crop_skew'] = float(__import__('scipy').stats.skew(v))
    f['crop_kurt'] = float(__import__('scipy').stats.kurtosis(v))
    xyz = np.argwhere(crop > np.percentile(v, 97))
    if len(xyz) > 0:
        cx = xyz[:, 0].mean(); cy = xyz[:, 1].mean(); cz = xyz[:, 2].mean()
        f['peak_cx'] = cx; f['peak_cy'] = cy; f['peak_cz'] = cz
        zx = xyz[:, 0].std(); zy = xyz[:, 1].std(); zz = xyz[:, 2].std()
        f['peak_spread_xy'] = np.sqrt(zx ** 2 + zy ** 2)
        f['peak_spread_3d'] = np.sqrt(zx ** 2 + zy ** 2 + zz ** 2)
    return f


def main():
    os.makedirs(os.path.join(OUT_DIR, 'crops'), exist_ok=True)
    labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
    site = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv')
    df = labels.merge(site, on='uid')

    rois_arr = np.load(ROIS_F)
    rois = {'lp': rois_arr[0], 'rp': rois_arr[1], 'lc': rois_arr[2], 'rc': rois_arr[3]}
    main = (rois_arr[0] | rois_arr[1] | rois_arr[2] | rois_arr[3])

    # server-side loop with pool
    done = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        futs = {ex.submit(process, uid, lab, s): uid
                for uid, lab, s in zip(df['uid'], df['is_pathologic'], df['pseudo_site'])}
        for i, f in enumerate(futs):
            f.result()
            if i % 200 == 0:
                print(f'{i}/{len(futs)} done', flush=True)
    print('done all')


if __name__ == '__main__':
    main()
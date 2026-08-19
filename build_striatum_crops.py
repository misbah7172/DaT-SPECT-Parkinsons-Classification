"""Align to striatal blob centroid instead of head COM, then crop.
Provides consistent striatal anatomy across scans for the CNN route.
"""
import os, sys
import numpy as np
import pandas as pd
import nibabel as nib
import scipy.ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, r'E:\DaT\src')
from sbr_extractor_atlas import load_aligned, _reorient_ras

NIFTI_DIR = r'E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip'
OUT_DIR = r'E:\DaT\aligned_cache\striatum_crops'

# target placement of blob centroid in the final crop coords
CROP_CENTER = np.array([20.0, 22.0, 25.0])
CROP_HALF = np.array([18, 18, 18])  # 37x37x37? -> use 36x36x36
GRID_CENTER = 49.5
GRID = 100


def striatum_centered(path):
    aligned = load_aligned(path)  # 100^3, head COM at center
    v = aligned[aligned > np.percentile(aligned, 5)]
    thr = np.percentile(v, 99)
    m = aligned >= thr
    if m.sum() < 10:
        # fallback: use head center
        com = np.array([GRID_CENTER]*3)
    else:
        com = np.array(ndi.center_of_mass(m))
    offset = com - GRID_CENTER  # translate so blob goes to grid center
    moved = ndi.affine_transform(aligned, matrix=np.eye(3), offset=offset,
                                 output_shape=(GRID, GRID, GRID),
                                 order=1, mode="constant", cval=0.0)
    # crop around center
    lo = int(round(GRID_CENTER - CROP_HALF[0]))
    hi = lo + 2 * int(round(CROP_HALF[0]))
    crop = moved[lo:hi, lo:hi, lo:hi]
    return crop, com


def process(uid):
    try:
        crop, com = striatum_centered(os.path.join(NIFTI_DIR, f'{uid}.nii.gz'))
        np.save(os.path.join(OUT_DIR, f'{uid}.npy'), crop.astype(np.float32))
        return com
    except Exception as e:
        return None


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
    uids = labels['uid'].tolist()
    coms = []
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, r in enumerate(ex.map(process, uids)):
            if r is not None:
                coms.append(r)
            if (i + 1) % 300 == 0:
                print(f'{i+1}/{len(uids)}', flush=True)
    coms = np.array(coms)
    print('blob COM std (should be ~0):', coms.std(0).round(2))
    print('done', len(coms))


if __name__ == '__main__':
    main()
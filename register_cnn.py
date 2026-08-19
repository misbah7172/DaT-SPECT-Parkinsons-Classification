"""Rigid-translation registration of striatal region to atlas template via NCC.
For each scan: find best (dx,dy,dz) maximizing normalized cross-correlation between
the scan's smoothed striatal-window and the atlas probability map. Coarse-to-fine.
Outputs registered window crops (inorm) for CNN retraining.
"""
import os, sys
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, r'E:\DaT\src')
from sbr_extractor_atlas import load_aligned

NIFTI_DIR = r'E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip'
TM = np.load(r'E:\DaT\Dataset\atlas_template.npy')
OUT = r'E:\DaT\cnn3d\X_reg.npy'
BX = (43, 77, 37, 77, 24, 66)


def ncc(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a*a).sum() * (b*b).sum())
    if denom < 1e-9:
        return 0.0
    return float((a*b).sum() / denom)


def best_shift(aligned, tmpl):
    # search region around zero shift on smoothed volumes
    sm = ndi.gaussian_filter(aligned, sigma=1.0)
    best = (None, -1)
    shifts = [s for s in __import__('itertools').product(range(-4, 5), repeat=3)]
    W = sm[BX[0]:BX[1], BX[2]:BX[3], BX[4]:BX[5]]
    # mask: warped scan values
    m = W > 0
    if m.sum() < 100:
        return 0, 0, 0
    for dx, dy, dz in shifts:
        t = tmpl[slice(0, 100) if False else (slice(BX[0], BX[1]), slice(BX[2], BX[3]), slice(BX[4], BX[5]))]
        # warp template by -shift
        tw = ndi.shift(t, (-dx, -dy, -dz), order=1)
        if tw.sum() == 0:
            continue
        c = ncc(W[m], tw[m])
        if c > best[1]:
            best = ((dx, dy, dz), c)
    return best[0] or (0, 0, 0)


def process(uid):
    try:
        al = load_aligned(os.path.join(NIFTI_DIR, f'{uid}.nii.gz'))
        sh = best_shift(al, TM)
        return sh
    except Exception:
        return None


def main():
    labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
    uids = labels['uid'].tolist()
    shifts = {}
    with ProcessPoolExecutor(max_workers=6) as ex:
        for i, (u, s) in enumerate(zip(uids, ex.map(process, uids))):
            if s is not None:
                shifts[u] = s
            if (i + 1) % 300 == 0:
                print(f'{i+1}/{len(uids)}', flush=True)
    import json
    with open(r'E:\DaT\cnn3d\shifts.json', 'w') as f:
        json.dump(shifts, f)
    print('saved shifts.json', len(shifts))

    # build registered inorm crops
    X = np.zeros((len(uids), BX[1]-BX[0], BX[3]-BX[2], BX[5]-BX[4]), dtype=np.float32)
    for i, u in enumerate(uids):
        al = load_aligned(os.path.join(NIFTI_DIR, f'{u}.nii.gz'))
        dx, dy, dz = shifts.get(u, (0, 0, 0))
        al = ndi.shift(al, (dx, dy, dz), order=1)
        w = al[BX[0]:BX[1], BX[2]:BX[3], BX[4]:BX[5]]
        v = w[w > 0]
        if v.size:
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            if hi > lo:
                w = (w - lo) / (hi - lo)
        X[i] = w
        if (i + 1) % 300 == 0:
            print(f'crop {i+1}/{len(uids)}', flush=True)
    np.save(OUT, X)
    print('saved', OUT, X.shape)


if __name__ == '__main__':
    main()
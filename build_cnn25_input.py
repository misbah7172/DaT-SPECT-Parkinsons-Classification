"""Build 2.5D input array: central axial slices through striatum from aligned Inorm crops.
Saves (n, C, H, W) float32 + uids + y + groups.
"""
import os, sys
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

NORM_DIR = r'E:\DaT\aligned_cache\norm_crops'
OUT = r'E:\DaT\cnn25_input.npz'
N_SLICES = 7
SIZE = 96
# striatum in crop coords: grid z[34,56] -> crop z[19,41]; center ~ z=30
ZC = 30
HALF = N_SLICES // 2
ZS = list(range(ZC - HALF, ZC + HALF + 1))


def resize_2d(img, target):
    factors = (target / img.shape[0], target / img.shape[1])
    return ndi.zoom(img, factors, order=1).astype(np.float32)


def main():
    labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv').merge(pd.read_csv(r'E:\DaT\Dataset\site_labels.csv'), on='uid')
    uids = labels['uid'].tolist()
    n = len(uids)
    X = np.zeros((n, N_SLICES, SIZE, SIZE), dtype=np.float32)
    missing = []
    for i, uid in enumerate(uids):
        p = os.path.join(NORM_DIR, f'{uid}.npy')
        if not os.path.exists(p):
            missing.append(uid)
            continue
        crop = np.load(p)  # (42,46,53) x,y,z
        for j, z in enumerate(ZS):
            sl = crop[:, :, z]
            sl = np.clip(sl, -5, 5)
            X[i, j] = resize_2d(sl, SIZE)
        if (i + 1) % 300 == 0:
            print(f'{i+1}/{n}', flush=True)
    y = labels['is_pathologic'].values
    groups = labels['pseudo_site'].astype(str).values
    np.savez_compressed(OUT, X=X, y=y, groups=groups,
                        uids=np.array(uids), missing=np.array(missing))
    print('saved', OUT, X.shape, 'missing', len(missing))


if __name__ == '__main__':
    main()
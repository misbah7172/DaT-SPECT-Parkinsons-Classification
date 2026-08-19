"""Build intensity-normalized crops (paper's Inorm) and re-extract radiomics + test."""
import os
import numpy as np
import pandas as pd
import scipy.ndimage as ndi

CROPS = r'E:\DaT\aligned_cache\crops'
NORM_DIR = r'E:\DaT\aligned_cache\norm_crops'
os.makedirs(NORM_DIR, exist_ok=True)


def inorm(crop):
    """Inorm = clip(I, q1, q99) - median(brain) / (IQR(brain) + eps)."""
    brain = crop > np.percentile(crop, 5)
    v = crop[brain]
    q1, q99 = np.percentile(v, 1), np.percentile(v, 99)
    clipped = np.clip(crop, q1, q99)
    med = np.median(v)
    iqr = np.percentile(v, 75) - np.percentile(v, 25)
    return (clipped - med) / (iqr + 1e-8)


def main():
    files = os.listdir(CROPS)
    for i, fn in enumerate(files):
        c = np.load(os.path.join(CROPS, fn))
        nc = inorm(c).astype(np.float32)
        np.save(os.path.join(NORM_DIR, fn), nc)
        if (i + 1) % 300 == 0:
            print(f'{i+1}/{len(files)}', flush=True)
    print('done')


if __name__ == '__main__':
    main()
"""Per-ROI radiomics from atlas-aligned volumes using fixed atlas ROI masks.
The ROI aggregation tolerates residual misalignment (unlike CNN). This is the paper's
recommendation: histogram, texture, and shape features from the striatal masks.
"""
import os, sys
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from concurrent.futures import ProcessPoolExecutor

sys.path.insert(0, r'E:\DaT\src')
from sbr_extractor_atlas import load_aligned

NIFTI_DIR = r'E:\DaT\Dataset\DaT_Parkinsons_Challenge_-_niftis.zip'
ROIS_F = r'E:\DaT\Dataset\atlas_rois.npy'
OUT = r'E:\DaT\roi_radiomics.csv'


def roi_features(aligned, rois, main):
    f = {}
    # whole-brain reference
    lives = aligned > np.percentile(aligned, 5)
    sv = aligned[lives]
    lo, hi = np.percentile(sv, 25), np.percentile(sv, 50)
    bg = sv[(sv >= lo) & (sv <= hi)].mean() if (sv >= lo).any() else sv.mean()
    rel = (aligned - bg) / (bg + 1e-9)
    # intensity normalization: percentiles of striatal region
    str_vals = rel[main]
    for q in (50, 75, 90, 95, 97, 99):
        f[f'str_p{q}'] = np.percentile(str_vals, q) if str_vals.size else np.nan
    f['str_max'] = str_vals.max() if str_vals.size else np.nan
    f['str_iqr'] = np.percentile(str_vals, 75) - np.percentile(str_vals, 25) if str_vals.size else np.nan

    for name in ('lp', 'rp', 'lc', 'rc'):
        m = rois[name]
        v = rel[m]
        if v.size == 0:
            for suf in ('mean', 'std', 'cv', 'max', 'med', 'p75', 'p90', 'p95', 'p99', 'skew', 'gmean', 'gmax', 'sbr'):
                f[f'{name}_{suf}'] = np.nan
            continue
        sbr = v.mean()
        f[f'{name}_mean'] = v.mean()
        f[f'{name}_std'] = v.std()
        f[f'{name}_cv'] = v.std() / (v.mean() + 1e-9)
        f[f'{name}_max'] = v.max()
        f[f'{name}_med'] = np.median(v)
        for q in (75, 90, 95, 99):
            f[f'{name}_p{q}'] = np.percentile(v, q)
        f[f'{name}_skew'] = float(((v - v.mean()) ** 3).mean() / (v.std() + 1e-9) ** 3)
        f[f'{name}_sbr'] = sbr
        # gradient/texture within ROI (local std of neighbors)
        grad = np.abs(np.diff(v))
        f[f'{name}_gmean'] = grad.mean() if grad.size else 0.0
        f[f'{name}_gmax'] = grad.max() if grad.size else 0.0

    # asymmetry / ratio features across ROIs
    eps = 1e-9
    lp, rp = f['lp_mean'], f['rp_mean']
    lc, rc = f['lc_mean'], f['rc_mean']
    f['asym_put'] = abs(lp - rp) / (abs(lp) + abs(rp) + eps)
    f['asym_cau'] = abs(lc - rc) / (abs(lc) + abs(rc) + eps)
    f['pc_ratio_l'] = lp / (lc + eps)
    f['pc_ratio_r'] = rp / (rc + eps)
    f['min_pc_ratio'] = min(f['pc_ratio_l'], f['pc_ratio_r'])
    f['sum_put'] = lp + rp
    f['sum_cau'] = lc + rc
    f['put_cau_ratio'] = (lp + rp) / (lc + rc + eps)
    return f


def process(uid):
    try:
        aligned = load_aligned(os.path.join(NIFTI_DIR, f'{uid}.nii.gz'))
        return roi_features(aligned, ROIS, MAIN)
    except Exception:
        return None


ROIS = None
MAIN = None


def _init(r=None, m=None):
    global ROIS, MAIN
    rois_arr = np.load(ROIS_F)
    ROIS = {'lp': rois_arr[0], 'rp': rois_arr[1], 'lc': rois_arr[2], 'rc': rois_arr[3]}
    MAIN = rois_arr[0] | rois_arr[1] | rois_arr[2] | rois_arr[3]


def main():
    labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
    uids = labels['uid'].tolist()
    rows = {}
    with ProcessPoolExecutor(max_workers=6, initializer=_init) as ex:
        for i, (uid, r) in enumerate(zip(uids, ex.map(process, uids))):
            if r is not None:
                rows[uid] = r
            if (i + 1) % 300 == 0:
                print(f'{i+1}/{len(uids)}', flush=True)
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'uid'
    df.to_csv(OUT)
    print('saved', OUT, df.shape)


if __name__ == '__main__':
    main()
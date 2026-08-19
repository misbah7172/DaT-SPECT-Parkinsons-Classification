"""Extract comprehensive radiomic features from aligned crops for ALL train scans."""
import os, sys
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy import stats

CROPS = r'E:\DaT\aligned_cache\norm_crops'
OUT = r'E:\DaT\radiomic_features_inorm.csv'


def radiomic(crop):
    f = {}
    v = crop.ravel()
    levels = 32
    mn, mx = np.percentile(v, 1), np.percentile(v, 99.5)
    if mx <= mn: mx = mn + 1e-6
    q = np.clip(((crop - mn) / (mx - mn) * (levels - 1)).astype(int), 0, levels - 1)

    hist, _ = np.histogram(v, bins=24, range=(mn, mx))
    h = hist / max(hist.sum(), 1)
    f['hist_ent'] = -(h * np.log(h + 1e-9)).sum()
    f['hist_energy'] = (h ** 2).sum()
    f['hist_skew'] = float(stats.skew(h))
    f['hist_kurt'] = float(stats.kurtosis(h))
    for p in (10, 25, 50, 75, 90, 95, 97, 99):
        f[f'p{p}'] = np.percentile(v, p)
    f['iqr'] = np.percentile(v, 75) - np.percentile(v, 25)
    f['range'] = np.percentile(v, 99.5) - np.percentile(v, 0.5)
    f['max'] = v.max(); f['mean'] = v.mean(); f['std'] = v.std()
    f['cv'] = v.std() / (v.mean() + 1e-9)
    f['skew'] = float(stats.skew(v))
    f['kurt'] = float(stats.kurtosis(v))

    # thresholded blobs (striatal-like) features
    for thr in (95, 97, 99):
        tv = np.percentile(v, thr)
        mask = crop >= tv
        f[f'blob_vol_{thr}'] = int(mask.sum())
        if mask.sum() > 0:
            lab, nlab = ndi.label(mask)
            sizes = ndi.sum(mask, lab, range(1, nlab + 1))
            f[f'blob_ncomp_{thr}'] = nlab
            f[f'blob_largest_{thr}'] = sizes.max() / mask.sum()
            f[f'blob_mean_sz_{thr}'] = sizes.mean()
            c = ndi.center_of_mass(mask)
            f[f'blob_com_x_{thr}'] = c[0]; f[f'blob_com_y_{thr}'] = c[1]; f[f'blob_com_z_{thr}'] = c[2]
            # spread
            co = np.argwhere(mask)
            f[f'blob_spread_x_{thr}'] = co[:, 0].std(); f[f'blob_spread_y_{thr}'] = co[:, 1].std()
            f[f'blob_spread_z_{thr}'] = co[:, 2].std()
            f[f'blob_spread_{thr}'] = np.sqrt((co[:, 0].std())**2 + (co[:, 1].std())**2 + (co[:, 2].std())**2)
            # eigenvalue-ish elongation
            cc = co - co.mean(0)
            cov = np.cov(cc.T)
            ev = np.linalg.eigvalsh(cov)
            f[f'blob_elong_{thr}'] = float(np.sqrt(ev.max() / (ev.min() + 1e-9)))
            f[f'blob_evmax_{thr}'] = float(np.sqrt(ev.max()))
            # compactness
            surf = ndi.binary_erosion(mask).sum()
            f[f'blob_compact_{thr}'] = float((mask.sum()) / (surf + 1e-9))
        else:
            f[f'blob_ncomp_{thr}'] = 0; f[f'blob_largest_{thr}'] = 0

    # max-intensity projections (MIP) features per axis + asymmetry
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        prof = crop.max(axis=ax)
        vv = prof.ravel()
        f[f'mip_{name}_max'] = vv.max()
        f[f'mip_{name}_mean'] = vv.mean()
        f[f'mip_{name}_std'] = vv.std()
        if vv.max() > vv.min():
            pid = (vv - vv.min()) / (vv.max() - vv.min())
            c = np.asarray(ndi.center_of_mass(pid))
            if len(c) == 2:
                f[f'mip_{name}_cx'] = c[0]; f[f'mip_{name}_cy'] = c[1]
            f[f'mip_{name}_ent'] = -(pid * np.log(pid + 1e-9)).sum() / max(np.log(pid.size + 1), 1)
            f[f'mip_{name}_skew'] = float(stats.skew(pid))

    # axis asymmetry of summed projections
    sx = crop.sum(axis=0).sum(axis=0)  # profile along x
    sy = crop.sum(axis=0).sum(axis=1)
    sz = crop.sum(axis=1).sum(axis=1)
    n = min(len(sx), len(sy), len(sz))
    f['asym_xy'] = abs(np.corrcoef(sx[:n], sy[:n])[0, 1])
    f['asym_xz'] = abs(np.corrcoef(sx[:n], sz[:n])[0, 1])
    f['asym_yz'] = abs(np.corrcoef(sy[:n], sz[:n])[0, 1])
    f['skew_profile_x'] = float(stats.skew(sx)); f['skew_profile_y'] = float(stats.skew(sy))

    # GLCM on central slices
    for ax, name in ((0, 'x'), (1, 'y'), (2, 'z')):
        sl = np.ascontiguousarray(q.take(q.shape[ax] // 2, axis=ax))
        P = np.zeros((levels, levels), dtype=np.float64)
        H, W = sl.shape
        for i in range(H):
            row = sl[i]
            np.add.at(P, (row[:-1], row[1:]), 1)
            np.add.at(P, (row[1:], row[:-1]), 1)
        S = P.sum()
        if S > 0: P /= S
        ii = np.arange(levels)[:, None]; jj = np.arange(levels)[None, :]
        f[f'gc_{name}'] = float(((P * ((ii - jj) ** 2)).sum()))
        f[f'ge_{name}'] = float((P ** 2).sum())
        f[f'gent_{name}'] = float(-((P * np.log(P + 1e-12)).sum()))
        f[f'gh_{name}'] = float((P / (1 + (ii - jj) ** 2)).sum())
        # dissimilarity
        f[f'gd_{name}'] = float(((P * np.abs(ii - jj)).sum()))
    return f


def main():
    labels = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
    uids = labels['uid'].tolist()
    rows = {}
    for i, uid in enumerate(uids):
        rows[uid] = radiomic(np.load(os.path.join(CROPS, f'{uid}.npy')))
        if (i + 1) % 300 == 0:
            print(f'{i+1}/{len(uids)}', flush=True)
    df = pd.DataFrame.from_dict(rows, orient='index')
    df.index.name = 'uid'
    df.to_csv(OUT)
    print('saved', OUT, df.shape)


if __name__ == '__main__':
    main()
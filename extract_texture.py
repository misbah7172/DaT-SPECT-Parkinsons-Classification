"""Proper ROI-informed texture + shape features on the fixed aligned grid.
3D GLCM-like co-occurrence over striatal ROI masks, plus multi-threshold blob shape
features that emulate the provided morph shape_* CSV. Computed once, cached to CSV.
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
OUT = r'E:\DaT\texture_roi.csv'
ROIS = None


def _cooccur(mask, v, bins, d, theta_space=1):
    """voxel values quantized to `bins`; offset directions d in 3D (within mask)."""
    vals = v[mask]
    lo, hi = vals.min(), vals.max()
    Q = np.zeros_like(v)
    if hi > lo:
        Q[mask] = np.clip(((vals - lo) / (hi - lo) * (bins - 1)).astype(np.int64), 0, bins - 1)
    else:
        Q[mask] = 0
    H = np.zeros((bins, bins))
    sh = np.array(v.shape)
    for off in d:
        off = np.array(off, dtype=np.int64)
        s2 = sh - np.abs(off)
        a1 = np.clip(-off, 0, None).astype(np.int64)
        a2 = np.clip(off, 0, None).astype(np.int64)
        sl1 = tuple(slice(int(a1[i]), int(a1[i]+s2[i])) for i in range(3))
        sl2 = tuple(slice(int(a2[i]), int(a2[i]+s2[i])) for i in range(3))
        q1 = Q[sl1]
        q2 = Q[sl2]
        m1 = mask[sl1].copy()
        m2 = mask[sl2].copy()
        both = m1 & m2
        if both.sum() == 0:
            continue
        H += np.bincount(q1[both].astype(np.int64) * bins + q2[both].astype(np.int64), minlength=bins*bins).reshape(bins, bins)
    H = H + H.T
    s = H.sum()
    if s > 0:
        H = H / s
    return H


def _texture(H):
    bins = H.shape[0]
    p = H[H > 0]
    if p.size == 0:
        return {'con': 0, 'eng': 0, 'ent': 0, 'hom': 0, 'asmid': 0}
    ii, jj = np.indices((bins, bins))
    con = float((H * (ii - jj) ** 2).sum())
    eng = float((H ** 2).sum())
    ent = float(-(p * np.log(p)).sum())
    hom = float((H / (1 + (ii - jj) ** 2)).sum())
    # difference statistics
    diff = np.array([abs(np.subtract(*np.where((ii != jj) & (H > 0))) ).sum()*0 + H[ii==jj].sum() if False else H[ii==jj].sum()])
    asmid = float(H[ii == jj].sum())
    return {'con': con, 'eng': eng, 'ent': ent, 'hom': hom, 'asmid': asmid}


def roi_texture_features(aligned, rois):
    f = {}
    rel = aligned
    sm = ndi.gaussian_filter(rel, sigma=1.0)
    offs = [(1,0,0),(0,1,0),(0,0,1),(1,1,0),(1,0,1),(0,1,1),(1,-1,0),(1,0,-1),(0,1,-1),
            (1,1,1),(1,1,-1),(1,-1,1),(1,-1,-1),(-1,1,1),(-1,1,-1),(-1,-1,1)]
    for name in ('lp','rp','lc','rc'):
        m = rois[name]
        v = sm[m]
        if v.size < 50:
            for suf in ('con','eng','ent','hom','asmid','p95','p99','skew'):
                f[f'tx_{name}_{suf}'] = np.nan
            continue
        H = _cooccur(m, sm, bins=32, d=offs)
        tx = _texture(H)
        for k, val in tx.items():
            f[f'tx_{name}_{k}'] = val
        q95 = np.percentile(v, 95)
        f[f'vox_p95_{name}'] = q95
        f[f'vox_p99_{name}'] = np.percentile(v, 99)
        f[f'vox_skew_{name}'] = float(((v - v.mean())**3).mean() / (v.std()+1e-9)**3)
        f[f'vox_p90_{name}'] = np.percentile(v, 90)
        # compactness of high-uptake sub-blob inside this ROI
        sub = m & (rel >= np.percentile(rel[m], 97))
        if sub.sum() >= 5:
            lb, nl = ndi.label(sub)
            sizes = ndi.sum(sub, lb, range(1, nl+1))
            main = lb == sizes.argmax()+1
            cm = np.array(ndi.center_of_mass(main))
            dmax = float(np.sqrt(((np.argwhere(main) - cm)**2).sum(1).max()))
            vol = main.sum()
            sph = (36*np.pi*vol**2)**(1/3) / (4*np.pi*(vol**2)+1e-9) if vol else 0
            f[f'blob_spher_{name}'] = float(6*np.sqrt(np.pi)*vol / (4*np.pi*((3*vol/(4*np.pi))**(2/3))+1e-9)) if vol else 0
            f[f'blob_diam_{name}'] = 2*dmax
            f[f'blob_vol_{name}'] = vol
        else:
            f[f'blob_spher_{name}'] = 0.0
            f[f'blob_diam_{name}'] = 0.0
            f[f'blob_vol_{name}'] = 0.0
    return f


def process(uid):
    try:
        aligned = load_aligned(os.path.join(NIFTI_DIR, f'{uid}.nii.gz'))
        return roi_texture_features(aligned, ROIS)
    except Exception:
        return None


def _init():
    global ROIS
    rois_arr = np.load(ROIS_F)
    ROIS = {'lp': rois_arr[0], 'rp': rois_arr[1], 'lc': rois_arr[2], 'rc': rois_arr[3]}


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
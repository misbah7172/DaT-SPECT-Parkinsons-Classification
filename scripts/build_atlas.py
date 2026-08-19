import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sbr_extractor_atlas import build_template, load_aligned, split_rois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-template", required=True)
    ap.add_argument("--out-rois", required=True)
    args = ap.parse_args()

    labels = pd.read_csv(os.path.join(args.data_dir, "train_labels.csv"))
    uids = labels["uid"].tolist()
    nifti_dir = os.path.join(args.data_dir, "DaT_Parkinsons_Challenge_-_niftis.zip")

    t0 = time.time()
    acc = None
    n = 0
    for i, uid in enumerate(uids):
        sm = load_aligned(os.path.join(nifti_dir, f"{uid}.nii.gz"))
        if acc is None:
            acc = np.zeros_like(sm)
        vals = sm[sm > 0]
        if vals.size == 0:
            continue
        thr = np.percentile(vals, 95)
        acc += (sm >= thr).astype(np.float32)
        n += 1
        if (i + 1) % 200 == 0:
            print(f"[{i+1}/{len(uids)}] elapsed {time.time()-t0:.0f}s", flush=True)

    prob = acc / max(n, 1)
    rois, main = split_rois(prob)
    np.save(args.out_template, prob)
    np.save(args.out_rois, np.stack([rois["lp"], rois["rp"], rois["lc"], rois["rc"]], axis=0))
    print(f"template saved {args.out_template}; prob max {prob.max():.3f}; n={n}")
    print("roi voxels:", {k: int(v.sum()) for k, v in rois.items()})


if __name__ == "__main__":
    main()
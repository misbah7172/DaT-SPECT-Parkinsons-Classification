import argparse
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from sbr_extractor_atlas import ATLAS_FEATURE_COLUMNS, extract_atlas_features, split_rois


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--template", required=True)
    ap.add_argument("--rois", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    labels = pd.read_csv(os.path.join(args.data_dir, "train_labels.csv"))
    uids = labels["uid"].tolist()
    if args.limit > 0:
        uids = uids[: args.limit]
    nifti_dir = os.path.join(args.data_dir, "DaT_Parkinsons_Challenge_-_niftis.zip")

    rois_arr = np.load(args.rois)
    roi_masks = {"lp": rois_arr[0].astype(bool), "rp": rois_arr[1].astype(bool),
                 "lc": rois_arr[2].astype(bool), "rc": rois_arr[3].astype(bool)}
    main = roi_masks["lp"] | roi_masks["rp"] | roi_masks["lc"] | roi_masks["rc"]

    existing = set()
    if os.path.exists(args.out):
        existing = set(pd.read_csv(args.out)["uid"])

    rows = []
    t0 = time.time()
    done = 0
    for uid in uids:
        if uid in existing:
            done += 1
            continue
        f = extract_atlas_features(os.path.join(nifti_dir, f"{uid}.nii.gz"), roi_masks, main)
        f["uid"] = uid
        rows.append(f)
        done += 1
        if len(rows) >= 100:
            pd.DataFrame(rows).to_csv(args.out, index=False, mode="a", header=not os.path.exists(args.out))
            rows = []
            print(f"[{done}/{len(uids)}] elapsed {time.time()-t0:.0f}s", flush=True)
    if rows:
        pd.DataFrame(rows).to_csv(args.out, index=False, mode="a", header=not os.path.exists(args.out))
    print(f"done {done}/{len(uids)} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
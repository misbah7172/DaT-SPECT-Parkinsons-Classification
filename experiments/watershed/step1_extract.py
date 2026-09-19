"""Step 1: Extract watershed features for all subjects."""
import os, sys, time, json, warnings
import numpy as np
from scipy import ndimage
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from segmentation import watershed_atlas_seeded, watershed_intensity_seeded, watershed_combined
from features import extract_watershed_features

ROOT = "E:/DaT"
CACHE = "D:/DaT_cache"
OUT = f"{CACHE}/watershed"
os.makedirs(OUT, exist_ok=True)

# Load data
print("Loading data...", flush=True)
X_reg = np.load(f"{ROOT}/cnn3d/X_reg.npy")  # (1362, 34, 40, 42)
y = np.load(f"{ROOT}/cnn3d/y.npy")
uids = np.load(f"{ROOT}/cnn3d/uids.npy", allow_pickle=True)
groups = np.load(f"{ROOT}/cnn3d/groups.npy", allow_pickle=True)
atlas_rois = np.load(f"{ROOT}/cnn3d/atlas_rois_cropped.npy")  # (4, 34, 40, 42)
print(f"  X_reg: {X_reg.shape}, ROIs: {atlas_rois.shape}")

METHODS = ["atlas", "intensity", "combined"]

for method in METHODS:
    print(f"\n=== Method: {method} ===", flush=True)
    out_path = f"{OUT}/features_{method}.npy"
    if os.path.exists(out_path):
        feat = np.load(out_path, allow_pickle=True).item()
        print(f"  Already exists: {len(feat)} subjects", flush=True)
        continue

    all_feats = []
    t0 = time.time()
    failed = 0

    for i in range(len(X_reg)):
        vol = X_reg[i]
        try:
            if method == "atlas":
                segs = watershed_atlas_seeded(vol, atlas_rois)
            elif method == "intensity":
                segs = watershed_intensity_seeded(vol)
            else:
                segs = watershed_combined(vol, atlas_rois)
            feat = extract_watershed_features(vol, atlas_rois, segs)
            feat["uid"] = str(uids[i])
            all_feats.append(feat)
        except Exception as e:
            failed += 1
            all_feats.append({"uid": str(uids[i]), "_failed": True})

        if (i + 1) % 100 == 0 or i == len(X_reg) - 1:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (len(X_reg) - i - 1) / rate if rate > 0 else 0
            print(f"  [{i+1}/{len(X_reg)}] failed={failed} rate={rate:.1f}/s ETA={eta/60:.1f}min", flush=True)

    np.save(out_path, all_feats)
    print(f"  Saved {len(all_feats)} features ({failed} failed)", flush=True)

print("\nDone!", flush=True)

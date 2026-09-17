"""V24 SBR feature extraction from preprocessed 2.0mm volumes with PVE intensity correction."""
import os
import sys
import time
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage

sys.path.insert(0, "E:/DaT/submission_v24")
from sbr_extractor import FEATURE_COLUMNS, extract_features

CACHE_DIR = "D:/DaT_cache/volumes_2mm"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"
NIFTI_DIR = "E:/DaT/Dataset/DaT_Parkinsons_Challenge_-_niftis.zip"
OUTPUT_CSV = "D:/DaT_cache/sbr_features_train_v24.csv"

# PVE intensity correction parameters
TARGET_SPACING = 2.0
K_PVE = 0.3  # Recovery coefficient slope


def voxel_vol_from_spacing(spacing):
    return spacing[0] * spacing[1] * spacing[2]


def recovery_coefficient(v, v_ref=2.0):
    """PVE recovery coefficient for scanner spacing v."""
    return 1.0 / (1.0 + K_PVE * ((v_ref / v) ** (1.0 / 3.0) - 1.0))


def main():
    df = pd.read_csv(LABELS_PATH)
    uids = df["uid"].tolist()
    print(f"Total scans: {len(uids)}")

    results = []
    t0 = time.time()

    for i, uid in enumerate(uids, 1):
        try:
            # Extract SBR features from original NIfTI
            nii_path = os.path.join(NIFTI_DIR, f"{uid}.nii.gz")
            img = nib.load(nii_path)
            zooms = img.header.get_zooms()[:3]
            vvol = voxel_vol_from_spacing(zooms)
            spacing = zooms[0]

            features = extract_features(nii_path, vvol)

            # Apply PVE intensity correction
            rc = recovery_coefficient(spacing, TARGET_SPACING)
            for key in features:
                if key.startswith("sbr_") and key.endswith("_vol"):
                    continue
                if key.startswith("sbr_"):
                    features[key] = features[key] / rc

            features["uid"] = uid
            features["voxel_spacing"] = spacing
            features["voxel_volume"] = vvol
            features["recovery_coefficient"] = rc
            results.append(features)

        except Exception as e:
            print(f"  ERROR {uid}: {e}")
            continue

        if i % 100 == 0 or i == len(uids):
            elapsed = time.time() - t0
            rate = i / elapsed
            eta = (len(uids) - i) / rate
            print(f"  [{i}/{len(uids)}] rate={rate:.1f}/s ETA={eta/60:.1f}min")

    # Save
    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_CSV, index=False)
    total = time.time() - t0
    print(f"\nDone: {len(results)} features extracted in {total:.0f}s")
    print(f"Output: {OUTPUT_CSV}")
    print(f"Shape: {result_df.shape}")


if __name__ == "__main__":
    main()

"""V24 preprocessing: resample NIfTIs to 2.0mm isotropic 80^3 grid."""
import os
import time
import numpy as np
import pandas as pd
import nibabel as nib
from scipy import ndimage
from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform

TARGET_SPACING = 2.0
GRID = 80
CENTER = (GRID - 1) / 2.0
CACHE_DIR = "D:/DaT_cache/volumes_2mm"
NIFTI_DIR = "E:/DaT/Dataset/DaT_Parkinsons_Challenge_-_niftis.zip"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"


def reorient_ras(data, affine):
    current = io_orientation(affine)
    target = axcodes2ornt("RAS")
    transform = ornt_transform(current, target)
    return apply_orientation(data, transform)


def process_one(uid):
    out_path = os.path.join(CACHE_DIR, f"{uid}.npy")
    if os.path.exists(out_path):
        return "cached"

    nii_path = os.path.join(NIFTI_DIR, f"{uid}.nii.gz")
    if not os.path.exists(nii_path):
        return "missing"

    try:
        img = nib.load(nii_path)
        data = np.asarray(img.dataobj, dtype=np.float32)
        affine = img.affine
        header = img.header

        data = reorient_ras(data, affine)
        zooms = np.array(header.get_zooms()[:3], dtype=np.float64)
        factor = zooms / TARGET_SPACING
        new_shape = tuple(max(1, int(round(s * f))) for s, f in zip(data.shape, factor))
        zf_arr = [n / s for s, n in zip(data.shape, new_shape)]
        data = ndimage.zoom(data, zoom=zf_arr, order=1)

        head = data > np.percentile(data, 5)
        com = np.array(ndimage.center_of_mass(head)) if head.any() else np.array(data.shape) / 2.0
        offset = com - CENTER
        data = ndimage.affine_transform(
            data, matrix=np.eye(3), offset=offset,
            output_shape=(GRID, GRID, GRID), order=1, mode="constant", cval=0.0
        )
        np.save(out_path, data.astype(np.float32))
        return "ok"
    except Exception as e:
        return f"error: {e}"


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)
    df = pd.read_csv(LABELS_PATH)
    uids = df["uid"].tolist()
    print(f"Total scans: {len(uids)}")

    existing = set(f.replace(".npy", "") for f in os.listdir(CACHE_DIR) if f.endswith(".npy"))
    to_process = [u for u in uids if u not in existing]
    print(f"Already cached: {len(existing)}, to process: {len(to_process)}")

    if not to_process:
        print("All volumes already preprocessed!")
        return

    t0 = time.time()
    ok_count = 0
    err_count = 0
    errors = []

    for i, uid in enumerate(to_process, 1):
        result = process_one(uid)
        if result in ("ok", "cached"):
            ok_count += 1
        else:
            err_count += 1
            errors.append((uid, result))

        if i % 50 == 0 or i == len(to_process):
            elapsed = time.time() - t0
            rate = i / elapsed if elapsed > 0 else 0
            eta = (len(to_process) - i) / rate if rate > 0 else 0
            print(f"  [{i}/{len(to_process)}] ok={ok_count} err={err_count} "
                  f"rate={rate:.1f}/s ETA={eta/60:.1f}min")

    total = time.time() - t0
    print(f"\nDone: {ok_count} ok, {err_count} errors in {total:.0f}s")
    if errors:
        print(f"Errors: {errors[:10]}")

    # Save metadata
    meta = pd.DataFrame(errors, columns=["uid", "status"])
    meta.to_csv(os.path.join(CACHE_DIR, "..", "preprocess_errors.csv"), index=False)


if __name__ == "__main__":
    main()

"""Stage 1: Preprocessing / Harmonization.

Reorients volumes to RAS, resamples to common spacing, crops/pads to fixed matrix,
and intensity-normalizes using occipital reference region.

Usage:
    python -m pipeline.preprocess --config pipeline/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
import SimpleITK as sitk
from scipy import ndimage

from .utils import (
    ensure_dirs,
    load_config,
    load_labels_and_groups,
    make_meta,
    setup_logging,
)

log = logging.getLogger(__name__)


# ─── RAS Reorientation ─────────────────────────────────────────────────

def reorient_to_ras(data: np.ndarray, affine: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient volume to neurological RAS convention.

    Args:
        data: 3D voxel array
        affine: 4x4 affine matrix

    Returns:
        (reoriented_data, new_affine)
    """
    img = nib.Nifti1Image(data, affine)
    img = nib.as_closest_canonical(img)
    return np.asarray(img.dataobj, dtype=np.float32), img.affine


# ─── Resampling ─────────────────────────────────────────────────────────

def resample_volume(
    data: np.ndarray,
    original_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    interpolation: str = "linear",
) -> np.ndarray:
    """Resample volume to target voxel spacing using SimpleITK.

    Args:
        data: 3D numpy array (Z, Y, X) in RAS orientation
        original_spacing: (dz, dy, dx) voxel sizes in mm
        target_spacing: (dz, dy, dx) target voxel sizes in mm
        interpolation: "linear" for intensity, "nearest" for labels

    Returns:
        Resampled 3D array
    """
    sitk_interp = sitk.sitkLinear if interpolation == "linear" else sitk.sitkNearestNeighbor

    # Convert numpy -> SimpleITK (note: SimpleITK expects x,y,z ordering)
    img = sitk.GetImageFromArray(data)
    img.SetSpacing([float(s) for s in reversed(original_spacing)])  # SimpleITK uses x,y,z

    # Compute output size
    original_size = np.array(data.shape)  # z,y,x
    new_spacing = np.array(target_spacing)
    new_size = np.round(original_size * original_spacing / new_spacing).astype(int)

    # Resample
    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputSpacing([float(s) for s in reversed(new_spacing)])
    resampler.SetSize(new_size.tolist())
    resampler.SetInterpolator(sitk_interp)
    resampler.SetDefaultPixelValue(0)  # default pixel value for out-of-bounds
    resampled = resampler.Execute(img)

    return sitk.GetArrayFromImage(resampled)


# ─── Crop/Pad ───────────────────────────────────────────────────────────

def center_crop_or_pad(
    data: np.ndarray,
    target_shape: Tuple[int, int, int],
    pad_value: float = 0.0,
) -> np.ndarray:
    """Center-crop or zero-pad volume to target shape.

    Args:
        data: 3D array (Z, Y, X)
        target_shape: (D, H, W) target dimensions
        pad_value: constant for padding

    Returns:
        Cropped/padded 3D array of shape target_shape
    """
    result = np.full(target_shape, pad_value, dtype=data.dtype)
    src = np.array(data.shape)
    tgt = np.array(target_shape)

    # Compute start/end for each axis
    src_start = np.maximum((src - tgt) // 2, 0)
    src_end = np.minimum(src_start + tgt, src)

    tgt_start = np.maximum((tgt - src) // 2, 0)
    tgt_end = tgt_start + (src_end - src_start)

    # Build slices for all 3 axes simultaneously
    slices_src = tuple(slice(int(s), int(e)) for s, e in zip(src_start, src_end))
    slices_tgt = tuple(slice(int(s), int(e)) for s, e in zip(tgt_start, tgt_end))

    result[slices_tgt] = data[slices_src]
    return result


# ─── Intensity Normalization ────────────────────────────────────────────

def normalize_occipital(
    data: np.ndarray,
    occipital_pct: float = 0.10,
) -> np.ndarray:
    """Normalize using occipital reference region heuristic.

    Uses posterior axial slices, low-uptake region as approximation
    of non-specific binding. This is a heuristic — flagged for refinement.

    Args:
        data: 3D array (Z, Y, X) in RAS orientation
        occipital_pct: fraction of lowest-intensity posterior voxels to use

    Returns:
        Normalized volume: (voxel - ref_mean) / ref_mean
    """
    z_dim = data.shape[0]

    # Take posterior 40% of axial slices (occipital lobe region)
    posterior_start = int(z_dim * 0.60)
    posterior_vol = data[posterior_start:, :, :]

    if posterior_vol.size == 0:
        # Fallback: use overall low percentile
        ref_val = np.percentile(data, occipital_pct * 100)
    else:
        # Within posterior slices, take bottom occipital_pct by intensity
        threshold = np.percentile(posterior_vol, occipital_pct * 100)
        ref_mask = posterior_vol <= threshold
        if ref_mask.sum() > 0:
            ref_val = posterior_vol[ref_mask].mean()
        else:
            ref_val = np.percentile(data, occipital_pct * 100)

    # Avoid division by zero
    ref_val = max(ref_val, 1e-8)

    return (data - ref_val) / ref_val


def normalize_percentile(data: np.ndarray, p_low: float = 1.0, p_high: float = 99.0) -> np.ndarray:
    """Percentile-based normalization to [0, 1]."""
    low = np.percentile(data, p_low)
    high = np.percentile(data, p_high)
    high = max(high - low, 1e-8)
    return np.clip((data - low) / high, 0, 1)


def normalize_zscore(data: np.ndarray) -> np.ndarray:
    """Z-score normalization."""
    mu, sigma = data.mean(), data.std()
    sigma = max(sigma, 1e-8)
    return (data - mu) / sigma


# ─── NIfTI Loading from Zip ─────────────────────────────────────────────

def load_nifti_from_zip(zip_path: str, uid: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single NIfTI from the competition zip archive.

    Returns:
        (data, affine) tuple
    """
    with zipfile.ZipFile(zip_path, "r") as zf:
        # Find matching file (uid.nii.gz)
        candidates = [n for n in zf.namelist() if uid in n and n.endswith(".nii.gz")]
        if not candidates:
            raise FileNotFoundError(f"No NIfTI found for uid={uid} in {zip_path}")

        with zf.open(candidates[0]) as f:
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

        try:
            img = nib.load(tmp_path)
            data = np.asarray(img.dataobj, dtype=np.float32)
            affine = img.affine
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    return data, affine


def extract_uid_from_zip(zip_path: str) -> List[str]:
    """Extract list of UIDs from zip filenames."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        uids = []
        for name in zf.namelist():
            if name.endswith(".nii.gz"):
                # Extract uid from filename like "01nouhtc.nii.gz" or "path/to/01nouhtc.nii.gz"
                uid = Path(name).stem.replace(".nii", "")
                uids.append(uid)
        return sorted(uids)


# ─── Single Volume Pipeline ─────────────────────────────────────────────

def harmonize_volume(
    data: np.ndarray,
    affine: np.ndarray,
    target_spacing: Tuple[float, float, float],
    crop_size: Tuple[int, int, int],
    normalize_method: str = "occipital",
    occipital_pct: float = 0.10,
    interpolation: str = "linear",
) -> np.ndarray:
    """Full harmonization pipeline for a single volume.

    1. Reorient to RAS
    2. Resample to target spacing
    3. Center crop/pad to fixed matrix
    4. Intensity normalize

    Returns:
        Harmonized 3D array
    """
    # 1. RAS reorientation
    data, affine = reorient_to_ras(data, affine)

    # 2. Determine original spacing from affine
    original_spacing = tuple(
        np.sqrt(np.sum(affine[:3, i] ** 2)) for i in range(3)
    )

    # 3. Resample to target spacing
    data = resample_volume(data, original_spacing, target_spacing, interpolation)

    # 4. Crop/pad to target matrix
    # Reorder crop_size from (H, W, D) to (D, H, W) for numpy
    crop_npy = (crop_size[2], crop_size[0], crop_size[1])
    data = center_crop_or_pad(data, crop_npy)

    # 5. Intensity normalize
    if normalize_method == "occipital":
        data = normalize_occipital(data, occipital_pct)
    elif normalize_method == "percentile":
        data = normalize_percentile(data)
    elif normalize_method == "zscore":
        data = normalize_zscore(data)
    elif normalize_method != "none":
        raise ValueError(f"Unknown normalization: {normalize_method}")

    return data


# ─── Batch Processing ───────────────────────────────────────────────────

def process_single_scan(
    uid: str,
    zip_path: str,
    cfg: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """Process one scan: load, harmonize, save.

    Returns metadata dict for this scan.
    """
    pp = cfg["preprocessing"]
    target_spacing = tuple(pp["target_spacing"])
    crop_size = tuple(pp["crop_size"])

    try:
        # Load
        data, affine = load_nifti_from_zip(zip_path, uid)
        orig_shape = data.shape
        orig_spacing = tuple(
            np.sqrt(np.sum(affine[:3, i] ** 2)) for i in range(3)
        )

        # Harmonize
        harmonized = harmonize_volume(
            data, affine,
            target_spacing=target_spacing,
            crop_size=crop_size,
            normalize_method=pp["normalize"],
            occipital_pct=pp.get("occipital_pct", 0.10),
            interpolation=pp.get("interpolation", "linear"),
        )

        # Save
        out_path = output_dir / f"{uid}.npy"
        np.save(str(out_path), harmonized)

        return {
            "uid": uid,
            "status": "ok",
            "orig_shape": list(orig_shape),
            "orig_spacing": [round(s, 3) for s in orig_spacing],
            "harmonized_shape": list(harmonized.shape),
            "output_path": str(out_path),
        }
    except Exception as e:
        log.error(f"Failed to process {uid}: {e}")
        return {
            "uid": uid,
            "status": "error",
            "error": str(e),
        }


def run_preprocessing(cfg: Dict[str, Any], n_workers: int = 1) -> pd.DataFrame:
    """Run full preprocessing pipeline.

    Returns:
        DataFrame with per-scan metadata (status, shapes, paths)
    """
    dirs = ensure_dirs(cfg)
    meta = make_meta(cfg, "preprocess")
    meta.save(dirs["base"] / "preprocess_meta.json")

    data_dir = Path(cfg["paths"]["data_dir"])
    zip_path = data_dir / "DaT_Parkinsons_Challenge_-_niftis.zip"
    cache_dir = dirs["cache"]
    cache_dir.mkdir(parents=True, exist_ok=True)

    # Get UIDs
    df, groups = load_labels_and_groups(data_dir)
    uids = df["uid"].tolist()

    log.info(f"Processing {len(uids)} scans → {cache_dir}")

    # Check cache — skip already-processed scans
    existing = {p.stem for p in cache_dir.glob("*.npy")}
    to_process = [u for u in uids if u not in existing]
    log.info(f"Cache hit: {len(existing)}, to process: {len(to_process)}")

    # Process
    results = []
    for uid in existing:
        results.append({"uid": uid, "status": "cached"})

    if to_process:
        if n_workers <= 1:
            for i, uid in enumerate(to_process):
                result = process_single_scan(uid, str(zip_path), cfg, cache_dir)
                results.append(result)
                if (i + 1) % 50 == 0:
                    log.info(f"  Processed {i+1}/{len(to_process)}")
        else:
            with ProcessPoolExecutor(max_workers=n_workers) as executor:
                futures = {
                    executor.submit(
                        process_single_scan, uid, str(zip_path), cfg, cache_dir
                    ): uid
                    for uid in to_process
                }
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    results.append(result)
                    if (i + 1) % 50 == 0:
                        log.info(f"  Processed {i+1}/{len(to_process)}")

    # Build metadata DataFrame
    meta_df = pd.DataFrame(results)

    # Merge scanner group info
    meta_df = meta_df.merge(df[["uid", "is_pathologic", "scanner_group"]], on="uid", how="left")

    # Save metadata
    meta_path = cache_dir / "preprocess_metadata.csv"
    meta_df.to_csv(meta_path, index=False)
    log.info(f"Metadata saved to {meta_path}")

    # Report
    n_ok = (meta_df["status"].isin(["ok", "cached"])).sum()
    n_err = (meta_df["status"] == "error").sum()
    log.info(f"Done: {n_ok} OK, {n_err} errors out of {len(uids)}")

    return meta_df


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 1: Preprocess / Harmonize NIfTI volumes")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Path to config YAML")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers")
    parser.add_argument("--force", action="store_true", help="Re-process even if cached")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(Path(cfg["paths"]["output_dir"]) / "logs", "preprocess")

    if args.force:
        import shutil
        cache_dir = Path(cfg["paths"]["output_dir"]) / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            log.info(f"Cleared cache: {cache_dir}")

    run_preprocessing(cfg, n_workers=args.workers)


if __name__ == "__main__":
    main()

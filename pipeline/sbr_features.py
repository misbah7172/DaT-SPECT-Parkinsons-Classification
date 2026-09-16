"""Stage 2: SBR Feature Extraction.

Computes striatal binding ratio features directly from harmonized volumes.
Independent of CNN — produces usable features even if CNN training is broken.

Usage:
    python -m pipeline.sbr_features --config pipeline/config.yaml
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import ndimage

from .utils import ensure_dirs, load_config, make_meta, setup_logging

log = logging.getLogger(__name__)


# ─── ROI Definition Heuristics ──────────────────────────────────────────

def estimate_striatum_mask(
    volume: np.ndarray,
    sigma: float = 2.0,
) -> np.ndarray:
    """Estimate striatum ROI from DaT-SPECT volume.

    Uses intensity thresholding + connected components on smoothed volume.
    The striatum is the highest-uptake bilateral structure in the center
    of the brain.

    Args:
        volume: 3D harmonized DaT-SPECT volume (Z, Y, X)
        sigma: Gaussian smoothing sigma

    Returns:
        Binary mask of estimated striatum region
    """
    smoothed = ndimage.gaussian_filter(volume, sigma=sigma)

    # Top 15% intensity voxels as initial striatum estimate
    threshold = np.percentile(smoothed, 85)
    mask = smoothed > threshold

    # Keep only the central region (striatum is central)
    z_dim, y_dim, x_dim = mask.shape
    # Central 50% in each axis
    z_margin = z_dim // 4
    y_margin = y_dim // 4
    x_margin = x_dim // 4

    central_mask = np.zeros_like(mask)
    central_mask[
        z_margin:z_dim - z_margin,
        y_margin:y_dim - y_margin,
        x_margin:x_dim - x_margin,
    ] = True

    mask = mask & central_mask

    # Morphological cleanup
    mask = ndimage.binary_closing(mask, iterations=2)
    mask = ndimage.binary_opening(mask, iterations=1)

    # Keep only largest connected components (top 2 = left + right striatum)
    labeled, n_features = ndimage.label(mask)
    if n_features > 0:
        component_sizes = ndimage.sum(mask, labeled, range(1, n_features + 1))
        top_components = np.argsort(component_sizes)[-2:] + 1  # top 2
        mask = np.isin(labeled, top_components)

    return mask.astype(bool)


def split_hemispheres(
    mask: np.ndarray,
    volume: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split a bilateral ROI into left and right hemispheres.

    Uses the midline (center of mass X-axis) as the split point.

    Returns:
        (left_mask, right_mask)
    """
    com = ndimage.center_of_mass(volume > np.percentile(volume, 50))
    x_mid = int(round(com[2]))  # X-axis is the third dimension

    left_mask = mask.copy()
    left_mask[:, :, x_mid:] = 0

    right_mask = mask.copy()
    right_mask[:, :, :x_mid] = 0

    return left_mask, right_mask


def estimate_occipital_mask(
    volume: np.ndarray,
    occipital_pct: float = 0.10,
) -> np.ndarray:
    """Estimate occipital reference region (posterior, low-uptake).

    Uses posterior axial slices and bottom intensity percentiles.
    """
    z_dim = volume.shape[0]

    # Posterior 40% of axial slices
    posterior_start = int(z_dim * 0.60)
    posterior_vol = volume[posterior_start:, :, :]

    # Bottom occipital_pct by intensity within posterior region
    threshold = np.percentile(posterior_vol, occipital_pct * 100)
    mask = np.zeros_like(volume, dtype=bool)
    mask[posterior_start:, :, :] = posterior_vol <= threshold

    return mask.astype(bool)


# ─── Feature Computation ────────────────────────────────────────────────

def compute_sbr_features(
    volume: np.ndarray,
    sigma_list: List[float] = [0.5, 2.0],
    thresholds: List[float] = [0.95, 0.97, 0.98, 0.99],
    occipital_pct: float = 0.10,
) -> Dict[str, float]:
    """Compute full SBR feature set from a harmonized volume.

    Features include:
    - Overall SBR (striatum-to-occipital ratio)
    - Per-hemisphere SBR (left/right)
    - Asymmetry index
    - Sub-region ratios (putamen/caudate estimates)
    - Multi-threshold SBR values

    Args:
        volume: 3D harmonized DaT-SPECT volume
        sigma_list: smoothing levels for ROI definition
        thresholds: intensity percentiles for ROI thresholding
        occipital_pct: fraction of posterior voxels for occipital reference

    Returns:
        Dict of feature_name -> value
    """
    features = {}

    occipital_mask = estimate_occipital_mask(volume, occipital_pct)
    occipital_val = volume[occipital_mask].mean() if occipital_mask.sum() > 0 else 1e-8
    occipital_val = max(occipital_val, 1e-8)

    for sigma in sigma_list:
        striatum_mask = estimate_striatum_mask(volume, sigma=sigma)
        left_mask, right_mask = split_hemispheres(striatum_mask, volume)

        # Overall SBR
        striatal_val = volume[striatum_mask].mean() if striatum_mask.sum() > 0 else 0.0
        sbr = striatal_val / occipital_val
        features[f"sbr_sigma{sigma}"] = sbr

        # Left/right SBR
        left_val = volume[left_mask].mean() if left_mask.sum() > 0 else 0.0
        right_val = volume[right_mask].mean() if right_mask.sum() > 0 else 0.0
        sbr_left = left_val / occipital_val
        sbr_right = right_val / occipital_val
        features[f"sbr_left_sigma{sigma}"] = sbr_left
        features[f"sbr_right_sigma{sigma}"] = sbr_right

        # Asymmetry index: (left - right) / (left + right)
        asym_num = sbr_left - sbr_right
        asym_den = sbr_left + sbr_right
        asymmetry = asym_num / max(abs(asym_den), 1e-8)
        features[f"asymmetry_sigma{sigma}"] = asymmetry

        # Absolute asymmetry
        features[f"abs_asymmetry_sigma{sigma}"] = abs(asymmetry)

        # Sub-region estimates: anterior/posterior split of striatum
        z_mid = striatum_mask.shape[0] // 2
        anterior_mask = striatum_mask.copy()
        anterior_mask[z_mid:, :, :] = 0
        posterior_mask = striatum_mask.copy()
        posterior_mask[:z_mid, :, :] = 0

        anterior_val = volume[anterior_mask].mean() if anterior_mask.sum() > 0 else 0.0
        posterior_val = volume[posterior_mask].mean() if posterior_mask.sum() > 0 else 0.0

        features[f"anterior_sbr_sigma{sigma}"] = anterior_val / occipital_val
        features[f"posterior_sbr_sigma{sigma}"] = posterior_val / occipital_val

        # Anterior-posterior ratio
        features[f"ap_ratio_sigma{sigma}"] = anterior_val / max(posterior_val, 1e-8)

    # Multi-threshold SBR
    for thresh in thresholds:
        threshold_val = np.percentile(volume, thresh * 100)
        thresh_mask = (volume > threshold_val) & (striatum_mask > 0)
        thresh_val = volume[thresh_mask].mean() if thresh_mask.sum() > 0 else 0.0
        features[f"sbr_p{int(thresh*100)}"] = thresh_val / occipital_val

    # Global features
    features["global_mean"] = volume.mean()
    features["global_std"] = volume.std()
    features["global_max"] = volume.max()
    features["global_p95"] = np.percentile(volume, 95)
    features["global_p99"] = np.percentile(volume, 99)

    # Volume statistics
    features["striatal_volume"] = float(striatum_mask.sum())
    features["striatal_fraction"] = float(striatum_mask.sum() / volume.size)

    return features


# ─── Batch Processing ───────────────────────────────────────────────────

def extract_sbr_features_batch(
    cfg: Dict[str, Any],
    meta_df: pd.DataFrame,
) -> pd.DataFrame:
    """Extract SBR features for all scans.

    Args:
        cfg: Pipeline config
        meta_df: Preprocessing metadata (must have 'uid' and 'status' columns)

    Returns:
        DataFrame with uid + all SBR features
    """
    sbr_cfg = cfg.get("sbr", {})
    cache_dir = Path(cfg["paths"]["output_dir"]) / "cache"

    # Filter to successfully preprocessed scans
    ok_uids = meta_df[meta_df["status"].isin(["ok", "cached"])]["uid"].tolist()
    log.info(f"Extracting SBR features for {len(ok_uids)} scans")

    all_features = []
    for i, uid in enumerate(ok_uids):
        vol_path = cache_dir / f"{uid}.npy"
        if not vol_path.exists():
            log.warning(f"Missing harmonized volume: {uid}")
            continue

        volume = np.load(str(vol_path))
        features = compute_sbr_features(
            volume,
            sigma_list=sbr_cfg.get("sigma_list", [0.5, 2.0]),
            thresholds=sbr_cfg.get("thresholds", [0.95, 0.97, 0.98, 0.99]),
            occipital_pct=cfg.get("preprocessing", {}).get("occipital_pct", 0.10),
        )
        features["uid"] = uid
        all_features.append(features)

        if (i + 1) % 100 == 0:
            log.info(f"  Extracted {i+1}/{len(ok_uids)}")

    df = pd.DataFrame(all_features)

    # Reorder columns: uid first, then sorted features
    cols = ["uid"] + sorted([c for c in df.columns if c != "uid"])
    df = df[cols]

    # Save
    out_dir = Path(cfg["paths"]["output_dir"])
    out_path = out_dir / sbr_cfg.get("output_csv", "sbr_features.csv")
    df.to_csv(out_path, index=False)
    log.info(f"SBR features saved: {out_path} ({df.shape[1]-1} features, {len(df)} scans)")

    return df


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 2: Extract SBR features from harmonized volumes")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Path to config YAML")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(Path(cfg["paths"]["output_dir"]) / "logs", "sbr")

    # Load preprocessing metadata
    cache_dir = Path(cfg["paths"]["output_dir"]) / "cache"
    meta_path = cache_dir / "preprocess_metadata.csv"
    if not meta_path.exists():
        log.error(f"Preprocessing metadata not found: {meta_path}")
        log.error("Run preprocessing first: python -m pipeline.preprocess")
        return

    meta_df = pd.read_csv(meta_path)
    extract_sbr_features_batch(cfg, meta_df)


if __name__ == "__main__":
    main()

"""Watershed feature extraction for DaT-SPECT."""
import numpy as np
from skimage.measure import regionprops


def extract_region_features(volume, mask):
    vals = volume[mask]
    props = regionprops(mask.astype(int))
    f = {}
    f["mean"] = float(vals.mean())
    f["median"] = float(np.median(vals))
    f["std"] = float(vals.std())
    f["min"] = float(vals.min())
    f["max"] = float(vals.max())
    f["p10"] = float(np.percentile(vals, 10))
    f["p25"] = float(np.percentile(vals, 25))
    f["p75"] = float(np.percentile(vals, 75))
    f["p90"] = float(np.percentile(vals, 90))
    f["voxel_count"] = int(mask.sum())
    f["physical_volume"] = float(mask.sum())
    if props:
        p = props[0]
        bb = p.bbox
        f["bb_dim0"] = int(bb[3] - bb[0])
        f["bb_dim1"] = int(bb[4] - bb[1])
        f["bb_dim2"] = int(bb[5] - bb[2])
        f["centroid_0"] = float(p.centroid[0])
        f["centroid_1"] = float(p.centroid[1])
        f["centroid_2"] = float(p.centroid[2])
        # Compactness: use surface area estimate for 3D
        # Surface voxels = voxels with at least one 26-neighbor outside the mask
        from scipy.ndimage import binary_erosion
        eroded = binary_erosion(mask, iterations=1)
        surface = mask.sum() - eroded.sum()
        f["compactness"] = float(surface**2 / (36 * np.pi * (p.area**2))) if p.area > 0 and surface > 0 else 0.0
    else:
        for k in ["bb_dim0", "bb_dim1", "bb_dim2", "centroid_0", "centroid_1", "centroid_2", "compactness"]:
            f[k] = 0.0
    return f


def extract_watershed_features(volume, atlas_rois, segments):
    n_regions = len(np.unique(segments)) - (1 if 0 in segments else 0)
    feat = {"n_watershed_regions": n_regions}

    # ROI intensity features
    roi_names = ["lp", "rp", "lc", "rc"]
    roi_feats = {}
    for i, name in enumerate(roi_names):
        roi_mask = atlas_rois[i] if i < atlas_rois.shape[0] else np.zeros_like(volume, dtype=bool)
        roi_vals = volume[roi_mask]
        if len(roi_vals) > 0:
            roi_feats[name + "_mean"] = float(roi_vals.mean())
            roi_feats[name + "_std"] = float(roi_vals.std())
            roi_feats[name + "_max"] = float(roi_vals.max())
            roi_feats[name + "_min"] = float(roi_vals.min())
            roi_feats[name + "_median"] = float(np.median(roi_vals))
            roi_feats[name + "_vol"] = float(roi_mask.sum())
            roi_feats[name + "_p10"] = float(np.percentile(roi_vals, 10))
            roi_feats[name + "_p90"] = float(np.percentile(roi_vals, 90))
        else:
            for s in ["_mean", "_std", "_max", "_min", "_median", "_vol", "_p10", "_p90"]:
                roi_feats[name + s] = 0.0
    feat.update(roi_feats)

    # Asymmetry features
    eps = 1e-6
    l_put = roi_feats.get("lp_mean", 0)
    r_put = roi_feats.get("rp_mean", 0)
    l_cau = roi_feats.get("lc_mean", 0)
    r_cau = roi_feats.get("rc_mean", 0)

    feat["put_asym_abs"] = abs(l_put - r_put)
    feat["put_asym_rel"] = abs(l_put - r_put) / (max(abs(l_put), abs(r_put)) + eps)
    feat["put_ratio_lr"] = l_put / (r_put + eps)
    feat["put_ratio_rl"] = r_put / (l_put + eps)
    feat["cau_asym_abs"] = abs(l_cau - r_cau)
    feat["cau_asym_rel"] = abs(l_cau - r_cau) / (max(abs(l_cau), abs(r_cau)) + eps)
    feat["cau_ratio_lr"] = l_cau / (r_cau + eps)
    feat["cau_ratio_rl"] = r_cau / (l_cau + eps)

    l_vol = roi_feats.get("lp_vol", 0) + roi_feats.get("lc_vol", 0)
    r_vol = roi_feats.get("rp_vol", 0) + roi_feats.get("rc_vol", 0)
    feat["total_vol_l"] = l_vol
    feat["total_vol_r"] = r_vol
    feat["vol_asym_abs"] = abs(l_vol - r_vol)
    feat["vol_asym_rel"] = abs(l_vol - r_vol) / (max(l_vol, r_vol) + eps)

    # Region overlap features
    for region_id in np.unique(segments):
        if region_id == 0:
            continue
        region_mask = segments == region_id
        rf = extract_region_features(volume, region_mask)
        if rf is None:
            continue
        overlap_scores = []
        for i in range(min(4, atlas_rois.shape[0])):
            overlap = (region_mask & atlas_rois[i]).sum()
            overlap_scores.append(overlap)
        best_roi = np.argmax(overlap_scores) if max(overlap_scores) > 0 else -1
        prefix = f"ws_r{region_id}"
        feat[prefix + "_mean"] = rf["mean"]
        feat[prefix + "_std"] = rf["std"]
        feat[prefix + "_vol"] = rf["physical_volume"]
        feat[prefix + "_compactness"] = rf["compactness"]
        feat[prefix + "_best_roi"] = float(best_roi)

    # QC
    feat["qc_total_voxels"] = int((segments > 0).sum())
    feat["qc_background_voxels"] = int((segments == 0).sum())
    for i, name in enumerate(roi_names):
        roi_mask = atlas_rois[i] if i < atlas_rois.shape[0] else np.zeros_like(volume, dtype=bool)
        ws_in_roi = segments[roi_mask]
        feat[f"qc_{name}_n_regions"] = len(np.unique(ws_in_roi)) - (1 if 0 in ws_in_roi else 0)

    return feat

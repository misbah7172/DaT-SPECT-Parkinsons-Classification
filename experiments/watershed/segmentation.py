"""Watershed segmentation module for DaT-SPECT."""
import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from skimage.measure import regionprops


def watershed_atlas_seeded(volume, atlas_rois, smooth_sigma=1.0):
    sm = ndimage.gaussian_filter(volume, sigma=smooth_sigma)
    markers = np.zeros(volume.shape, dtype=np.int32)
    mid = 1
    for i in range(atlas_rois.shape[0]):
        roi = atlas_rois[i]
        if roi.sum() > 0:
            markers[roi] = mid
            mid += 1
    pos = sm[sm > 0]
    bg_mask = sm < np.percentile(pos, 10) if len(pos) > 0 else np.ones_like(sm, dtype=bool)
    markers[bg_mask] = mid
    try:
        return watershed(-sm, markers, mask=sm > 0)
    except Exception:
        return np.zeros_like(sm, dtype=np.int32)


def watershed_intensity_seeded(volume, smooth_sigma=1.0, min_distance=5):
    sm = ndimage.gaussian_filter(volume, sigma=smooth_sigma)
    if sm.max() <= 0:
        return np.zeros_like(sm, dtype=np.int32)
    coords = peak_local_max(sm, min_distance=min_distance, exclude_border=False)
    if len(coords) == 0:
        return np.zeros_like(sm, dtype=np.int32)
    markers = np.zeros(sm.shape, dtype=np.int32)
    for i, c in enumerate(coords, 1):
        markers[tuple(c)] = i
    pos = sm[sm > 0]
    bg_mask = sm < np.percentile(pos, 5) if len(pos) > 0 else np.ones_like(sm, dtype=bool)
    markers[bg_mask] = len(coords) + 1
    try:
        return watershed(-sm, markers, mask=sm > 0)
    except Exception:
        return np.zeros_like(sm, dtype=np.int32)


def watershed_combined(volume, atlas_rois, smooth_sigma=1.0, max_markers=20):
    sm = ndimage.gaussian_filter(volume, sigma=smooth_sigma)
    markers = np.zeros(volume.shape, dtype=np.int32)
    mid = 1
    for i in range(atlas_rois.shape[0]):
        roi = atlas_rois[i]
        if roi.sum() > 0:
            markers[roi] = mid
            mid += 1
    pos = sm[sm > 0]
    if len(pos) > 0:
        thresh = np.percentile(pos, 20)
        unseeded = (markers == 0) & (sm > thresh)
        if unseeded.any():
            lm = peak_local_max(sm, min_distance=3, exclude_border=False)
            for c in lm:
                if unseeded[tuple(c)]:
                    markers[tuple(c)] = mid
                    mid += 1
                    if mid > max_markers:
                        break
        bg_mask = (markers == 0) & (sm < np.percentile(pos, 10))
    else:
        bg_mask = np.ones_like(sm, dtype=bool)
    markers[bg_mask] = mid
    try:
        return watershed(-sm, markers, mask=sm > 0)
    except Exception:
        return np.zeros_like(sm, dtype=np.int32)

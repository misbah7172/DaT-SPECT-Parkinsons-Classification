import numpy as np
import nibabel as nib
import scipy.ndimage as ndi

TARGET_SPACING = 2.5
GRID = 100  # voxels per dim at 2.5mm => 250mm FOV
CENTER = (GRID - 1) / 2.0


def _reorient_ras(data, affine):
    from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
    current = io_orientation(affine)
    target = axcodes2ornt("RAS")
    transform = ornt_transform(current, target)
    return apply_orientation(data, transform)


def load_aligned(nii_path, target=TARGET_SPACING, grid=GRID):
    """Load a scan into a canonical aligned grid.

    RAS orientation, isotropic resampling to `target` mm, then translated so the
    head center-of-mass lands at the grid center. All scans share one coordinate
    frame here -- the foundation for atlas ROIs.
    """
    img = nib.load(str(nii_path))
    data = _reorient_ras(img.get_fdata().astype(np.float32), img.affine)
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    factor = zooms / target
    shape = tuple(max(1, int(round(s * f))) for s, f in zip(data.shape, factor))
    zf = [n / s for s, n in zip(data.shape, shape)]
    data = ndi.zoom(data, zoom=zf, order=1)
    head = data > np.percentile(data, 5)
    com = np.array(ndi.center_of_mass(head)) if head.any() else np.array(data.shape) / 2.0
    offset = com - CENTER
    aligned = ndi.affine_transform(
        data, matrix=np.eye(3), offset=offset, output_shape=(grid, grid, grid),
        order=1, mode="constant", cval=0.0,
    )
    return aligned


def build_template(smoothed_aligneds, thresh_pctile=95.0):
    """Average striatal (>= p95) binary masks across scans -> anatomy probability atlas."""
    n = len(smoothed_aligneds)
    prob = np.zeros_like(smoothed_aligneds[0])
    for sm in smoothed_aligneds:
        vals = sm[sm > 0]
        if vals.size == 0:
            continue
        thr = np.percentile(vals, thresh_pctile)
        prob += (sm >= thr).astype(np.float32)
    return prob / max(n, 1)


def split_rois(probmap, thresh=0.4):
    """Derive putamen/caudate L/R ROI masks from the anatomy probability atlas."""
    M = (probmap >= thresh)
    M = ndi.binary_opening(M, iterations=2)
    lab, nlab = ndi.label(M)
    if nlab == 0:
        return {}, M
    sizes = ndi.sum(M, lab, range(1, nlab + 1))
    main = lab == (sizes.argmax() + 1)
    coords = np.argwhere(main)
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    cx = x.mean()
    cy = y.mean()
    left = main.copy()
    for (xx, yy, zz) in coords[coords[:, 0] <= cx]:
        left[xx, yy, zz] = False
    right = main.copy()
    for (xx, yy, zz) in coords[coords[:, 0] > cx]:
        right[xx, yy, zz] = False
    putamen = main.copy()
    for (xx, yy, zz) in coords[coords[:, 1] >= cy]:
        putamen[xx, yy, zz] = False
    caudate = main.copy()
    for (xx, yy, zz) in coords[coords[:, 1] < cy]:
        caudate[xx, yy, zz] = False
    rois = {
        "lp": left & putamen,
        "rp": right & putamen,
        "lc": left & caudate,
        "rc": right & caudate,
    }
    return rois, main


def tissue_roi(sm, seed=50.0):
    """Background tissue region for reference: head voxels below p60 of suit."""
    lives = sm > np.percentile(sm, 5)
    sv = sm[lives]
    lo = np.percentile(sv, 25)
    hi = np.percentile(sv, 50)
    if hi <= lo:
        hi = lo + 1e-6
    return lives & (sm >= lo) & (sm <= hi)


def extract_atlas_features(nii_path, rois, main, target=TARGET_SPACING, grid=GRID):
    """Compute clinical-style binding features over fixed anatomical ROI masks."""
    sm = load_aligned(nii_path, target, grid)
    sm = ndi.gaussian_filter(sm, sigma=1.0)
    lives = sm > np.percentile(sm, 5)
    sv = sm[lives]
    lo = np.percentile(sv, 25)
    hi = np.percentile(sv, 50)
    bg = sv[(sv >= lo) & (sv <= hi)].mean() if (sv >= lo).any() else np.float64(sm.mean())

    def sbr(m):
        sel = sm[m]
        if sel.size == 0:
            return np.nan
        return (sel.mean() - bg) / bg

    def vol(m):
        return int(m.sum()) * (target ** 3)

    f = {}
    for name in ("lp", "rp", "lc", "rc"):
        m = rois[name]
        f[f"sbr_{name}"] = sbr(m)
        f[f"vol_{name}_mm3"] = vol(m)

    lp, rp, lc, rc = f["sbr_lp"], f["sbr_rp"], f["sbr_lc"], f["sbr_rc"]
    vlp, vlr, vlc, vrc = f["vol_lp_mm3"], f["vol_rp_mm3"], f["vol_lc_mm3"], f["vol_rc_mm3"]
    f["sbr_left_total"] = (lp + lc) / 2.0
    f["sbr_right_total"] = (rp + rc) / 2.0
    f["sbr_total"] = (lp + rp + lc + rc) / 4.0
    f["vol_striatal_mm3"] = vlp + vlr + vlc + vrc
    f["frac_putamen_sbr"] = (lp + rp) / (lp + rp + lc + rc + 1e-6)
    f["frac_caudate_sbr"] = (lc + rc) / (lp + rp + lc + rc + 1e-6)
    f["min_pc_ratio"] = min(lp / lc, rp / rc) if lc and rc else np.nan
    f["mean_pc_ratio"] = ((lp + rp) / 2.0) / ((lc + rc) / 2.0 + 1e-6)
    f["putamen_asym"] = abs(lp - rp) / max(abs(lp), abs(rp), 1e-9)
    f["caudate_asym"] = abs(lc - rc) / max(abs(lc), abs(rc), 1e-9)
    f["vol_putamen_asym"] = abs(vlp - vlr) / max(vlp + vlr, 1e-9)
    f["vol_caudate_asym"] = abs(vlc - vrc) / max(vlc + vrc, 1e-9)
    f["min_putamen_sbr"] = min(lp, rp)
    f["min_caudate_sbr"] = min(lc, rc)

    # posterior-putamen emphasis: posterior half of the putamen masks (toward -y)
    for side, m in (("l", rois["lp"]), ("r", rois["rp"])):
        coords = np.argwhere(m)
        if coords.size == 0:
            f[f"post_{side}_sbr"] = np.nan
            continue
        med_y = coords[:, 1].mean()
        post = m.copy()
        post = m & (np.indices(m.shape)[1] < med_y)
        f[f"post_{side}_sbr"] = sbr(post)

    f["post_asym"] = abs(f["post_l_sbr"] - f["post_r_sbr"]) / max(abs(f["post_l_sbr"]), abs(f["post_r_sbr"]), 1e-9)
    f["min_post_putamen_sbr"] = min(f["post_l_sbr"], f["post_r_sbr"])

    # region-wise intensity stats
    for name in ("lp", "rp", "lc", "rc"):
        sel = sm[rois[name]]
        if sel.size == 0:
            f[f"roi_{name}_max"] = np.nan
            f[f"roi_{name}_cv"] = np.nan
        else:
            f[f"roi_{name}_max"] = (sel.max() - bg) / bg
            f[f"roi_{name}_cv"] = sel.std() / sel.mean()

    return f


ATLAS_FEATURE_COLUMNS = [
    "sbr_lp", "sbr_rp", "sbr_lc", "sbr_rc",
    "vol_lp_mm3", "vol_rp_mm3", "vol_lc_mm3", "vol_rc_mm3",
    "sbr_left_total", "sbr_right_total", "sbr_total", "vol_striatal_mm3",
    "frac_putamen_sbr", "frac_caudate_sbr", "min_pc_ratio", "mean_pc_ratio",
    "putamen_asym", "caudate_asym", "vol_putamen_asym", "vol_caudate_asym",
    "min_putamen_sbr", "min_caudate_sbr",
    "post_l_sbr", "post_r_sbr", "post_asym", "min_post_putamen_sbr",
    "roi_lp_max", "roi_rp_max", "roi_lc_max", "roi_rc_max",
    "roi_lp_cv", "roi_rp_cv", "roi_lc_cv", "roi_rc_cv",
]
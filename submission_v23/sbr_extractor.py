import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from scipy import stats


def _reorient_ras(data, affine):
    from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
    current = io_orientation(affine)
    target = axcodes2ornt("RAS")
    transform = ornt_transform(current, target)
    return apply_orientation(data, transform)


def _sbr(values, bg):
    return (values.mean() - bg) / bg


def _region_values(sm, coords, mask):
    x = coords[mask, 0]
    y = coords[mask, 1]
    z = coords[mask, 2]
    return sm[x, y, z]


def extract_features(nii_path):
    img = nib.load(str(nii_path))
    data = img.get_fdata().astype(np.float32)
    data = _reorient_ras(data, img.affine)

    sm = ndi.gaussian_filter(data, sigma=1.0)

    head = sm > np.percentile(sm, 5)
    sv = sm[head]
    p25 = np.percentile(sv, 25)
    p50 = np.percentile(sv, 50)
    bg = sv[(sv >= p25) & (sv <= p50)].mean()

    thr = np.percentile(sv, 98)
    co = np.argwhere(sm >= thr)
    x = co[:, 0]
    y = co[:, 1]
    z = co[:, 2]

    head_coords = np.argwhere(head)
    hx = head_coords[:, 0].mean()
    cy = y.mean()

    def region_mask(hside_p=False, y_pro=False, cx=None, cyy=None):
        left = (x > cx) if hside_p else (x <= cx)
        return left & (y >= cyy if y_pro else y < cyy)

    def region_sbr(hside_p, y_pro, cx=None, cyy=None):
        m = region_mask(hside_p, y_pro, cx, cyy)
        return _sbr(sm[x[m], y[m], z[m]], bg)

    feat = {}

    rl = True
    rr = False
    up = False
    uc = True

    lp = region_sbr(rl, up, hx, cy)
    rp = region_sbr(rr, up, hx, cy)
    lc = region_sbr(rl, uc, hx, cy)
    rc = region_sbr(rr, uc, hx, cy)

    def region_max_sbr(hside_p, y_pro):
        m = region_mask(hside_p, y_pro, hx, cy)
        return (sm[x[m], y[m], z[m]].max() - bg) / bg

    def region_med_sbr(hside_p, y_pro):
        m = region_mask(hside_p, y_pro, hx, cy)
        return (np.median(sm[x[m], y[m], z[m]]) - bg) / bg

    feat["sbr_left_putamen"] = lp
    feat["sbr_right_putamen"] = rp
    feat["sbr_left_caudate"] = lc
    feat["sbr_right_caudate"] = rc
    feat["sbr_left_putamen_max"] = region_max_sbr(rl, up)
    feat["sbr_right_putamen_max"] = region_max_sbr(rr, up)
    feat["sbr_left_caudate_max"] = region_max_sbr(rl, uc)
    feat["sbr_right_caudate_max"] = region_max_sbr(rr, uc)
    feat["sbr_left_putamen_med"] = region_med_sbr(rl, up)
    feat["sbr_right_putamen_med"] = region_med_sbr(rr, up)
    feat["sbr_left_caudate_med"] = region_med_sbr(rl, uc)
    feat["sbr_right_caudate_med"] = region_med_sbr(rr, uc)

    def region_all(hside_p):
        m = (x > hx) if hside_p else (x <= hx)
        return _sbr(sm[x[m], y[m], z[m]], bg)

    ltot = region_all(rl)
    rtot = region_all(rr)
    feat["sbr_left_total"] = ltot
    feat["sbr_right_total"] = rtot

    f_ltot = ltot + rtot
    feat["frac_left"] = ltot / f_ltot if f_ltot != 0 else 0.5

    ps = (lp + rp) / 2.0
    cs = (lc + rc) / 2.0
    mstr = (ltot + rtot) / 2.0
    feat["frac_putamen"] = ps / mstr if mstr != 0 else 1.0
    feat["frac_caudate"] = cs / mstr if mstr != 0 else 1.0

    feat["min_putamen_sbr"] = min(lp, rp)
    feat["max_putamen_sbr"] = max(lp, rp)
    feat["min_caudate_sbr"] = min(lc, rc)
    feat["max_caudate_sbr"] = max(lc, rc)

    def asym(a, b):
        d = max(abs(a), abs(b))
        return abs(a - b) / d if d != 0 else 0.0

    feat["putamen_asymmetry"] = asym(lp, rp)
    feat["caudate_asymmetry"] = asym(lc, rc)
    feat["overall_asymmetry"] = asym(ltot, rtot)
    feat["left_pc_ratio"] = lp / lc if lc != 0 else 0.0
    feat["right_pc_ratio"] = rp / rc if rc != 0 else 0.0
    feat["min_pc_ratio"] = min(feat["left_pc_ratio"], feat["right_pc_ratio"])

    def ant_post_sbr(hside_p):
        m = region_mask(hside_p, up, hx, cy)
        px = x[m]
        py = y[m]
        pz = z[m]
        med_y = np.median(py)
        ant = py >= med_y
        post = py < med_y
        s_ant = _sbr(sm[px[ant], py[ant], pz[ant]], bg)
        s_post = _sbr(sm[px[post], py[post], pz[post]], bg)
        return s_ant, s_post

    l_ant, l_post = ant_post_sbr(rl)
    r_ant, r_post = ant_post_sbr(rr)
    feat["sbr_left_putamen_ant"] = l_ant
    feat["sbr_left_putamen_post"] = l_post
    feat["sbr_right_putamen_ant"] = r_ant
    feat["sbr_right_putamen_post"] = r_post
    feat["left_ap_ratio"] = l_post / l_ant if l_ant != 0 else 0.0
    feat["right_ap_ratio"] = r_post / r_ant if r_ant != 0 else 0.0
    feat["min_ap_ratio"] = min(feat["left_ap_ratio"], feat["right_ap_ratio"])

    def region_vol(hside_p, y_pro):
        m = region_mask(hside_p, y_pro, hx, cy)
        return int(m.sum())

    v_lp = region_vol(rl, up)
    v_rp = region_vol(rr, up)
    v_lc = region_vol(rl, uc)
    v_rc = region_vol(rr, uc)
    feat["vol_left_putamen"] = v_lp
    feat["vol_right_putamen"] = v_rp
    feat["vol_left_caudate"] = v_lc
    feat["vol_right_caudate"] = v_rc
    feat["vol_total_striatal"] = v_lp + v_rp + v_lc + v_rc

    def region_cv(hside_p, y_pro):
        m = region_mask(hside_p, y_pro, hx, cy)
        v = sm[x[m], y[m], z[m]]
        return v.std() / v.mean()

    feat["cv_left_putamen"] = region_cv(rl, up)
    feat["cv_right_putamen"] = region_cv(rr, up)
    feat["cv_left_caudate"] = region_cv(rl, uc)
    feat["cv_right_caudate"] = region_cv(rr, uc)

    val_p = {q: np.percentile(sv, q) for q in (25, 50, 75, 90, 95, 98, 99)}
    feat["bg_ref"] = bg
    feat["striatal_thresh"] = thr
    for q in (25, 50, 75, 90, 95, 98, 99):
        feat[f"val_p{q}"] = val_p[q]
    feat["val_max"] = float(sv.max())
    feat["val_mean"] = float(sv.mean())
    feat["val_std"] = float(sv.std())

    feat["brain_skew"] = float(stats.skew(sv))
    feat["brain_kurt"] = float(stats.kurtosis(sv))
    feat["contrast_ratio"] = (val_p[99] - bg) / bg
    feat["peak_to_mean_ratio"] = float(sv.max() / sv.mean())
    feat["high_uptake_frac"] = float((sv >= val_p[95]).mean())

    for q in (95, 97, 99):
        thr_q = np.percentile(sv, q)
        coq = np.argwhere(sm >= thr_q)
        xq = coq[:, 0]
        yq = coq[:, 1]
        zq = coq[:, 2]
        cyq = yq.mean()
        for name, hs in (("lp", True), ("rp", False)):
            m = (xq > hx) if hs else (xq <= hx)
            put = m & (yq < cyq)
            cau = m & (yq >= cyq)
            feat[f"sbr_{name}_p{q}"] = _sbr(sm[xq[put], yq[put], zq[put]], bg)
            feat[f"vol_{name}_p{q}"] = int(put.sum())
        for name, hs in (("lc", True), ("rc", False)):
            m = (xq > hx) if hs else (xq <= hx)
            put = m & (yq < cyq)
            cau = m & (yq >= cyq)
            feat[f"sbr_{name}_p{q}"] = _sbr(sm[xq[cau], yq[cau], zq[cau]], bg)
            feat[f"vol_{name}_p{q}"] = int(cau.sum())

    for s in (0.5, 2.0):
        sms = ndi.gaussian_filter(data, sigma=s)
        for name, hs, yp in (("lp", True, False), ("rp", False, False),
                             ("lc", True, True), ("rc", False, True)):
            m = region_mask(hs, yp, hx, cy)
            tag = "s05" if s == 0.5 else "s20"
            feat[f"sbr_{name}_{tag}"] = _sbr(sms[x[m], y[m], z[m]], bg)

    return feat


FEATURE_COLUMNS = [
    "sbr_left_putamen", "sbr_right_putamen", "sbr_left_caudate", "sbr_right_caudate",
    "sbr_left_putamen_max", "sbr_right_putamen_max", "sbr_left_caudate_max", "sbr_right_caudate_max",
    "sbr_left_putamen_med", "sbr_right_putamen_med", "sbr_left_caudate_med", "sbr_right_caudate_med",
    "sbr_left_total", "sbr_right_total", "min_putamen_sbr", "max_putamen_sbr",
    "min_caudate_sbr", "max_caudate_sbr", "putamen_asymmetry", "caudate_asymmetry",
    "overall_asymmetry", "left_pc_ratio", "right_pc_ratio", "min_pc_ratio",
    "sbr_left_putamen_ant", "sbr_left_putamen_post", "sbr_right_putamen_ant", "sbr_right_putamen_post",
    "left_ap_ratio", "right_ap_ratio", "min_ap_ratio",
    "vol_left_putamen", "vol_right_putamen", "vol_left_caudate", "vol_right_caudate", "vol_total_striatal",
    "cv_left_putamen", "cv_right_putamen", "cv_left_caudate", "cv_right_caudate",
    "bg_ref", "striatal_thresh", "val_p25", "val_p50", "val_p75", "val_p90", "val_p95",
    "val_p98", "val_p99", "val_max", "val_mean", "val_std", "brain_skew", "brain_kurt",
    "contrast_ratio", "peak_to_mean_ratio", "high_uptake_frac",
    "sbr_lp_p95", "sbr_rp_p95", "sbr_lc_p95", "sbr_rc_p95",
    "vol_lp_p95", "vol_rp_p95", "vol_lc_p95", "vol_rc_p95",
    "sbr_lp_p97", "sbr_rp_p97", "sbr_lc_p97", "sbr_rc_p97",
    "vol_lp_p97", "vol_rp_p97", "vol_lc_p97", "vol_rc_p97",
    "sbr_lp_p99", "sbr_rp_p99", "sbr_lc_p99", "sbr_rc_p99",
    "vol_lp_p99", "vol_rp_p99", "vol_lc_p99", "vol_rc_p99",
    "sbr_lp_s05", "sbr_rp_s05", "sbr_lc_s05", "sbr_rc_s05",
    "sbr_lp_s20", "sbr_rp_s20", "sbr_lc_s20", "sbr_rc_s20",
    "frac_putamen", "frac_caudate", "frac_left",
]
import nibabel as nib
import numpy as np
import scipy.ndimage as ndi
from scipy import stats

TARGET_SPACING = 2.5


def _reorient_ras(data, affine):
    from nibabel.orientations import apply_orientation, axcodes2ornt, io_orientation, ornt_transform
    current = io_orientation(affine)
    target = axcodes2ornt("RAS")
    transform = ornt_transform(current, target)
    return apply_orientation(data, transform)


def _sbr(values, bg):
    return (values.mean() - bg) / bg


def load_isotropic(nii_path, target=TARGET_SPACING):
    img = nib.load(str(nii_path))
    data = img.get_fdata().astype(np.float32)
    data = _reorient_ras(data, img.affine)
    zooms = np.array(img.header.get_zooms()[:3], dtype=np.float64)
    factor = zooms / target
    new_shape = tuple(max(1, int(round(s * f))) for s, f in zip(data.shape, factor))
    zf = [n / s for s, n in zip(data.shape, new_shape)]
    data = ndi.zoom(data, zoom=zf, order=1)
    return data, new_shape


def extract_physical_features(nii_path, target=TARGET_SPACING):
    sm, shape = load_isotropic(nii_path, target)
    volume_mm3 = target ** 3

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

    def mask(hp, yp, cx=None, cyy=None):
        left = (x > cx) if hp else (x <= cx)
        return left & (y < cyy if not yp else y >= cyy)

    def rsbr(hp, yp):
        m = mask(hp, yp, hx, cy)
        return _sbr(sm[x[m], y[m], z[m]], bg)

    def rvol(hp, yp):
        m = mask(hp, yp, hx, cy)
        return int(m.sum())

    lp, rp = rsbr(True, False), rsbr(False, False)
    lc, rc = rsbr(True, True), rsbr(False, True)
    ltot_ = (x > hx)
    ltot = _sbr(sm[x[ltot_], y[ltot_], z[ltot_]], bg)
    rtot = _sbr(sm[x[~ltot_], y[~ltot_], z[~ltot_]], bg)

    f = {}
    f["sbr_left_putamen"] = lp
    f["sbr_right_putamen"] = rp
    f["sbr_left_caudate"] = lc
    f["sbr_right_caudate"] = rc

    def rmax(hp, yp):
        m = mask(hp, yp, hx, cy)
        return (sm[x[m], y[m], z[m]].max() - bg) / bg

    def rmed(hp, yp):
        m = mask(hp, yp, hx, cy)
        return (np.median(sm[x[m], y[m], z[m]]) - bg) / bg

    f["sbr_left_putamen_max"] = rmax(True, False)
    f["sbr_right_putamen_max"] = rmax(False, False)
    f["sbr_left_caudate_max"] = rmax(True, True)
    f["sbr_right_caudate_max"] = rmax(False, True)
    f["sbr_left_putamen_med"] = rmed(True, False)
    f["sbr_right_putamen_med"] = rmed(False, False)
    f["sbr_left_caudate_med"] = rmed(True, True)
    f["sbr_right_caudate_med"] = rmed(False, True)
    f["sbr_left_total"] = ltot
    f["sbr_right_total"] = rtot
    f["frac_left"] = ltot / (ltot + rtot)

    ps = (lp + rp) / 2.0
    cs = (lc + rc) / 2.0
    mstr = (ltot + rtot) / 2.0
    f["frac_putamen"] = ps / mstr
    f["frac_caudate"] = cs / mstr
    f["min_putamen_sbr"] = min(lp, rp)
    f["max_putamen_sbr"] = max(lp, rp)
    f["min_caudate_sbr"] = min(lc, rc)
    f["max_caudate_sbr"] = max(lc, rc)

    def asym(a, b):
        d = max(abs(a), abs(b))
        return abs(a - b) / d if d != 0 else 0.0

    f["putamen_asymmetry"] = asym(lp, rp)
    f["caudate_asymmetry"] = asym(lc, rc)
    f["overall_asymmetry"] = asym(ltot, rtot)
    f["left_pc_ratio"] = lp / lc
    f["right_pc_ratio"] = rp / rc
    f["min_pc_ratio"] = min(f["left_pc_ratio"], f["right_pc_ratio"])

    def ant_post(hp):
        m = mask(hp, False, hx, cy)
        px, py, pz = x[m], y[m], z[m]
        med_y = np.median(py)
        ant = py >= med_y
        post = py < med_y
        return (_sbr(sm[px[ant], py[ant], pz[ant]], bg),
                _sbr(sm[px[post], py[post], pz[post]], bg))

    la, lpst = ant_post(True)
    ra, rpst = ant_post(False)
    f["sbr_left_putamen_ant"] = la
    f["sbr_left_putamen_post"] = lpst
    f["sbr_right_putamen_ant"] = ra
    f["sbr_right_putamen_post"] = rpst
    f["left_ap_ratio"] = lpst / la
    f["right_ap_ratio"] = rpst / ra
    f["min_ap_ratio"] = min(f["left_ap_ratio"], f["right_ap_ratio"])

    v = {"vol_left_putamen": rvol(True, False), "vol_right_putamen": rvol(False, False),
         "vol_left_caudate": rvol(True, True), "vol_right_caudate": rvol(False, True)}
    v["vol_total_striatal"] = sum(v.values())
    for k in v:
        f[k] = v[k]
    for k in ("vol_left_putamen", "vol_right_putamen", "vol_left_caudate", "vol_right_caudate", "vol_total_striatal"):
        f[k + "_mm3"] = f[k] * volume_mm3
    f["vol_putamen_mm3"] = (v["vol_left_putamen"] + v["vol_right_putamen"]) * volume_mm3
    f["vol_caudate_mm3"] = (v["vol_left_caudate"] + v["vol_right_caudate"]) * volume_mm3

    def rcv(hp, yp):
        m = mask(hp, yp, hx, cy)
        vv = sm[x[m], y[m], z[m]]
        return vv.std() / vv.mean()

    f["cv_left_putamen"] = rcv(True, False)
    f["cv_right_putamen"] = rcv(False, False)
    f["cv_left_caudate"] = rcv(True, True)
    f["cv_right_caudate"] = rcv(False, True)

    val_p = {q: np.percentile(sv, q) for q in (25, 50, 75, 90, 95, 98, 99)}
    f["bg_ref"] = bg
    f["striatal_thresh"] = thr
    for q in (25, 50, 75, 90, 95, 98, 99):
        f[f"val_p{q}"] = val_p[q]
    f["val_max"] = float(sv.max())
    f["val_mean"] = float(sv.mean())
    f["val_std"] = float(sv.std())
    f["brain_skew"] = float(stats.skew(sv))
    f["brain_kurt"] = float(stats.kurtosis(sv))
    f["contrast_ratio"] = (val_p[99] - bg) / bg
    f["peak_to_mean_ratio"] = float(sv.max() / sv.mean())
    f["high_uptake_frac"] = float((sv >= val_p[95]).mean())

    for q in (95, 97, 99):
        thr_q = np.percentile(sv, q)
        coq = np.argwhere(sm >= thr_q)
        xq, yq, zq = coq[:, 0], coq[:, 1], coq[:, 2]
        cyq = yq.mean()
        for name, hs in (("lp", True), ("rp", False), ("lc", True), ("rc", False)):
            is_put = name in ("lp", "rp")
            m = (xq > hx) if hs else (xq <= hx)
            sel = m & (yq < cyq if is_put else yq >= cyq)
            f[f"sbr_{name}_p{q}"] = _sbr(sm[xq[sel], yq[sel], zq[sel]], bg)
            f[f"vol_{name}_p{q}"] = int(sel.sum())

    for s in (0.5, 2.0):
        sms = ndi.gaussian_filter(sm, sigma=s)
        tag = "s05" if s == 0.5 else "s20"
        for name, hp, yp in (("lp", True, False), ("rp", False, False),
                             ("lc", True, True), ("rc", False, True)):
            m = mask(hp, yp, hx, cy)
            f[f"sbr_{name}_{tag}"] = _sbr(sms[x[m], y[m], z[m]], bg)

    pc = np.cov(np.column_stack([x, y, z]).T)
    ev = np.linalg.eigvalsh(pc)
    ev = np.sort(ev)[::-1]
    f["shape_elongation"] = float(np.sqrt(ev[0] / ev[2])) if ev[2] > 0 else 0.0
    f["shape_compactness"] = float((x.size ** (2.0 / 3.0)) / (4 * np.pi / 3.0 * np.prod(np.sqrt(ev))))
    f["shape_components"] = int(ndi.label(sm >= thr)[1])
    f["striatal_cx"] = float(x.mean())
    f["striatal_cy"] = float(y.mean())
    f["striatal_cz"] = float(z.mean())
    f["head_cx"] = float(hx)
    f["brain_volume_mm3"] = float(head.sum() * volume_mm3)

    # posterior putamen focal asymmetry (PD hallmark)
    post_asym = asym(f["sbr_left_putamen_post"], f["sbr_right_putamen_post"])
    f["post_putamen_asymmetry"] = post_asym
    f["min_post_putamen_sbr"] = min(f["sbr_left_putamen_post"], f["sbr_right_putamen_post"])

    return f


PHYS_FEATURE_COLUMNS = [
    "sbr_left_putamen", "sbr_right_putamen", "sbr_left_caudate", "sbr_right_caudate",
    "sbr_left_putamen_max", "sbr_right_putamen_max", "sbr_left_caudate_max", "sbr_right_caudate_max",
    "sbr_left_putamen_med", "sbr_right_putamen_med", "sbr_left_caudate_med", "sbr_right_caudate_med",
    "sbr_left_total", "sbr_right_total", "min_putamen_sbr", "max_putamen_sbr",
    "min_caudate_sbr", "max_caudate_sbr", "putamen_asymmetry", "caudate_asymmetry",
    "overall_asymmetry", "left_pc_ratio", "right_pc_ratio", "min_pc_ratio",
    "sbr_left_putamen_ant", "sbr_left_putamen_post", "sbr_right_putamen_ant", "sbr_right_putamen_post",
    "left_ap_ratio", "right_ap_ratio", "min_ap_ratio",
    "vol_left_putamen", "vol_right_putamen", "vol_left_caudate", "vol_right_caudate", "vol_total_striatal",
    "vol_left_putamen_mm3", "vol_right_putamen_mm3", "vol_left_caudate_mm3", "vol_right_caudate_mm3",
    "vol_total_striatal_mm3", "vol_putamen_mm3", "vol_caudate_mm3",
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
    "shape_elongation", "shape_compactness", "shape_components",
    "striatal_cx", "striatal_cy", "striatal_cz", "head_cx",
    "brain_volume_mm3", "post_putamen_asymmetry", "min_post_putamen_sbr",
]
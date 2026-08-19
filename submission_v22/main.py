"""
V22 submission main.py: Mega ensemble (global 12-stream + per-site + MLP).
Uses sbr + phys features only (the two with extractors).
"""
import json, os, sys
import joblib
import numpy as np
import pandas as pd
from sbr_extractor import FEATURE_COLUMNS, extract_features
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS, extract_physical_features

PVE_REF = 15.625
BIO_FEATURES = [
    "lr_put_asym", "lr_put_ratio", "lr_put_sum", "lr_put_diff",
    "lr_cau_asym", "lr_cau_ratio", "lr_cau_sum",
    "lp_post_ant", "rp_post_ant", "lp_cr", "rp_cr",
    "total_str", "lr_total_asym",
    "striatal_mean", "striatal_std", "striatal_cv", "striatal_range",
    "lp_x_rp", "lc_x_rc",
]
RAW1 = list(FEATURE_COLUMNS) + BIO_FEATURES
RAW2 = list(PHYS_FEATURE_COLUMNS)


def _voxel_vol(path):
    import nibabel as nib
    z = nib.load(str(path)).header.get_zooms()[:3]
    return float(z[0] * z[1] * z[2])


def _pve_factor(voxel_vol):
    return (PVE_REF / voxel_vol) ** (1.0 / 3.0)


def infer_site(voxel_vol):
    vs = voxel_vol ** (1.0 / 3.0)
    if vs < 1.8:
        return "1.5x1.5x1.5"
    elif vs < 2.0:
        return "1.5x1.5x3.0"
    elif vs < 3.0:
        return "2.5x2.5x2.5"
    elif vs < 3.8:
        return "3.5x3.5x3.5"
    else:
        return "4.0x4.0x4.0"


def build_matrix(raw, raw_cols):
    a = np.array([raw.get(c, np.nan) for c in raw_cols], dtype=np.float64).reshape(1, -1)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def add_bio_inference(raw_dict):
    eps = 1e-6
    lp = raw_dict.get("sbr_left_putamen", 0)
    rp = raw_dict.get("sbr_right_putamen", 0)
    lc = raw_dict.get("sbr_left_caudate", 0)
    rc = raw_dict.get("sbr_right_caudate", 0)
    lt = raw_dict.get("sbr_left_total", 0)
    rt = raw_dict.get("sbr_right_total", 0)
    lpa = raw_dict.get("sbr_left_putamen_ant", 0)
    lpp = raw_dict.get("sbr_left_putamen_post", 0)
    rpa = raw_dict.get("sbr_right_putamen_ant", 0)
    rpp = raw_dict.get("sbr_right_putamen_post", 0)
    raw_dict["lr_put_asym"] = (lp - rp) / (lp + rp + eps)
    raw_dict["lr_put_ratio"] = min(lp, rp) / (max(lp, rp) + eps)
    raw_dict["lr_put_sum"] = lp + rp
    raw_dict["lr_put_diff"] = lp - rp
    raw_dict["lr_cau_asym"] = (lc - rc) / (lc + rc + eps)
    raw_dict["lr_cau_ratio"] = min(lc, rc) / (max(lc, rc) + eps)
    raw_dict["lr_cau_sum"] = lc + rc
    raw_dict["lp_post_ant"] = lpp / (lpa + eps)
    raw_dict["rp_post_ant"] = rpp / (rpa + eps)
    raw_dict["lp_cr"] = lp / (lc + eps)
    raw_dict["rp_cr"] = rp / (rc + eps)
    raw_dict["total_str"] = lt + rt
    raw_dict["lr_total_asym"] = (lt - rt) / (lt + rt + eps)
    vals = [lp, rp, lc, rc]
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    raw_dict["striatal_mean"] = mean_v
    raw_dict["striatal_std"] = std_v
    raw_dict["striatal_cv"] = std_v / (mean_v + eps)
    raw_dict["striatal_range"] = max(vals) - min(vals)
    raw_dict["lp_x_rp"] = lp * rp
    raw_dict["lc_x_rc"] = lc * rc
    return raw_dict


def logit(x):
    return np.log(np.clip(x, 1e-7, 1 - 1e-7) / (1.0 - np.clip(x, 1e-7, 1 - 1e-7)))


def expit(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def predict_mlp(model_path, X_tab):
    import torch
    import torch.nn as nn
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    arch = ckpt["arch"]
    d = ckpt["input_dim"]

    class MLP(nn.Module):
        def __init__(self, d):
            super().__init__()
            if arch == "256_128":
                self.net = nn.Sequential(
                    nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, 1))
            elif arch == "128_64_32":
                self.net = nn.Sequential(
                    nn.Linear(d, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(64, 32), nn.BatchNorm1d(32), nn.ReLU(), nn.Dropout(0.1),
                    nn.Linear(32, 1))
            else:
                self.net = nn.Sequential(
                    nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
                    nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.2),
                    nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(),
                    nn.Linear(64, 1))
        def forward(self, x):
            return self.net(x)

    model = MLP(d)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    scaler_mean = ckpt["scaler_mean"]
    scaler_scale = ckpt["scaler_scale"]
    X_scaled = (X_tab - scaler_mean) / (scaler_scale + 1e-8)
    with torch.no_grad():
        pred = torch.sigmoid(model(torch.FloatTensor(X_scaled))).numpy().ravel()
    return pred


def main():
    data_dir = os.path.join(os.getcwd(), "data")
    nifti_dir = os.path.join(data_dir, "niftis")
    fmt = pd.read_csv(os.path.join(data_dir, "submission_format.csv"))
    cfg = json.load(open("model_config.json"))
    weights_dir = "weights"

    order = fmt["uid"].tolist()
    n_total = len(order)
    probas = []

    for idx, uid in enumerate(order):
        try:
            path = os.path.join(nifti_dir, f"{uid}.nii.gz")
            vvol = _voxel_vol(path)
            fac = _pve_factor(vvol)
            site_str = infer_site(vvol)

            f1 = extract_features(path)
            for k in list(f1.keys()):
                if k.startswith("sbr_"):
                    f1[k] = f1[k] * fac
            f2 = extract_physical_features(path)
            f1 = add_bio_inference(f1)
            X1 = build_matrix(f1, RAW1)
            X2 = build_matrix(f2, RAW2)
            X_tab = np.hstack([X1, X2])

            # --- Global 12-stream ---
            global_scores = []
            mnames = ["lr", "xgb", "lgb", "cb", "et", "ridge"]
            for seed_idx in range(cfg["n_seeds"]):
                for xi in range(2):
                    X = [X1, X2][xi]
                    for mk in mnames:
                        model = joblib.load(os.path.join(weights_dir, f"s{seed_idx}_ds{xi}_{mk}.pkl"))
                        scaler = joblib.load(os.path.join(weights_dir, f"s{seed_idx}_ds{xi}_scaler.pkl"))
                        if mk == "lr" or mk == "ridge":
                            inp = scaler.transform(X)
                        else:
                            mi_cols = joblib.load(os.path.join(weights_dir, f"s{seed_idx}_ds{xi}_mi.pkl"))
                            inp = scaler.transform(X)[:, mi_cols]
                        if mk == "ridge":
                            s = expit(model.decision_function(inp))
                        else:
                            s = model.predict_proba(inp)[:, 1]
                        global_scores.append(s)

            w_g = np.array(cfg["global_blend_weights"])
            blend_global = sum(w * s for w, s in zip(w_g, global_scores))
            blend_global = expit(logit(np.clip(blend_global, 1e-6, 1-1e-6)) * cfg["platt_a_global"] + cfg["platt_b_global"])

            # --- Per-site ---
            if site_str in cfg["persite_models"]:
                site_safe = site_str.replace(".", "_")
                ps_scaler = joblib.load(os.path.join(weights_dir, f"ps_{site_safe}_scaler.pkl"))
                ps_mi = joblib.load(os.path.join(weights_dir, f"ps_{site_safe}_mi.pkl"))
                Xs_ps = ps_scaler.transform(X_tab)
                Xt_ps = Xs_ps[:, ps_mi]
                ps_lr = joblib.load(os.path.join(weights_dir, f"ps_{site_safe}_lr.pkl"))
                ps_xgb = joblib.load(os.path.join(weights_dir, f"ps_{site_safe}_xgb.pkl"))
                ps_lgb = joblib.load(os.path.join(weights_dir, f"ps_{site_safe}_lgb.pkl"))
                w_ps = np.array(cfg["persite_blend_weights"])
                ps_scores = [
                    ps_lr.predict_proba(Xs_ps)[:, 1],
                    ps_xgb.predict_proba(Xt_ps)[:, 1],
                    ps_lgb.predict_proba(Xt_ps)[:, 1],
                ]
                blend_ps = sum(w * s for w, s in zip(w_ps, ps_scores))
                blend_ps = expit(logit(np.clip(blend_ps, 1e-6, 1-1e-6)) * cfg["platt_a_persite"] + cfg["platt_b_persite"])
            else:
                blend_ps = blend_global

            # --- MLP ---
            mlp_scores = []
            for seed_idx, seed_val in enumerate([42, 777, 2024, 12345, 999]):
                for arch in ["256_128", "128_64_32"]:
                    mlp_path = os.path.join(weights_dir, f"mlp_{arch}_s{seed_val}.pt")
                    if os.path.exists(mlp_path):
                        mlp_scores.append(predict_mlp(mlp_path, X_tab))
            if mlp_scores:
                blend_mlp = float(np.mean(mlp_scores))
                blend_mlp = expit(logit(np.clip(blend_mlp, 1e-6, 1-1e-6)) * cfg["platt_a_mlp"] + cfg["platt_b_mlp"])
            else:
                blend_mlp = blend_global

            # --- Mega blend ---
            w_mega = np.array(cfg["mega_blend_weights"])
            mega_raw = w_mega[0] * blend_global + w_mega[1] * blend_ps + (1 - w_mega[0] - w_mega[1]) * blend_mlp

            # --- Site calibration ---
            if site_str in cfg["site_cal_params"]:
                sa, sb = cfg["site_cal_params"][site_str]
                mega = expit(sa * logit(np.clip(mega_raw, 1e-6, 1-1e-6)) + sb)
            else:
                mega = expit(logit(np.clip(mega_raw, 1e-6, 1-1e-6)) * cfg["platt_a_mega"] + cfg["platt_b_mega"])

            mega = min(max(mega, cfg["clip_min"]), cfg["clip_max"])
            probas.append(float(mega))
        except Exception as e:
            print(f"  ERROR {uid}: {e}, using 0.5")
            probas.append(0.5)

        if (idx + 1) % 10 == 0 or (idx + 1) == n_total:
            print(f"[progress] {idx + 1}/{n_total}")

    pd.DataFrame({"uid": order, "is_pathologic": probas}).to_csv("submission.csv", index=False)
    print("submission written")


if __name__ == "__main__":
    main()

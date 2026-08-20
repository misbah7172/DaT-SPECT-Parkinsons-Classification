"""V23 submission: deep6 CNN stream + sbr tabular stream, fixed blend + temp.
Reproduces the v26 sbr feature pipeline (PVE-corrected sbr + bio + log + scaler + MI)
and the deep-CNN inference pipeline (load_aligned + NCC reg + crop + inorm + deep3D net).
"""
import json
import os
import pickle
import sys

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sbr_extractor import FEATURE_COLUMNS, extract_features
from cnn_infer import DeepEnsemble, TemplateCache, load_aligned

PVE_REF = 15.625
W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

with open(os.path.join(W, "ship_final.json")) as f:
    FC = json.load(f)
W_DEEP = FC["w_deep6"]
W_SBR = FC["w_sbr"]
TEMP = FC["T"]


def _voxel_vol(path):
    import nibabel as nib
    z = nib.load(str(path)).header.get_zooms()[:3]
    return float(z[0] * z[1] * z[2])


def add_bio(df):
    eps = 1e-6
    L, R = df["sbr_left_putamen"].values.astype(float), df["sbr_right_putamen"].values.astype(float)
    df["lr_put_asym"] = (L - R) / (L + R + eps)
    df["lr_put_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
    df["lr_put_sum"] = L + R
    df["lr_put_diff"] = L - R
    L, R = df["sbr_left_caudate"].values.astype(float), df["sbr_right_caudate"].values.astype(float)
    df["lr_cau_asym"] = (L - R) / (L + R + eps)
    df["lr_cau_ratio"] = np.minimum(L, R) / (np.maximum(L, R) + eps)
    df["lr_cau_sum"] = L + R
    df["lp_post_ant"] = df["sbr_left_putamen_post"].values.astype(float) / (df["sbr_left_putamen_ant"].values.astype(float) + eps)
    df["rp_post_ant"] = df["sbr_right_putamen_post"].values.astype(float) / (df["sbr_right_putamen_ant"].values.astype(float) + eps)
    df["lp_cr"] = df["sbr_left_putamen"].values.astype(float) / (df["sbr_left_caudate"].values.astype(float) + eps)
    df["rp_cr"] = df["sbr_right_putamen"].values.astype(float) / (df["sbr_right_caudate"].values.astype(float) + eps)
    df["total_str"] = df["sbr_left_total"].values.astype(float) + df["sbr_right_total"].values.astype(float)
    L, R = df["sbr_left_total"].values.astype(float), df["sbr_right_total"].values.astype(float)
    df["lr_total_asym"] = (L - R) / (L + R + eps)
    return df


def build_matrix(raw):
    cols = [c for c in raw.columns]
    a = raw[cols].values.astype(np.float64)
    return np.nan_to_num(np.concatenate([a, np.log1p(np.clip(a, 0, None))], axis=1),
                         nan=0.0, posinf=0.0, neginf=0.0)


def sbr_stream_prob(extracted, vvol):
    fac = (PVE_REF / vvol) ** (1.0 / 3.0)
    d = {}
    for c in FEATURE_COLUMNS:
        v = extracted.get(c, np.nan)
        if c.startswith("sbr_") and np.isfinite(v):
            v = v * fac
        d[c] = v
    df = pd.DataFrame([d])
    df = add_bio(df)
    COLS = json.load(open(os.path.join(W, "sbr_full_cols.json")))
    df = df[[c for c in COLS if c in df.columns]]
    X = build_matrix(df)
    sc = joblib.load(os.path.join(W, "sbr_full_scaler.pkl"))
    sel = joblib.load(os.path.join(W, "sbr_full_sel.pkl"))
    Xs = sc.transform(X)
    Xt = Xs[:, sel]
    scores = []
    for name in ["lr", "xgb", "lgb", "cb", "et", "ridge"]:
        m = joblib.load(os.path.join(W, f"sbr_full_{name}.pkl"))
        if name in ("lr", "ridge"):
            # NOTE: lr/ridge were fit on FULL scaled X in v47
            scores.append(m.predict_proba(Xs)[:, 1][0] if name == "lr"
                          else 1.0 / (1.0 + np.exp(-m.decision_function(Xs)[0])))
        else:
            scores.append(m.predict_proba(Xt)[:, 1][0])
    return float(np.mean(scores))


def main():
    PACKAGE = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.getcwd(), "data")
    nifti_dir = os.path.join(data_dir, "niftis")
    fmt = pd.read_csv(os.path.join(data_dir, "submission_format.csv"))
    _tpl_path = os.path.join(data_dir, "atlas_template.npy")
    if not os.path.exists(_tpl_path):
        _tpl_path = os.path.join(PACKAGE, "atlas_template.npy")
    template = np.load(_tpl_path, allow_pickle=True)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    allow = {"deep_r_42.pt", "deep_r_777.pt", "deep_r_2024.pt", "deep_r_100.pt",
             "deep_big_2025.pt", "deep_big_1984.pt"}
    ens = DeepEnsemble(W, device=device, allow=allow)
    cache = TemplateCache(template)

    order = fmt["uid"].tolist()
    probas = []
    for idx, uid in enumerate(order, 1):
        try:
            path = os.path.join(nifti_dir, f"{uid}.nii.gz")
            vvol = _voxel_vol(path)
            p_sbr = sbr_stream_prob(extract_features(path), vvol)

            crop = None
            try:
                crop = __import__("cnn_infer", fromlist=["to_model_input"]).to_model_input(path, template, cache)
                p_deep = ens.predict_one(crop)
            except Exception:
                p_deep = 0.5

            blend = W_DEEP * p_deep + W_SBR * p_sbr
            blend = np.clip(blend, 1e-6, 1 - 1e-6)
            lg = np.log(blend / (1.0 - blend))
            p = 1.0 / (1.0 + np.exp(-lg / TEMP))
            p = float(min(max(p, 0.005), 0.995))
            probas.append(p)
        except Exception as e:
            print(f"  ERROR {uid}: {e}, using 0.5")
            probas.append(0.5)

        if idx % 10 == 0 or idx == len(order):
            print(f"[progress] {idx}/{len(order)}")

    pd.DataFrame({"uid": order, "is_pathologic": probas}).to_csv("submission.csv", index=False)
    print("submission written")


if __name__ == "__main__":
    main()
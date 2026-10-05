"""V25 submission: CNN-only inference with Platt calibration.

V25 improvements over v24:
1. Deeper ResNet: 3 residual blocks + 3 downsampling stages
2. Net3dR-v25: 16->32->64->128 (845K params)
3. Net3dBig-v25: 20->40->80->160 (1.3M params)
4. 10 models: 5 Net3dR + 5 Net3dBig (different seeds)
5. Input: full 80^3 grid at 2.0mm isotropic
6. Mixup + Label Smoothing + Cosine Annealing during training
7. Platt scaling (a=1.45, b=0.20) improves LL from 0.2618 to 0.2453
8. 15 TTA views per model
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_infer import DeepEnsemble, TemplateCache, to_model_input

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

with open(os.path.join(W, "calibration_v25.json")) as f:
    CAL = json.load(f)

PLATT_A = CAL["a"]
PLATT_B = CAL["b"]


def calibrate(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    cal = 1.0 / (1.0 + np.exp(-(PLATT_A * logit + PLATT_B)))
    return float(np.clip(cal, 0.005, 0.995))


def get_crop_fast(uid, cache_dir, nifti_dir, template, cache):
    cached_path = os.path.join(cache_dir, f"{uid}.npy")
    if os.path.exists(cached_path):
        w = np.load(cached_path).astype(np.float32)
        v = w[w > 0]
        if v.size:
            lo, hi = np.percentile(v, 1), np.percentile(v, 99)
            if hi > lo:
                w = (w - lo) / (hi - lo)
        return np.clip(w, 0, 1).astype(np.float32)
    
    path = os.path.join(nifti_dir, f"{uid}.nii.gz")
    if not os.path.exists(path):
        path = os.path.join(nifti_dir, f"{uid}.nii")
    return to_model_input(path, template, cache)


def main():
    PACKAGE = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.getcwd(), "data")
    if not os.path.exists(data_dir) and os.path.exists(os.path.join(os.getcwd(), "Dataset")):
        data_dir = os.path.join(os.getcwd(), "Dataset")

    cache_dir = "D:/DaT_cache/volumes_2mm"

    nifti_candidates = [
        os.path.join(data_dir, "niftis"),
        os.path.join(data_dir, "DaT_Parkinsons_Challenge_-_niftis.zip"),
        os.path.join(data_dir, "DaT_Parkinsons_Challenge_-_smoke_test_data.tar.gz", "niftis"),
    ]
    nifti_dir = nifti_candidates[0]
    for cand in nifti_candidates:
        if os.path.exists(cand) and any(f.endswith(".nii.gz") or f.endswith(".nii") for f in os.listdir(cand)):
            nifti_dir = cand
            break

    fmt_path = os.path.join(data_dir, "submission_format.csv")
    if not os.path.exists(fmt_path) and os.path.exists(os.path.join(data_dir, "train_labels.csv")):
        fmt_path = os.path.join(data_dir, "train_labels.csv")

    if os.path.exists(fmt_path):
        fmt = pd.read_csv(fmt_path)
        order = fmt["uid"].tolist()
    elif os.path.exists(nifti_dir):
        nii_files = [f.replace(".nii.gz", "").replace(".nii", "") for f in os.listdir(nifti_dir) if f.endswith(".nii.gz") or f.endswith(".nii")]
        order = sorted(list(set(nii_files)))
    else:
        raise FileNotFoundError(f"Could not locate scan files or labels in {data_dir}")

    _tpl_path = os.path.join(data_dir, "atlas_template.npy")
    if not os.path.exists(_tpl_path):
        _tpl_path = os.path.join(PACKAGE, "atlas_template.npy")
    template = np.load(_tpl_path, allow_pickle=True)

    device = "cuda" if __import__("torch").cuda.is_available() else "cpu"
    print(f"Using device: {device} | Scans to process: {len(order)}", flush=True)

    allow = {
        "v25_r_42.pt", "v25_r_777.pt", "v25_r_2024.pt", "v25_r_1984.pt", "v25_r_100.pt",
        "v25_big_2025.pt", "v25_big_314.pt", "v25_big_271.pt", "v25_big_1337.pt", "v25_big_999.pt"
    }
    ens = DeepEnsemble(W, device=device, allow=allow)
    cache = TemplateCache(template)

    probas = []
    for idx, uid in enumerate(order, 1):
        try:
            crop = get_crop_fast(uid, cache_dir, nifti_dir, template, cache)
            p_raw = ens.predict_one(crop)
            p = calibrate(p_raw)
            probas.append(p)
        except Exception as e:
            print(f"  ERROR {uid}: {e}, using 0.5", flush=True)
            probas.append(0.5)

        if idx % 50 == 0 or idx == len(order):
            print(f"[progress] {idx}/{len(order)}", flush=True)

    pd.DataFrame({"uid": order, "is_pathologic": probas}).to_csv("submission.csv", index=False)
    print("submission written successfully!", flush=True)


if __name__ == "__main__":
    main()

"""V24 submission: CNN-only inference with Platt calibration.

Changes from v23:
1. Input: 2.0mm isotropic, 32x40x48 crop from 80^3 grid
2. 6 CNN models (3 Net3dR + 3 Net3dBig), weights averaged across 5 folds
3. 15 TTA views per model (translation + rotation)
4. Rotation alignment before translation matching
5. Platt scaling (near-identity, models well-calibrated)
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cnn_infer import DeepEnsemble, TemplateCache, to_model_input

W = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")

with open(os.path.join(W, "calibration_v24.json")) as f:
    CAL = json.load(f)

PLATT_A = CAL["platt_a"]
PLATT_B = CAL["platt_b"]


def calibrate(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    logit = np.log(p / (1 - p))
    cal = 1.0 / (1.0 + np.exp(-(PLATT_A * logit + PLATT_B)))
    return float(np.clip(cal, 0.005, 0.995))


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

    allow = {"deep_r_42.pt", "deep_r_777.pt", "deep_r_2024.pt",
             "deep_big_2025.pt", "deep_big_1984.pt", "deep_big_100.pt"}
    ens = DeepEnsemble(W, device=device, allow=allow)
    cache = TemplateCache(template)

    order = fmt["uid"].tolist()
    probas = []
    for idx, uid in enumerate(order, 1):
        try:
            path = os.path.join(nifti_dir, f"{uid}.nii.gz")
            crop = to_model_input(path, template, cache)
            p_raw = ens.predict_one(crop)
            p = calibrate(p_raw)
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

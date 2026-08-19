import json
import os
import zipfile

import joblib
import numpy as np
import pandas as pd

from sbr_extractor import FEATURE_COLUMNS as SBR_COLUMNS, extract_features
from sbr_extractor_phys import PHYS_FEATURE_COLUMNS as PHYS_COLUMNS, extract_physical_features
from sbr_extractor_atlas import ATLAS_FEATURE_COLUMNS as ATLAS_COLUMNS, extract_atlas_features

VIEW_COLUMNS = {
    "sbr": list(SBR_COLUMNS),
    "phys": list(PHYS_COLUMNS),
    "atlas": list(ATLAS_COLUMNS),
}


def augment(x):
    return np.concatenate([x, np.log1p(np.clip(x, 0, None))], axis=1)


def feature_matrix(feat_dict, columns):
    values = np.array([feat_dict.get(col, np.nan) for col in columns], dtype=np.float64).reshape(1, -1)
    return np.nan_to_num(augment(values), nan=0.0, posinf=0.0, neginf=0.0)


def load_atlas_assets(data_dir):
    rois = np.load(os.path.join(data_dir, "atlas_rois.npy"), allow_pickle=True)
    template = np.load(os.path.join(data_dir, "atlas_template.npy"), allow_pickle=True)
    if isinstance(rois, np.ndarray) and rois.ndim == 4 and rois.shape[0] == 4:
        rois = {"lp": rois[0], "rp": rois[1], "lc": rois[2], "rc": rois[3]}
    elif isinstance(rois, np.ndarray) and rois.shape == ():
        rois = rois.item()
    return rois, template


def resolve_nifti_path(data_dir, uid):
    direct = os.path.join(data_dir, "niftis", f"{uid}.nii.gz")
    if os.path.exists(direct):
        return direct

    for root, _, files in os.walk(data_dir):
        candidate = os.path.join(root, f"{uid}.nii.gz")
        if os.path.exists(candidate):
            return candidate

    zip_path = os.path.join(data_dir, "DaT_Parkinsons_Challenge_-_niftis.zip")
    if os.path.exists(zip_path):
        cache_dir = os.path.join(data_dir, "_nifti_cache")
        os.makedirs(cache_dir, exist_ok=True)
        cached = os.path.join(cache_dir, f"{uid}.nii.gz")
        if os.path.exists(cached):
            return cached
        with zipfile.ZipFile(zip_path) as zf:
            member = next((name for name in zf.namelist() if name.endswith(f"{uid}.nii.gz")), None)
            if member is None:
                raise FileNotFoundError(uid)
            with zf.open(member) as src, open(cached, "wb") as dst:
                dst.write(src.read())
        return cached

    raise FileNotFoundError(uid)


def extract_view(data_dir, uid, view_name, atlas_assets):
    path = resolve_nifti_path(data_dir, uid)
    if view_name == "sbr":
        return extract_features(path)
    if view_name == "phys":
        return extract_physical_features(path)
    rois, template = atlas_assets
    return extract_atlas_features(path, rois, template)


def logit_clip(p):
    return np.log(p / (1.0 - p))


def expit_fn(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    data_dir = os.path.join(os.getcwd(), "data")
    fmt = pd.read_csv(os.path.join(data_dir, "submission_format.csv"))
    cfg = json.load(open("model_config.json", "r", encoding="utf-8"))
    atlas_assets = load_atlas_assets(data_dir)
    weights_dir = "weights"

    order = fmt["uid"].tolist()
    probas = []
    for idx, uid in enumerate(order, 1):
        stream_values = []
        for view_name in cfg["views"]:
            feat = extract_view(data_dir, uid, view_name, atlas_assets)
            x = feature_matrix(feat, VIEW_COLUMNS[view_name])
            for model_name in cfg["model_names"]:
                per_seed_preds = []
                for seed_idx, _seed in enumerate(cfg["seeds"]):
                    scaler = joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_scaler.pkl"))
                    sel = joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_sel.pkl"))
                    xs = scaler.transform(x)
                    xt = xs[:, sel]
                    if model_name == "lr":
                        per_seed_preds.append(joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_lr.pkl")).predict_proba(xs)[:, 1][0])
                    elif model_name == "et":
                        per_seed_preds.append(joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_et.pkl")).predict_proba(xt)[:, 1][0])
                    elif model_name == "hgb":
                        per_seed_preds.append(joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_hgb.pkl")).predict_proba(xt)[:, 1][0])
                    else:
                        per_seed_preds.append(joblib.load(os.path.join(weights_dir, f"s{seed_idx}_{view_name}_rf.pkl")).predict_proba(xt)[:, 1][0])
                stream_values.append(float(np.mean(per_seed_preds)))

        blend = float(np.dot(cfg["blend_weights"], np.array(stream_values, dtype=np.float64)))
        p = expit_fn(logit_clip(np.clip(blend, 1e-6, 1 - 1e-6)) / cfg["temperature"])
        p = float(np.clip(p, cfg["clip_min"], cfg["clip_max"]))
        probas.append(p)
        if idx % 10 == 0 or idx == len(order):
            print(f"[progress] {idx}/{len(order)}")

    pd.DataFrame({"uid": order, "is_pathologic": probas}).to_csv("submission.csv", index=False)


if __name__ == "__main__":
    main()

"""Step 2: OOF validation with 5 models x 5 feature sets."""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from scipy.special import expit

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

ROOT = "E:/DaT"
CACHE = "D:/DaT_cache"
OOF_DIR = f"{ROOT}/oof"
W_DIR = f"{CACHE}/watershed"
os.makedirs(OOF_DIR, exist_ok=True)

# Load data
print("Loading data...", flush=True)
y = np.load(f"{ROOT}/cnn3d/y.npy")
uids = np.load(f"{ROOT}/cnn3d/uids.npy", allow_pickle=True)
groups = np.load(f"{ROOT}/cnn3d/groups.npy", allow_pickle=True)

# Load watershed features
feat_atlas = np.load(f"{W_DIR}/features_atlas.npy", allow_pickle=True)
feat_intensity = np.load(f"{W_DIR}/features_intensity.npy", allow_pickle=True)
feat_combined = np.load(f"{W_DIR}/features_combined.npy", allow_pickle=True)

# Build feature matrices
def build_matrix(feat_list, exclude_keys=None):
    if exclude_keys is None:
        exclude_keys = {"uid", "_failed"}
    keys = sorted(set(k for f in feat_list for k in f.keys() if k not in exclude_keys))
    rows = []
    for f in feat_list:
        row = [float(f.get(k, 0.0)) for k in keys]
        rows.append(row)
    X = np.array(rows, dtype=np.float64)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, keys

X_w, w_cols = build_matrix(feat_atlas)
print(f"  Watershed (atlas): {X_w.shape}")

# Load SBR features
sbr_df = pd.read_csv(f"{ROOT}/Dataset/sbr_features_train.csv")
sbr_cols = [c for c in sbr_df.columns if c not in ["uid"]]
X_sbr = sbr_df[sbr_cols].values.astype(np.float64)
X_sbr = np.nan_to_num(X_sbr, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  SBR: {X_sbr.shape}")

# Load phys features
phys_df = pd.read_csv(f"{ROOT}/Dataset/phys_features_train.csv")
phys_cols = [c for c in phys_df.columns if c not in ["uid"]]
X_phys = phys_df[phys_cols].values.astype(np.float64)
X_phys = np.nan_to_num(X_phys, nan=0.0, posinf=0.0, neginf=0.0)
print(f"  Phys: {X_phys.shape}")

# Load CNN OOF (from v24 trained models)
cnn_oof_path = f"{CACHE}/models/oof_preds.npy"
if os.path.exists(cnn_oof_path):
    cnn_oof = np.load(cnn_oof_path)  # (1362, 6)
    X_cnn = cnn_oof.mean(axis=1).reshape(-1, 1)
    print(f"  CNN OOF: {X_cnn.shape}")
else:
    print("  CNN OOF not found, using zeros")
    X_cnn = np.zeros((len(y), 1))

# Define feature sets
feature_sets = {
    "watershed": X_w,
    "sbr_watershed": np.hstack([X_sbr, X_w]),
    "cnn_watershed": np.hstack([X_cnn, X_w]),
    "sbr_cnn_watershed": np.hstack([X_sbr, X_cnn, X_w]),
    "sbr_phys_cnn_watershed": np.hstack([X_sbr, X_phys, X_cnn, X_w]),
}

# Define models
def get_models():
    return {
        "lgb": None,  # Will use LightGBM
        "xgb": None,  # Will use XGBoost
        "et": None,   # Will use ExtraTrees
        "lr": LogisticRegression(C=1.0, max_iter=1000),
        "ridge": RidgeClassifier(alpha=1.0),
    }

# StratifiedGroupKFold splits
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(y, y, groups))

# Run OOF
results = []
all_oof_preds = {}

for fs_name, X in feature_sets.items():
    print(f"\n--- Feature set: {fs_name} ({X.shape[1]} features) ---", flush=True)

    for model_name in ["lgb", "xgb", "et", "lr", "ridge"]:
        print(f"  Model: {model_name}", flush=True)
        oof_preds = np.zeros(len(y))
        fold_metrics = []

        for fold, (train_idx, val_idx) in enumerate(splits):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va, y_va = X[val_idx], y[val_idx]

            # Feature selection on training fold only
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_va_s = scaler.transform(X_va)

            sel = SelectKBest(f_classif, k=min(50, X_tr_s.shape[1]))
            X_tr_k = sel.fit_transform(X_tr_s, y_tr)
            X_va_k = sel.transform(X_va_s)

            if model_name == "lgb":
                from lightgbm import LGBMClassifier
                m = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, verbose=-1)
                m.fit(X_tr_k, y_tr)
                pred = m.predict_proba(X_va_k)[:, 1]
            elif model_name == "xgb":
                from xgboost import XGBClassifier
                m = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                                  use_label_encoder=False, eval_metric="logloss")
                m.fit(X_tr_k, y_tr)
                pred = m.predict_proba(X_va_k)[:, 1]
            elif model_name == "et":
                from sklearn.ensemble import ExtraTreesClassifier
                m = ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42)
                m.fit(X_tr_k, y_tr)
                pred = m.predict_proba(X_va_k)[:, 1]
            elif model_name == "lr":
                m = LogisticRegression(C=1.0, max_iter=1000)
                m.fit(X_tr_s, y_tr)
                pred = m.predict_proba(X_va_s)[:, 1]
            elif model_name == "ridge":
                m = RidgeClassifier(alpha=1.0)
                m.fit(X_tr_s, y_tr)
                pred = expit(m.decision_function(X_va_s))

            oof_preds[val_idx] = pred
            ll = log_loss(y_va, np.clip(pred, 1e-7, 1-1e-7))
            auc = roc_auc_score(y_va, pred)
            bs = brier_score_loss(y_va, pred)
            fold_metrics.append({"fold": fold, "ll": ll, "auc": auc, "brier": bs})

        # Overall metrics
        overall_ll = log_loss(y, np.clip(oof_preds, 1e-7, 1-1e-7))
        overall_auc = roc_auc_score(y, oof_preds)
        overall_brier = brier_score_loss(y, oof_preds)
        avg_fold_ll = np.mean([m["ll"] for m in fold_metrics])
        avg_fold_auc = np.mean([m["auc"] for m in fold_metrics])

        print(f"    LL={overall_ll:.4f} AUC={overall_auc:.4f} Brier={overall_brier:.4f}", flush=True)

        results.append({
            "feature_set": fs_name,
            "model": model_name,
            "n_features": X.shape[1],
            "logloss": overall_ll,
            "auroc": overall_auc,
            "brier": overall_brier,
            "avg_fold_ll": avg_fold_ll,
            "avg_fold_auc": avg_fold_auc,
        })

        # Save OOF
        oof_key = f"{fs_name}_{model_name}"
        all_oof_preds[oof_key] = oof_preds
        np.save(f"{OOF_DIR}/watershed_{oof_key}.npy", oof_preds)

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(f"{ROOT}/reports/watershed_results.csv", index=False)
print(f"\n{'='*60}", flush=True)
print("RESULTS SUMMARY:", flush=True)
print(results_df.to_string(index=False), flush=True)
print(f"\nSaved to {ROOT}/reports/watershed_results.csv", flush=True)

# Save all OOF predictions
np.save(f"{W_DIR}/all_oof_preds.npy", all_oof_preds)

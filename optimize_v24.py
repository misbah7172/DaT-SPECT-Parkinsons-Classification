"""Joint blend+T optimization on OOF predictions."""
import os, sys, json
sys.stdout.reconfigure(line_buffering=True)
import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.linear_model import LogisticRegression

CACHE_DIR = "D:/DaT_cache/volumes_2mm"
MODEL_DIR = "D:/DaT_cache/models"
LABELS_PATH = "E:/DaT/Dataset/train_labels.csv"

labels_df = pd.read_csv(LABELS_PATH)
avail = [f.replace(".npy","") for f in os.listdir(CACHE_DIR) if f.endswith(".npy")]
labels_df = labels_df[labels_df["uid"].isin(avail)].reset_index(drop=True)
uids = labels_df["uid"].tolist()
labs = labels_df["is_pathologic"].values.astype(float)
groups = labels_df["uid"].values

skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
splits = list(skf.split(labs, labs, groups))

# Load CNN OOF
cnn_oof = np.load(os.path.join(MODEL_DIR, "oof_preds.npy"))
print("CNN OOF shape:", cnn_oof.shape)

# Load SBR tabular OOF (need to compute)
# For now, use existing sbr features
SBR_CSV = "E:/DaT/Dataset/sbr_features_train.csv"
site_csv = "E:/DaT/Dataset/site_labels.csv"

import joblib
sbr_df = pd.read_csv(SBR_CSV)
try:
    site_df = pd.read_csv(site_csv)
    sbr_df = sbr_df.merge(site_df, on="uid", how="left")
except:
    pass

# Build biological features
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

sbr_df = add_bio(sbr_df)

# Use submission_v24's sbr_full_cols if available
COLS_PATH = "E:/DaT/submission_v24/weights/sbr_full_cols.json"
if os.path.exists(COLS_PATH):
    with open(COLS_PATH) as f:
        COLS = json.load(f)
else:
    COLS = [c for c in sbr_df.columns if c.startswith("sbr_") or c in ["lr_put_asym","lr_put_ratio","lr_put_sum","lr_put_diff","lr_cau_asym","lr_cau_ratio","lr_cau_sum","lp_post_ant","rp_post_ant","lp_cr","rp_cr","total_str","lr_total_asym"]]

sbr_df_ordered = sbr_df[[c for c in COLS if c in sbr_df.columns]].copy()
sbr_df_ordered = sbr_df_ordered.apply(pd.to_numeric, errors="coerce").fillna(0)

# Compute SBR OOF using pre-trained models
SBR_W = "E:/DaT/submission_v24/weights"
from scipy.special import expit

def compute_sbr_oof(sbr_df_ordered, sbr_cols, splits, labs):
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif

    sbr_oof = np.zeros(len(sbr_df_ordered))
    X_all = sbr_df_ordered.values.astype(np.float64)

    for fold, (train_idx, val_idx) in enumerate(splits):
        X_tr, y_tr = X_all[train_idx], labs[train_idx]
        X_va = X_all[val_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr)
        X_va_s = sc.transform(X_va)

        sel = SelectKBest(f_classif, k=min(50, X_tr_s.shape[1]))
        X_tr_k = sel.fit_transform(X_tr_s, y_tr)
        X_va_k = sel.transform(X_va_s)

        scores = []
        # LR
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1.0, max_iter=1000)
        lr.fit(X_tr_s, y_tr)
        scores.append(lr.predict_proba(X_va_s)[:, 1])

        # Ridge
        from sklearn.linear_model import RidgeClassifier
        ridge = RidgeClassifier(alpha=1.0)
        ridge.fit(X_tr_s, y_tr)
        ridge_pred = expit(ridge.decision_function(X_va_s))
        scores.append(ridge_pred)

        # ET
        from sklearn.ensemble import ExtraTreesClassifier
        et = ExtraTreesClassifier(n_estimators=200, max_depth=8, random_state=42)
        et.fit(X_tr_k, y_tr)
        scores.append(et.predict_proba(X_va_k)[:, 1])

        # XGB
        from xgboost import XGBClassifier
        xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, use_label_encoder=False, eval_metric="logloss")
        xgb.fit(X_tr_k, y_tr)
        scores.append(xgb.predict_proba(X_va_k)[:, 1])

        # LGB
        from lightgbm import LGBMClassifier
        lgb = LGBMClassifier(n_estimators=200, max_depth=4, learning_rate=0.1, verbose=-1)
        lgb.fit(X_tr_k, y_tr)
        scores.append(lgb.predict_proba(X_va_k)[:, 1])

        sbr_oof[val_idx] = np.mean(scores, axis=0)
        print("  SBR fold {}: LL={:.4f}".format(fold, log_loss(labs[val_idx], np.clip(sbr_oof[val_idx],1e-7,1-1e-7))), flush=True)

    return sbr_oof

print("\n--- Computing SBR OOF ---", flush=True)
sbr_oof = compute_sbr_oof(sbr_df_ordered, COLS, splits, labs)

# Now do joint optimization
print("\n--- Joint Blend+T Optimization ---", flush=True)

# For each fold, find optimal w, T
best_ll = float("inf")
best_w = 0.85
best_T = 0.77
results = []

for w in np.arange(0.60, 0.98, 0.02):
    for T in np.arange(0.50, 1.30, 0.02):
        fold_lls = []
        for fold, (_, val_idx) in enumerate(splits):
            p_deep = cnn_oof[val_idx].mean(axis=1)
            p_sbr = sbr_oof[val_idx]
            p_blend = w * p_deep + (1 - w) * p_sbr
            p_blend = np.clip(p_blend, 1e-6, 1 - 1e-6)
            lg = np.log(p_blend / (1.0 - p_blend))
            p_cal = 1.0 / (1.0 + np.exp(-lg / T))
            p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)
            try:
                ll = log_loss(labs[val_idx], p_cal)
            except:
                ll = float("inf")
            fold_lls.append(ll)
        avg_ll = np.mean(fold_lls)
        results.append({"w": w, "T": T, "ll": avg_ll})
        if avg_ll < best_ll:
            best_ll = avg_ll
            best_w = w
            best_T = T

print(f"Best: w={best_w:.2f}, T={best_T:.2f}, LL={best_ll:.4f}", flush=True)

# Platt scaling
print("\n--- Platt Scaling ---", flush=True)
p_deep_all = cnn_oof.mean(axis=1)
p_sbr_all = sbr_oof
p_blend_all = best_w * p_deep_all + (1 - best_w) * p_sbr_all
logits = np.log(np.clip(p_blend_all, 1e-7, 1-1e-7) / (1 - np.clip(p_blend_all, 1e-7, 1-1e-7))).reshape(-1, 1)
lr = LogisticRegression(C=1.0, solver="lbfgs")
lr.fit(logits, labs)
a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
platt_pred = lr.predict_proba(logits)[:, 1]
platt_ll = log_loss(labs, np.clip(platt_pred, 1e-7, 1-1e-7))
platt_auc = roc_auc_score(labs, platt_pred)
print(f"Platt: a={a:.4f}, b={b:.4f}, LL={platt_ll:.4f}, AUC={platt_auc:.4f}", flush=True)

# Also try temperature-only
temp_lls = []
for T in np.arange(0.5, 1.5, 0.01):
    lg = np.log(np.clip(p_blend_all, 1e-6, 1-1e-6) / (1 - np.clip(p_blend_all, 1e-6, 1-1e-6)))
    p_cal = 1.0 / (1.0 + np.exp(-lg / T))
    p_cal = np.clip(p_cal, 1e-7, 1 - 1e-7)
    try:
        ll = log_loss(labs, p_cal)
    except:
        ll = float("inf")
    temp_lls.append((T, ll))
temp_lls.sort(key=lambda x: x[1])
print(f"Temperature-only best: T={temp_lls[0][0]:.2f}, LL={temp_lls[0][1]:.4f}", flush=True)

# Save calibration config
cal = {
    "version": "v24",
    "w_deep": float(best_w),
    "w_sbr": float(1 - best_w),
    "platt_a": float(a),
    "platt_b": float(b),
    "temperature": float(best_T),
    "joint_ll": float(best_ll),
    "platt_ll": float(platt_ll),
    "platt_auc": float(platt_auc),
}
with open(os.path.join(MODEL_DIR, "calibration_v24.json"), "w") as f:
    json.dump(cal, f, indent=2)
print(f"\nSaved calibration to {MODEL_DIR}/calibration_v24.json", flush=True)
print(json.dumps(cal, indent=2), flush=True)

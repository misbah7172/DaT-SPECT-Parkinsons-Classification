"""CNN-only joint T optimization + full OOF evaluation."""
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

cnn_oof = np.load(os.path.join(MODEL_DIR, "oof_preds.npy"))

# Evaluate CNN-only
for fold in range(5):
    _, vi = splits[fold]
    fp = cnn_oof[vi].mean(axis=1)
    ll = log_loss(labs[vi], np.clip(fp,1e-7,1-1e-7))
    auc = roc_auc_score(labs[vi], fp)
    print("Fold {}: CNN LL={:.4f} AUC={:.4f}".format(fold, ll, auc), flush=True)

avg = cnn_oof.mean(axis=1)
ll = log_loss(labs, np.clip(avg,1e-7,1-1e-7))
auc = roc_auc_score(labs, avg)
print("CNN ALL: LL={:.4f} AUC={:.4f}".format(ll, auc), flush=True)

# Temperature optimization (CNN-only)
print("\n--- Temperature Optimization (CNN-only) ---", flush=True)
best_T = 0.77
best_ll = ll
for T in np.arange(0.50, 1.50, 0.01):
    lg = np.log(np.clip(avg,1e-6,1-1e-6) / (1-np.clip(avg,1e-6,1-1e-6)))
    p_cal = 1.0 / (1.0 + np.exp(-lg / T))
    p_cal = np.clip(p_cal, 1e-7, 1-1e-7)
    try:
        ll_t = log_loss(labs, p_cal)
    except:
        continue
    if ll_t < best_ll:
        best_ll = ll_t
        best_T = T

print("Best T={:.2f}, LL={:.4f}".format(best_T, best_ll), flush=True)

# Platt scaling (CNN-only)
logits = np.log(np.clip(avg,1e-7,1-1e-7) / (1-np.clip(avg,1e-7,1-1e-7))).reshape(-1, 1)
lr = LogisticRegression(C=1.0, solver="lbfgs")
lr.fit(logits, labs)
a, b = float(lr.coef_[0][0]), float(lr.intercept_[0])
platt_pred = lr.predict_proba(logits)[:, 1]
platt_ll = log_loss(labs, np.clip(platt_pred,1e-7,1-1e-7))
platt_auc = roc_auc_score(labs, platt_pred)
print("Platt: a={:.4f}, b={:.4f}, LL={:.4f}, AUC={:.4f}".format(a, b, platt_ll, platt_auc), flush=True)

# Per-fold Platt
print("\n--- Per-Fold Platt ---", flush=True)
for fold in range(5):
    _, vi = splits[fold]
    logits_f = np.log(np.clip(cnn_oof[vi].mean(axis=1),1e-7,1-1e-7) / (1-np.clip(cnn_oof[vi].mean(axis=1),1e-7,1-1e-7))).reshape(-1,1)
    lr_f = LogisticRegression(C=1.0, solver="lbfgs")
    lr_f.fit(logits_f, labs[vi])
    pp = lr_f.predict_proba(logits_f)[:,1]
    ll_f = log_loss(labs[vi], np.clip(pp,1e-7,1-1e-7))
    auc_f = roc_auc_score(labs[vi], pp)
    print("  Fold {}: a={:.4f} b={:.4f} LL={:.4f} AUC={:.4f}".format(fold, lr_f.coef_[0][0], lr_f.intercept_[0], ll_f, auc_f), flush=True)

# Save
cal = {
    "version": "v24_cnn_only",
    "w_deep": 1.0,
    "w_sbr": 0.0,
    "platt_a": float(a),
    "platt_b": float(b),
    "temperature": float(best_T),
    "cnn_only_ll": float(ll),
    "cnn_only_auc": float(auc),
    "temp_ll": float(best_ll),
    "platt_ll": float(platt_ll),
    "platt_auc": float(platt_auc),
}
with open(os.path.join(MODEL_DIR, "calibration_v24.json"), "w") as f:
    json.dump(cal, f, indent=2)
print("\nSaved", flush=True)
print(json.dumps(cal, indent=2), flush=True)

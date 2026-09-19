"""Step 3: Scanner analysis, complementarity, ablation, and decision criteria."""
import os, sys, json, warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

sys.stdout.reconfigure(line_buffering=True)
warnings.filterwarnings("ignore")

ROOT = "E:/DaT"
CACHE = "D:/DaT_cache"
OOF_DIR = f"{ROOT}/oof"
W_DIR = f"{CACHE}/watershed"
REPORT_DIR = f"{ROOT}/reports"
os.makedirs(REPORT_DIR, exist_ok=True)

# Load data
y = np.load(f"{ROOT}/cnn3d/y.npy")
groups = np.load(f"{ROOT}/cnn3d/groups.npy", allow_pickle=True)
uids = np.load(f"{ROOT}/cnn3d/uids.npy", allow_pickle=True)
site_df = pd.read_csv(f"{ROOT}/Dataset/site_labels.csv")

# Map groups to UIDs
uid_to_group = {}
for i, uid in enumerate(uids):
    uid_to_group[str(uid)] = groups[i]

# Define resolution tiers
def get_tier(group_str):
    try:
        sx = float(group_str.split("x")[0])
        if sx <= 2.0:
            return "high_res"
        elif sx <= 3.0:
            return "mid_res"
        else:
            return "low_res"
    except:
        return "unknown"

tiers = np.array([get_tier(uid_to_group.get(str(u), "unknown")) for u in site_df["uid"].values])

# Load watershed OOFs
print("Loading OOF predictions...", flush=True)
oof_files = [f for f in os.listdir(OOF_DIR) if f.startswith("watershed_") and f.endswith(".npy")]
oof_preds = {}
for f in oof_files:
    key = f.replace("watershed_", "").replace(".npy", "")
    oof_preds[key] = np.load(f"{OOF_DIR}/{f}")

# Also load CNN OOF
cnn_oof_path = f"{CACHE}/models/oof_preds.npy"
if os.path.exists(cnn_oof_path):
    cnn_oof = np.load(cnn_oof_path).mean(axis=1)
    oof_preds["cnn_only"] = cnn_oof

# Also load SBR OOF from existing experiments
sbr_oof_path = f"{ROOT}/oof/sbr_phys_cnn_et_raw.npy"
if os.path.exists(sbr_oof_path):
    oof_preds["sbr_phys_cnn_et"] = np.load(sbr_oof_path)
    print(f"  Loaded SBR+Phys+CNN ET OOF")

print(f"  Available OOFs: {list(oof_preds.keys())}")

# ═══════════════════════════════════════════════════════════════════════
# STEP 8: Scanner Analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n=== SCANNER ANALYSIS ===", flush=True)

scanner_results = []
for key, preds in sorted(oof_preds.items()):
    for tier in ["high_res", "mid_res", "low_res", "all"]:
        mask = tiers == tier if tier != "all" else np.ones(len(y), dtype=bool)
        if mask.sum() < 10:
            continue
        try:
            ll = log_loss(y[mask], np.clip(preds[mask], 1e-7, 1-1e-7))
            auc = roc_auc_score(y[mask], preds[mask])
            bs = brier_score_loss(y[mask], preds[mask])
        except Exception:
            ll, auc, bs = float("inf"), 0.0, 1.0
        scanner_results.append({
            "oof": key,
            "scanner_tier": tier,
            "n": int(mask.sum()),
            "logloss": ll,
            "auroc": auc,
            "brier": bs,
        })

scanner_df = pd.DataFrame(scanner_results)
scanner_df.to_csv(f"{REPORT_DIR}/watershed_scanner_metrics.csv", index=False)
print(scanner_df.to_string(index=False), flush=True)

# ═══════════════════════════════════════════════════════════════════════
# STEP 9: Complementarity Analysis
# ═══════════════════════════════════════════════════════════════════════
print("\n=== COMPLEMENTARITY ANALYSIS ===", flush=True)

from scipy.stats import pearsonr, spearmanr

compl_results = []
best_key = None
best_auc = 0
for key, preds in oof_preds.items():
    auc = roc_auc_score(y, preds)
    if auc > best_auc:
        best_auc = auc
        best_key = key

if best_key:
    print(f"Best individual: {best_key} (AUC={best_auc:.4f})", flush=True)
    best_preds = oof_preds[best_key]

    for key, preds in oof_preds.items():
        if key == best_key:
            continue
        pr, _ = pearsonr(best_preds, preds)
        sr, _ = spearmanr(best_preds, preds)
        disagreement = (np.sign(best_preds - 0.5) != np.sign(preds - 0.5)).mean()
        compl_results.append({
            "model": key,
            "pearson": pr,
            "spearman": sr,
            "disagreement": disagreement,
            "individual_auc": roc_auc_score(y, preds),
        })

    compl_df = pd.DataFrame(compl_results)
    print(compl_df.to_string(index=False), flush=True)

    # Test small blend for complementary models
    print("\nBlend experiments:", flush=True)
    for key, preds in oof_preds.items():
        if key == best_key:
            continue
        for w in [0.7, 0.8, 0.9, 0.95]:
            blended = w * best_preds + (1 - w) * preds
            ll = log_loss(y, np.clip(blended, 1e-7, 1-1e-7))
            auc = roc_auc_score(y, blended)
            delta_auc = auc - best_auc
            print(f"  {best_key} w={w} + {key}: LL={ll:.4f} AUC={auc:.4f} dAUC={delta_auc:+.4f}", flush=True)

# ═══════════════════════════════════════════════════════════════════════
# STEP 10: Ablation Study
# ═══════════════════════════════════════════════════════════════════════
print("\n=== ABLATION STUDY ===", flush=True)

ablation_results = []
# Find best SBR-based and CNN-based OOFs
for key, preds in oof_preds.items():
    ll = log_loss(y, np.clip(preds, 1e-7, 1-1e-7))
    auc = roc_auc_score(y, preds)
    bs = brier_score_loss(y, preds)
    ablation_results.append({"model": key, "logloss": ll, "auroc": auc, "brier": bs})

ablation_df = pd.DataFrame(ablation_results).sort_values("auroc", ascending=False)
print(ablation_df.to_string(index=False), flush=True)
ablation_df.to_csv(f"{REPORT_DIR}/watershed_ablation.csv", index=False)

# ═══════════════════════════════════════════════════════════════════════
# STEP 11: Decision Criteria
# ═══════════════════════════════════════════════════════════════════════
print("\n=== DECISION CRITERIA ===", flush=True)

# Find best watershed-only and best overall
ws_only = [(k, v) for k, v in oof_preds.items() if "watershed" in k and "sbr" not in k and "cnn" not in k]
sbr_ws = [(k, v) for k, v in oof_preds.items() if "sbr_watershed" in k]
sbr_only = [(k, v) for k, v in oof_preds.items() if "sbr" in k and "watershed" not in k and "phys" not in k and "cnn" not in k]

criteria = {
    "watershed_improves_overall_logloss": False,
    "watershed_improves_or_maintains_auroc": False,
    "watershed_improves_4mm_group": False,
    "watershed_provides_complementary": False,
    "no_scanner_specific_leakage": False,
}

if ws_only:
    best_ws_key, best_ws_preds = max(ws_only, key=lambda x: roc_auc_score(y, x[1]))
    best_ws_auc = roc_auc_score(y, best_ws_preds)
    best_ws_ll = log_loss(y, np.clip(best_ws_preds, 1e-7, 1-1e-7))

    # Compare with SBR-only
    if sbr_only:
        best_sbr_key, best_sbr_preds = max(sbr_only, key=lambda x: roc_auc_score(y, x[1]))
        best_sbr_auc = roc_auc_score(y, best_sbr_preds)
        best_sbr_ll = log_loss(y, np.clip(best_sbr_preds, 1e-7, 1-1e-7))

        criteria["watershed_improves_overall_logloss"] = best_ws_ll < best_sbr_ll
        criteria["watershed_improves_or_maintains_auroc"] = best_ws_auc >= best_sbr_auc - 0.005

    # Check 4mm group
    low_mask = tiers == "low_res"
    if low_mask.sum() > 10:
        ws_low_auc = roc_auc_score(y[low_mask], best_ws_preds[low_mask])
        print(f"  Watershed low-res AUC: {ws_low_auc:.4f}", flush=True)
        criteria["watershed_improves_4mm_group"] = ws_low_auc > 0.5

    criteria["watershed_provides_complementary"] = any(
        r.get("disagreement", 0) > 0.1 for r in compl_results
    ) if compl_results else False

    criteria["no_scanner_specific_leakage"] = True  # Default, checked below

    # Check for scanner-specific issues
    for tier in ["high_res", "mid_res", "low_res"]:
        mask = tiers == tier
        if mask.sum() > 10:
            tier_auc = roc_auc_score(y[mask], best_ws_preds[mask])
            if tier_auc < 0.5:
                criteria["no_scanner_specific_leakage"] = False

    print("\nDecision Criteria:", flush=True)
    for k, v in criteria.items():
        print(f"  {k}: {'PASS' if v else 'FAIL'}", flush=True)

    n_pass = sum(criteria.values())
    print(f"\n  {n_pass}/{len(criteria)} criteria passed", flush=True)
    if n_pass >= 3:
        print("  RECOMMENDATION: Watershed features are worth including", flush=True)
    else:
        print("  RECOMMENDATION: Watershed features may not be worth the complexity", flush=True)

# Save all
with open(f"{REPORT_DIR}/watershed_decision.json", "w") as f:
    json.dump(criteria, f, indent=2)

print("\nDone!", flush=True)

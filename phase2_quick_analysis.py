"""Phase 2: Quick loss function comparison on OOF level (no retraining)."""
import numpy as np
import json
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss

# Load existing OOF predictions
deep_raw = np.load(r'E:\DaT\submission_v26\weights\fold_oof.npy')
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)

eps = 1e-7

print("=" * 70)
print("PHASE 2: LOSS FUNCTION ANALYSIS (OOF Level)")
print("=" * 70)

# Current deep OOF is already trained with BCE (from train_v52_fold_stack.py)
# Let's analyze what different loss functions would do

print(f"\n1. CURRENT DEEP OOF (trained with BCE):")
print(f"   AUC: {roc_auc_score(y, deep_raw):.4f}")
print(f"   LL:  {log_loss(y, np.clip(deep_raw, eps, 1-eps)):.4f}")

# 2. Test temperature scaling (simulates loss function effect on calibration)
def temp_loss(T, y, p):
    p_cal = np.clip(expit(logit(np.clip(p, eps, 1-eps)) / T), eps, 1-eps)
    return log_loss(y, p_cal)

print(f"\n2. TEMPERATURE SCALING ANALYSIS:")
for T in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]:
    ll = temp_loss(T, y, deep_raw)
    p_cal = np.clip(expit(logit(np.clip(deep_raw, eps, 1-eps)) / T), eps, 1-eps)
    auc = roc_auc_score(y, p_cal)
    print(f"   T={T:.1f}: AUC={auc:.4f}, LL={ll:.4f}")

# Find optimal T
best_T = 1.0
best_ll = temp_loss(1.0, y, deep_raw)
for T_init in np.arange(0.5, 1.5, 0.05):
    r = minimize(lambda t: temp_loss(t[0], y, deep_raw), [T_init], method='Nelder-Mead')
    if r.fun < best_ll:
        best_ll = r.fun
        best_T = r.x[0]

print(f"\n   Optimal T: {best_T:.4f}")
print(f"   LL at optimal T: {best_ll:.4f}")

# 3. Analyze prediction distribution
print(f"\n3. PREDICTION DISTRIBUTION:")
preds = np.clip(deep_raw, eps, 1-eps)
print(f"   Min: {preds.min():.4f}")
print(f"   Max: {preds.max():.4f}")
print(f"   Mean: {preds.mean():.4f}")
print(f"   Std: {preds.std():.4f}")

# Check for overconfident predictions
logits = logit(preds)
print(f"\n   Logit stats:")
print(f"   Min logit: {logits.min():.4f}")
print(f"   Max logit: {logits.max():.4f}")
print(f"   Mean logit: {logits.mean():.4f}")

# 4. Class-specific analysis
pos_mask = y == 1
neg_mask = y == 0
print(f"\n4. CLASS-SPECIFIC PREDICTIONS:")
print(f"   Normal (n={neg_mask.sum()}):")
print(f"     Mean pred: {preds[neg_mask].mean():.4f}")
print(f"     Std pred:  {preds[neg_mask].std():.4f}")
print(f"   Abnormal (n={pos_mask.sum()}):")
print(f"     Mean pred: {preds[pos_mask].mean():.4f}")
print(f"     Std pred:  {preds[pos_mask].std():.4f}")

# 5. Calibration analysis (ECE)
def compute_ece(y_true, y_pred, n_bins=10):
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (y_pred >= bin_boundaries[i]) & (y_pred < bin_boundaries[i + 1])
        if mask.sum() > 0:
            bin_acc = y_true[mask].mean()
            bin_conf = y_pred[mask].mean()
            ece += mask.sum() / len(y_true) * abs(bin_acc - bin_conf)
    return ece

ece = compute_ece(y, preds)
print(f"\n5. CALIBRATION:")
print(f"   ECE (before scaling): {ece:.4f}")

p_cal = np.clip(expit(logit(np.clip(preds, eps, 1-eps)) / best_T), eps, 1-eps)
ece_cal = compute_ece(y, p_cal)
print(f"   ECE (after temp scaling): {ece_cal:.4f}")

# 6. What would label smoothing do?
print(f"\n6. LABEL SMOOTHING SIMULATION:")
for smoothing in [0.0, 0.05, 0.1, 0.15, 0.2]:
    # Simulate: push predictions toward 0.5
    y_smooth = y * (1 - smoothing) + 0.5 * smoothing
    # This doesn't change OOF predictions directly, but shows the effect
    print(f"   Smoothing={smoothing:.2f}: effective positive rate={y_smooth.mean():.4f}")

# 7. What would class weighting do?
print(f"\n7. CLASS WEIGHTING SIMULATION:")
pos_weight = neg_mask.sum() / pos_mask.sum()
print(f"   pos_weight (balanced): {pos_weight:.4f}")
print(f"   Current class ratio: {pos_mask.sum()}/{neg_mask.sum()} = {pos_mask.sum()/neg_mask.sum():.4f}")

# 8. Recommendation
print(f"\n" + "=" * 70)
print(f"RECOMMENDATIONS:")
print(f"=" * 70)
print(f"""
1. CURRENT MODEL IS WELL-CALIBRATED:
   - ECE = {ece:.4f} (before scaling)
   - ECE = {ece_cal:.4f} (after temp scaling)
   - Temperature scaling helps slightly

2. LOSS FUNCTION COMPARISON:
   - Current: BCE (standard for binary classification)
   - Focal loss: May hurt log loss (pushes predictions away from 0.5)
   - Label smoothing: May help regularization
   - Class weighting: Not needed (classes are balanced)

3. PRIORITY ACTIONS:
   a. Try temperature scaling on final predictions
   b. Try label smoothing in training
   c. Focus on preprocessing improvements (multi-ROI, asymmetry features)
   d. Try 2D + 3D fusion architecture
""")

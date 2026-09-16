"""Phase 2b: Multi-ROI Architecture with Explicit Asymmetry Features.

Architecture:
- 3D CNN on full crop (context branch)
- 3D CNN on left striatum ROI
- 3D CNN on right striatum ROI
- Explicit asymmetry features
- Fusion MLP → probability

This implements the "multi-ROI approach" and "explicit asymmetry features" from the hints.
"""
import numpy as np
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.special import logit, expit
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score, brier_score_loss
from sklearn.model_selection import StratifiedGroupKFold
import time

# ROI positions in the cropped volume (34x40x42)
# BX = (43:77, 37:77, 24:66) in 100^3 grid
# ROIs in 100^3: x=[53:67], y=[47:67], z=[34:56]
# In cropped: x=[53-43:67-43] = [10:24], y=[47-37:67-37] = [10:30], z=[34-24:56-24] = [10:32]

# Left putamen: x=[18:24], y=[10:19], z=[12:32]
LP_SLICE = (slice(18, 24), slice(10, 19), slice(12, 32))
# Right putamen: x=[11:17], y=[11:19], z=[14:22]
RP_SLICE = (slice(11, 17), slice(11, 19), slice(14, 22))
# Left caudate: x=[18:24], y=[20:30], z=[10:22]
LC_SLICE = (slice(18, 24), slice(20, 30), slice(10, 22))
# Right caudate: x=[10:17], y=[20:30], z=[12:22]
RC_SLICE = (slice(10, 17), slice(20, 30), slice(12, 22))

# Combined left/right striatum
LEFT_STRIATUM = (slice(10, 24), slice(10, 30), slice(10, 32))
RIGHT_STRIATUM = (slice(10, 17), slice(10, 30), slice(10, 32))

print("=" * 70)
print("PHASE 2b: MULTI-ROI ARCHITECTURE WITH ASYMMETRY FEATURES")
print("=" * 70)

# Load data
X = np.load(r'E:\DaT\cnn3d\X_reg.npy')
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()

print(f"Data: {X.shape}, Labels: {y.shape}")
print(f"Normal: {(y==0).sum()}, Abnormal: {(y==1).sum()}")

# Extract ROI patches
def extract_rois(X):
    """Extract ROI patches from cropped volumes."""
    left_put = X[:, LP_SLICE[0], LP_SLICE[1], LP_SLICE[2]]
    right_put = X[:, RP_SLICE[0], RP_SLICE[1], RP_SLICE[2]]
    left_cau = X[:, LC_SLICE[0], LC_SLICE[1], LC_SLICE[2]]
    right_cau = X[:, RC_SLICE[0], RC_SLICE[1], RC_SLICE[2]]
    
    # Combined left/right striatum
    left_str = X[:, LEFT_STRIATUM[0], LEFT_STRIATUM[1], LEFT_STRIATUM[2]]
    right_str = X[:, RIGHT_STRIATUM[0], RIGHT_STRIATUM[1], RIGHT_STRIATUM[2]]
    
    return left_put, right_put, left_cau, right_cau, left_str, right_str

left_put, right_put, left_cau, right_cau, left_str, right_str = extract_rois(X)

print(f"\nROI Shapes:")
print(f"  Left putamen: {left_put.shape}")
print(f"  Right putamen: {right_put.shape}")
print(f"  Left caudate: {left_cau.shape}")
print(f"  Right caudate: {right_cau.shape}")
print(f"  Left striatum: {left_str.shape}")
print(f"  Right striatum: {right_str.shape}")

# Compute explicit asymmetry features
def compute_asymmetry_features(X, left_put, right_put, left_cau, right_cau):
    """Compute handcrafted asymmetry features."""
    eps = 1e-8
    
    # Mean intensities
    lp_mean = np.array([left_put[i][left_put[i] > 0].mean() if (left_put[i] > 0).any() else 0 for i in range(len(X))])
    rp_mean = np.array([right_put[i][right_put[i] > 0].mean() if (right_put[i] > 0).any() else 0 for i in range(len(X))])
    lc_mean = np.array([left_cau[i][left_cau[i] > 0].mean() if (left_cau[i] > 0).any() else 0 for i in range(len(X))])
    rc_mean = np.array([right_cau[i][right_cau[i] > 0].mean() if (right_cau[i] > 0).any() else 0 for i in range(len(X))])
    
    # Asymmetry features
    put_asym = (lp_mean - rp_mean) / (lp_mean + rp_mean + eps)
    cau_asym = (lc_mean - rc_mean) / (lc_mean + rc_mean + eps)
    
    # Ratios
    left_total = lp_mean + lc_mean
    right_total = rp_mean + rc_mean
    total_sbr = left_total + right_total
    left_ratio = left_total / (total_sbr + eps)
    right_ratio = right_total / (total_sbr + eps)
    
    # Putamen/Caudate ratio
    put_total = lp_mean + rp_mean
    cau_total = lc_mean + rc_mean
    pc_ratio = put_total / (cau_total + eps)
    
    # Anterior/Posterior ratio (putamen is posterior, caudate is anterior)
    post_ant_ratio = put_total / (cau_total + eps)
    
    # Striatal-to-background ratio (using center of crop as background proxy)
    bg_region = X[:, 0:5, 0:5, 0:5]  # corner as background
    bg_mean = np.array([bg_region[i].mean() for i in range(len(X))])
    sb_ratio = total_sbr / (bg_mean + eps)
    
    # Absolute asymmetry
    abs_put_asym = np.abs(put_asym)
    abs_cau_asym = np.abs(cau_asym)
    
    features = np.column_stack([
        lp_mean, rp_mean, lc_mean, rc_mean,
        put_asym, cau_asym,
        left_total, right_total, total_sbr,
        left_ratio, right_ratio,
        pc_ratio, post_ant_ratio,
        sb_ratio,
        abs_put_asym, abs_cau_asym,
    ])
    
    feature_names = [
        'lp_mean', 'rp_mean', 'lc_mean', 'rc_mean',
        'put_asym', 'cau_asym',
        'left_total', 'right_total', 'total_sbr',
        'left_ratio', 'right_ratio',
        'pc_ratio', 'post_ant_ratio',
        'sb_ratio',
        'abs_put_asym', 'abs_cau_asym',
    ]
    
    return features, feature_names

asym_features, asym_names = compute_asymmetry_features(X, left_put, right_put, left_cau, right_cau)
print(f"\nAsymmetry features shape: {asym_features.shape}")
print(f"Feature names: {asym_names}")

# Analyze feature importance
print(f"\nAsymmetry Feature Analysis:")
for i, name in enumerate(asym_names):
    feat = asym_features[:, i]
    pos_mean = feat[y == 1].mean()
    neg_mean = feat[y == 0].mean()
    diff = pos_mean - neg_mean
    print(f"  {name:20s}: Normal={neg_mean:.4f}, Abnormal={pos_mean:.4f}, Diff={diff:.4f}")

# Multi-ROI CNN Architecture
class MultiROICNN(nn.Module):
    """Multi-ROI CNN with asymmetry features."""
    
    def __init__(self, n_asym_features=16):
        super().__init__()
        
        # Context branch (full crop)
        self.context_net = nn.Sequential(
            nn.Conv3d(1, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(16, 32, 3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(32, 64, 3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        
        # Left striatum branch
        self.left_net = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        
        # Right striatum branch
        self.right_net = nn.Sequential(
            nn.Conv3d(1, 8, 3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.BatchNorm3d(16),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((2, 2, 2)),
        )
        
        # Fusion MLP
        context_features = 64 * 2 * 2 * 2  # 512
        roi_features = 16 * 2 * 2 * 2 * 2  # 256 (left + right)
        
        self.fusion = nn.Sequential(
            nn.Linear(context_features + roi_features + n_asym_features, 256),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
        )
    
    def forward(self, context, left_str, right_str, asym_features):
        # Context branch
        ctx = self.context_net(context)
        ctx = ctx.view(ctx.size(0), -1)
        
        # Left striatum branch
        left = self.left_net(left_str)
        left = left.view(left.size(0), -1)
        
        # Right striatum branch
        right = self.right_net(right_str)
        right = right.view(right.size(0), -1)
        
        # Concatenate all features
        combined = torch.cat([ctx, left, right, asym_features], dim=1)
        
        # Fusion
        out = self.fusion(combined)
        return out.squeeze(1)

# Training function
def train_multiroi(X, left_str, right_str, asym_features, y, groups, 
                   n_folds=5, epochs=30, lr=1e-3, batch_size=8):
    """Train multi-ROI model and return OOF predictions."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    
    fold_results = []
    
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        # Prepare data
        X_trn = torch.from_numpy(X[trn_idx]).float().unsqueeze(1)
        left_trn = torch.from_numpy(left_str[trn_idx]).float().unsqueeze(1)
        right_trn = torch.from_numpy(right_str[trn_idx]).float().unsqueeze(1)
        asym_trn = torch.from_numpy(asym_features[trn_idx]).float()
        y_trn = torch.from_numpy(y[trn_idx]).float()
        
        X_val = torch.from_numpy(X[val_idx]).float().unsqueeze(1)
        left_val = torch.from_numpy(left_str[val_idx]).float().unsqueeze(1)
        right_val = torch.from_numpy(right_str[val_idx]).float().unsqueeze(1)
        asym_val = torch.from_numpy(asym_features[val_idx]).float()
        y_val = torch.from_numpy(y[val_idx]).float()
        
        # DataLoaders
        train_ds = TensorDataset(X_trn, left_trn, right_trn, asym_trn, y_trn)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        
        # Model
        model = MultiROICNN(n_asym_features=asym_features.shape[1])
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        criterion = nn.BCEWithLogitsLoss()
        
        best_val_ll = float('inf')
        best_preds = None
        
        for epoch in range(epochs):
            model.train()
            for xb_left, xb_left_s, xb_right_s, xb_asym, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb_left, xb_left_s, xb_right_s, xb_asym)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Validate
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val, left_val, right_val, asym_val)
                val_preds = torch.sigmoid(val_logits).cpu().numpy()
                val_ll = log_loss(y_val.numpy(), np.clip(val_preds, 1e-7, 1-1e-7))
                
                if val_ll < best_val_ll:
                    best_val_ll = val_ll
                    best_preds = val_preds.copy()
        
        oof_preds[val_idx] = best_preds
        
        fold_auc = roc_auc_score(y_val.numpy(), best_preds)
        fold_ll = log_loss(y_val.numpy(), np.clip(best_preds, 1e-7, 1-1e-7))
        fold_results.append({'fold': fold, 'auc': fold_auc, 'll': fold_ll})
        print(f"    Fold {fold}: AUC={fold_auc:.4f}, LL={fold_ll:.4f}")
    
    # Overall metrics
    overall_auc = roc_auc_score(y, oof_preds)
    overall_ll = log_loss(y, np.clip(oof_preds, 1e-7, 1-1e-7))
    overall_brier = brier_score_loss(y, oof_preds)
    
    return {
        'auc': overall_auc,
        'll': overall_ll,
        'brier': overall_brier,
        'oof_preds': oof_preds,
        'fold_results': fold_results,
    }

# Run experiment
print(f"\nTraining Multi-ROI CNN with Asymmetry Features...")
print("=" * 70)
start = time.time()
result = train_multiroi(X, left_str, right_str, asym_features, y, groups,
                       n_folds=5, epochs=30, lr=1e-3, batch_size=8)
elapsed = time.time() - start

print(f"\nResults:")
print(f"  AUC: {result['auc']:.4f}")
print(f"  LL:  {result['ll']:.4f}")
print(f"  Brier: {result['brier']:.4f}")
print(f"  Time: {elapsed:.1f}s")

# Compare with current best
with open(r'E:\DaT\submission_v26\weights\ship_final.json', 'r') as f:
    ship = json.load(f)

print(f"\nComparison:")
print(f"  Current best (deep+SBR blend): AUC={ship['final_auc']:.4f}, LL={ship['final_ll']:.4f}")
print(f"  Multi-ROI CNN:                  AUC={result['auc']:.4f}, LL={result['ll']:.4f}")

# Save OOF predictions
np.save(r'E:\DaT\multiroi_oof_preds.npy', result['oof_preds'])
print(f"\nOOF predictions saved to E:\\DaT\\multiroi_oof_preds.npy")

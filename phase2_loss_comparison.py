"""Phase 2: Loss function comparison - BCE vs Focal for log loss optimization."""
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

# Load data
X = np.load(r'E:\DaT\cnn3d\X_reg.npy')
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()

print("=" * 70)
print("PHASE 2: LOSS FUNCTION COMPARISON")
print("=" * 70)
print(f"Data: {X.shape}, Labels: {y.shape}")
print(f"Normal: {(y==0).sum()}, Abnormal: {(y==1).sum()}")

# Define loss functions
class FocalBCE(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean()

class BCEWithPosWeight(nn.Module):
    def __init__(self, pos_weight=1.0):
        super().__init__()
        self.pos_weight = torch.tensor([pos_weight])
    
    def forward(self, logits, targets):
        return F.binary_cross_entropy_with_logits(logits, targets, pos_weight=self.pos_weight.to(logits.device))

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    
    def forward(self, logits, targets):
        smooth_targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return F.binary_cross_entropy_with_logits(logits, smooth_targets)

# Simple 3D CNN for testing
class SimpleCNN3D(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 2 * 2 * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x.squeeze(1)

def train_model(X, y, groups, loss_name, loss_fn, n_folds=5, epochs=30, lr=1e-3, batch_size=8):
    """Train model with specified loss and return OOF predictions."""
    sgkf = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
    oof_preds = np.zeros(len(y))
    oof_targets = np.zeros(len(y))
    
    fold_results = []
    
    for fold, (trn_idx, val_idx) in enumerate(sgkf.split(X, y, groups)):
        X_trn = torch.from_numpy(X[trn_idx]).float().unsqueeze(1)
        y_trn = torch.from_numpy(y[trn_idx]).float()
        X_val = torch.from_numpy(X[val_idx]).float().unsqueeze(1)
        y_val = torch.from_numpy(y[val_idx]).float()
        
        train_ds = TensorDataset(X_trn, y_trn)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        
        model = SimpleCNN3D()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        best_val_ll = float('inf')
        best_preds = None
        
        for epoch in range(epochs):
            model.train()
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            # Validate
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_preds = torch.sigmoid(val_logits).cpu().numpy()
                val_ll = log_loss(y_val.numpy(), np.clip(val_preds, 1e-7, 1-1e-7))
                
                if val_ll < best_val_ll:
                    best_val_ll = val_ll
                    best_preds = val_preds.copy()
        
        oof_preds[val_idx] = best_preds
        oof_targets[val_idx] = y_val.numpy()
        
        fold_auc = roc_auc_score(y_val.numpy(), best_preds)
        fold_ll = log_loss(y_val.numpy(), np.clip(best_preds, 1e-7, 1-1e-7))
        fold_results.append({'fold': fold, 'auc': fold_auc, 'll': fold_ll, 'best_epoch': epoch})
        print(f"    Fold {fold}: AUC={fold_auc:.4f}, LL={fold_ll:.4f}")
    
    # Overall metrics
    overall_auc = roc_auc_score(oof_targets, oof_preds)
    overall_ll = log_loss(oof_targets, np.clip(oof_preds, 1e-7, 1-1e-7))
    overall_brier = brier_score_loss(oof_targets, oof_preds)
    
    return {
        'loss': loss_name,
        'auc': overall_auc,
        'll': overall_ll,
        'brier': overall_brier,
        'oof_preds': oof_preds,
        'fold_results': fold_results,
    }

# Define experiments
experiments = [
    ('BCE', nn.BCEWithLogitsLoss()),
    ('Focal (gamma=2)', FocalBCE(alpha=1.0, gamma=2.0)),
    ('Focal (gamma=1)', FocalBCE(alpha=1.0, gamma=1.0)),
    ('BCE + pos_weight=0.82', BCEWithPosWeight(pos_weight=0.82)),
    ('BCE + pos_weight=1.0', BCEWithPosWeight(pos_weight=1.0)),
    ('Label Smoothing (0.1)', LabelSmoothingBCE(smoothing=0.1)),
]

print(f"\nRunning {len(experiments)} experiments...")
print("=" * 70)

results = []
for name, loss_fn in experiments:
    print(f"\nExperiment: {name}")
    print("-" * 40)
    start = time.time()
    result = train_model(X, y, groups, name, loss_fn, n_folds=5, epochs=30, lr=1e-3, batch_size=8)
    elapsed = time.time() - start
    result['time'] = elapsed
    results.append(result)
    print(f"  Overall: AUC={result['auc']:.4f}, LL={result['ll']:.4f}, Brier={result['brier']:.4f}")
    print(f"  Time: {elapsed:.1f}s")

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"{'Loss Function':30s} {'AUC':>8s} {'LL':>8s} {'Brier':>8s}")
print("-" * 60)
for r in results:
    print(f"{r['loss']:30s} {r['auc']:8.4f} {r['ll']:8.4f} {r['brier']:8.4f}")

# Find best
best_ll = min(results, key=lambda x: x['ll'])
best_auc = max(results, key=lambda x: x['auc'])
print(f"\nBest LL:  {best_ll['loss']} (LL={best_ll['ll']:.4f})")
print(f"Best AUC: {best_auc['loss']} (AUC={best_auc['auc']:.4f})")

# Save results
save_results = [{k: v for k, v in r.items() if k != 'oof_preds'} for r in results]
with open(r'E:\DaT\phase2_loss_comparison.json', 'w') as f:
    json.dump(save_results, f, indent=2)
print(f"\nResults saved to E:\\DaT\\phase2_loss_comparison.json")

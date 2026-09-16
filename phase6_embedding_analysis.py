"""PHASE 6: Test whether CNN is learning the wrong signal.
Extract final CNN embeddings, run PCA, visualize clustering by label/site/spacing."""
import os, json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit

# ============ MODEL (matches actual trained Net3dR weights) ============
class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b1 = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.b2 = nn.Sequential(nn.Conv3d(c,c,3,padding=1), nn.BatchNorm3d(c))
    def forward(self, x):
        return F.relu(x + self.b2(self.b1(x)))

class Net3dR(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(1,16,3,padding=1), nn.BatchNorm3d(16), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            nn.Conv3d(16,32,3,padding=1), nn.BatchNorm3d(32), nn.ReLU(inplace=True), nn.MaxPool3d(2),
            ResBlock3D(32), ResBlock3D(32),
            nn.Conv3d(32,64,3,padding=1), nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(2))
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(64*8, 192), nn.ReLU(inplace=True), nn.Dropout(0.4), nn.Linear(192, 1))
    def forward(self, x):
        return self.head(self.net(x)).squeeze(1)
    def embed(self, x):
        return self.net(x).flatten(1)

# ============ DATA ============
X = np.load(r'E:\DaT\cnn3d\X_reg.npy')[:, None].astype(np.float32)
y = np.load(r'E:\DaT\v26_oof\y.npy').ravel().astype(int)
groups = np.load(r'E:\DaT\v26_oof\groups.npy', allow_pickle=True).ravel()
uids = np.load(r'E:\DaT\cnn3d\uids.npy', allow_pickle=True).ravel() if os.path.exists(r'E:\DaT\cnn3d\uids.npy') else np.arange(len(y))

# Load metadata
voxel_df = pd.read_csv(r'E:\DaT\Dataset\voxel_geometry.csv')
labels_df = pd.read_csv(r'E:\DaT\Dataset\train_labels.csv')
site_df = pd.read_csv(r'E:\DaT\Dataset\site_labels.csv')

# Merge metadata
meta = labels_df.merge(voxel_df, on='uid', how='left')
if 'pseudo_site' in site_df.columns:
    meta = meta.merge(site_df[['uid', 'pseudo_site']], on='uid', how='left')

# Align metadata with X
uid_to_idx = {uid: i for i, uid in enumerate(uids)}
meta_aligned = meta[meta['uid'].isin(uid_to_idx.keys())].copy()
meta_aligned['idx'] = meta_aligned['uid'].map(uid_to_idx)
meta_aligned = meta_aligned.sort_values('idx')

print(f"X: {X.shape}, y: {y.shape}")
print(f"Metadata rows: {len(meta_aligned)}")

# ============ EXTRACT EMBEDDINGS ============
# Load best fold model or train one
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Try to load an existing trained model
model_path = None
for seed in [42, 777, 2024]:
    path = f'E:\\DaT\\submission_v26\\weights\\deep_r_{seed}_f0.pt'
    if os.path.exists(path):
        model_path = path
        break

if model_path:
    print(f"Loading model from {model_path}")
    model = Net3dR().to(DEVICE)
    state_raw = torch.load(model_path, map_location=DEVICE)
    if 'state' in state_raw:
        state = state_raw['state']
    elif 'model' in state_raw:
        state = state_raw['model']
    else:
        state = state_raw
    model.load_state_dict(state)
else:
    print("No pretrained model found, training a quick one...")
    import random
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    model = Net3dR().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).float())
    loader = DataLoader(dataset, batch_size=16, shuffle=True)
    for epoch in range(20):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            loss = F.binary_cross_entropy_with_logits(model(xb), yb)
            optimizer.zero_grad(); loss.backward(); optimizer.step()
    print(f"Quick training done, loss={loss.item():.4f}")

# Extract embeddings
model.eval()
dataset = TensorDataset(torch.from_numpy(X))
loader = DataLoader(dataset, batch_size=32, shuffle=False)

embeddings = []
with torch.no_grad():
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        emb = model.embed(xb)
        embeddings.append(emb.cpu().numpy())

embeddings = np.concatenate(embeddings)
print(f"Embeddings shape: {embeddings.shape}")

# Get predictions
probs = []
with torch.no_grad():
    for (xb,) in loader:
        xb = xb.to(DEVICE)
        logits = model(xb)
        probs.append(expit(logits.cpu().numpy()))
oof_probs = np.concatenate(probs)

# ============ PCA ============
print(f"\nRunning PCA...")
pca = PCA(n_components=50)
pca_emb = pca.fit_transform(StandardScaler().fit_transform(embeddings))
print(f"PCA explained variance: {pca.explained_variance_ratio_[:10].sum():.3f} (top 10)")

# ============ t-SNE ============
print(f"Running t-SNE...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, max_iter=1000)
tsne_emb = tsne.fit_transform(pca_emb[:, :20])

# ============ VISUALIZE ============
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# By label
ax = axes[0, 0]
for label in [0, 1]:
    mask = y == label
    name = 'Normal' if label == 0 else 'Parkinsons'
    ax.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1], s=10, alpha=0.5, label=name)
ax.set_title('t-SNE colored by Label')
ax.legend()

# By prediction confidence
ax = axes[0, 1]
scatter = ax.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=oof_probs, s=10, cmap='RdYlGn_r')
plt.colorbar(scatter, ax=ax)
ax.set_title('t-SNE colored by Prediction (P abnormal)')

# By group/site
ax = axes[0, 2]
groups_aligned = groups[meta_aligned['idx'].values]
unique_groups = np.unique(groups_aligned)
for g in unique_groups[:10]:
    mask = groups_aligned == g
    ax.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1], s=10, alpha=0.5, label=f'Group {g}')
ax.set_title('t-SNE colored by Group (site)')
ax.legend(fontsize=6, markerscale=2)

# By voxel volume
ax = axes[1, 0]
if 'voxel_vol' in meta_aligned.columns:
    vv = meta_aligned['voxel_vol'].values
    scatter = ax.scatter(tsne_emb[:, 0], tsne_emb[:, 1], c=vv, s=10, cmap='viridis')
    plt.colorbar(scatter, ax=ax)
ax.set_title('t-SNE colored by voxel_vol')

# By pseudo_site
ax = axes[1, 1]
if 'pseudo_site' in meta_aligned.columns:
    sites = meta_aligned['pseudo_site'].fillna('unknown').values
    unique_sites = np.unique(sites)
    for s in unique_sites[:8]:
        mask = sites == s
        ax.scatter(tsne_emb[mask, 0], tsne_emb[mask, 1], s=10, alpha=0.5, label=str(s)[:15])
    ax.set_title('t-SNE colored by pseudo_site')
    ax.legend(fontsize=6, markerscale=2)

# PCA variance
ax = axes[1, 2]
ax.bar(range(20), pca.explained_variance_ratio_[:20])
ax.set_title('PCA Explained Variance (top 20)')
ax.set_xlabel('Component')
ax.set_ylabel('Variance')

plt.tight_layout()
plt.savefig(r'E:\DaT\phase6_embedding_analysis.png', dpi=150)
plt.close()
print(f"Saved phase6_embedding_analysis.png")

# ============ QUANTITATIVE ANALYSIS ============
print(f"\n{'='*60}")
print(f"EMBEDDING ANALYSIS")
print(f"{'='*60}")

# Label separation
from sklearn.metrics import silhouette_score, roc_auc_score, log_loss
sil_label = silhouette_score(pca_emb[:, :10], y)
print(f"Silhouette score (label): {sil_label:.4f}")

# Group separation
groups_num = pd.factorize(groups_aligned)[0]
sil_group = silhouette_score(pca_emb[:, :10], groups_num)
print(f"Silhouette score (group): {sil_group:.4f}")

# Per-group AUC
print(f"\nPer-group AUC:")
if 'pseudo_site' in meta_aligned.columns:
    for site in meta_aligned['pseudo_site'].unique():
        mask = meta_aligned['pseudo_site'] == site
        if mask.sum() > 10:
            site_y = y[meta_aligned.loc[mask, 'idx'].values]
            site_p = oof_probs[meta_aligned.loc[mask, 'idx'].values]
            if len(np.unique(site_y)) > 1:
                site_auc = roc_auc_score(site_y, site_p)
                print(f"  {site}: AUC={site_auc:.4f} (n={mask.sum()})")

# Per-spacing AUC
print(f"\nPer-spacing AUC (from voxel_vol):")
if 'voxel_vol' in meta_aligned.columns:
    vv = meta_aligned['voxel_vol'].values
    vv_bins = [0, 10, 20, 50, 100, 1000]
    for i in range(len(vv_bins)-1):
        mask = (vv >= vv_bins[i]) & (vv < vv_bins[i+1])
        if mask.sum() > 10:
            bin_y = y[meta_aligned.loc[mask, 'idx'].values]
            bin_p = oof_probs[meta_aligned.loc[mask, 'idx'].values]
            if len(np.unique(bin_y)) > 1:
                bin_auc = roc_auc_score(bin_y, bin_p)
                print(f"  vol [{vv_bins[i]:.0f}, {vv_bins[i+1]:.0f}): AUC={bin_auc:.4f} (n={mask.sum()})")

# Check first few PCA components
print(f"\nFirst 5 PCA components loading (top features):")
for comp_i in range(5):
    loadings = pca.components_[comp_i]
    top_idx = np.argsort(np.abs(loadings))[-5:][::-1]
    print(f"  PC{comp_i+1}: {[f'{v:.3f}' for v in loadings[top_idx]]}")

# Save embeddings
np.save(r'E:\DaT\phase6_pca_emb.npy', pca_emb)
np.save(r'E:\DaT\phase6_tsne_emb.npy', tsne_emb)
np.save(r'E:\DaT\phase6_embeddings.npy', embeddings)
print(f"\nSaved embeddings to phase6_*.npy")

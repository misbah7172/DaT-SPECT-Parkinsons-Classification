"""Stage 3: CNN Embedding Extractor.

Trains a compact 3D CNN (MONAI ResNet-18 or DenseNet121) and extracts
penultimate-layer embeddings for all scans.

Key design decisions:
- Classifier head is trained with BCE, but deliverable is the embedding
- AMP + gradient checkpointing configurable for 4GB VRAM (local) vs 16GB (Kaggle)
- Early stopping patience accounts for OneCycleLR warmup+peak
- Multiple seeds for ensemble diversity

Usage:
    python -m pipeline.cnn_embed --config pipeline/config.yaml --fold 0
    python -m pipeline.cnn_embed --config pipeline/config.yaml --extract-only
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset

from .utils import ensure_dirs, load_config, make_meta, set_seed, setup_logging

log = logging.getLogger(__name__)


# ─── Dataset ────────────────────────────────────────────────────────────

class VolumeDataset(Dataset):
    """Dataset for loading harmonized volumes + labels."""

    def __init__(
        self,
        cache_dir: Path,
        uids: List[str],
        labels: Optional[List[int]] = None,
        augment: bool = False,
    ):
        self.cache_dir = cache_dir
        self.uids = uids
        self.labels = labels
        self.augment = augment

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        vol = np.load(str(self.cache_dir / f"{uid}.npy"))
        vol = vol[np.newaxis, ...]  # add channel dim: (1, D, H, W)

        if self.augment:
            vol = self._augment(vol)

        vol = torch.from_numpy(vol.astype(np.float32))

        if self.labels is not None:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
            return vol, label
        return vol, uid

    def _augment(self, vol: np.ndarray) -> np.ndarray:
        """Light augmentation: random flip + small rotation."""
        # Random flip along X axis (left-right symmetry)
        if np.random.random() > 0.5:
            vol = vol[:, :, :, ::-1].copy()

        # Random flip along Z axis (anterior-posterior)
        if np.random.random() > 0.5:
            vol = vol[:, ::-1, :, :].copy()

        # Random intensity scale [0.9, 1.1]
        vol = vol * np.random.uniform(0.9, 1.1)

        # Random Gaussian noise (small)
        if np.random.random() > 0.7:
            noise = np.random.normal(0, 0.01, vol.shape).astype(np.float32)
            vol = vol + noise

        return vol


# ─── Model ──────────────────────────────────────────────────────────────

def build_model(
    backbone: str = "resnet18",
    in_channels: int = 1,
    embed_dim: int = 512,
    pretrained: bool = True,
) -> nn.Module:
    """Build 3D CNN model with configurable backbone.

    Args:
        backbone: "resnet18" | "resnet34" | "densenet121"
        in_channels: input channels (1 for DaT-SPECT)
        embed_dim: penultimate layer width
        pretrained: use ImageNet/pretrained weights

    Returns:
        nn.Module with forward() returning (logits, embedding)
    """
    if backbone in ("resnet18", "resnet34"):
        from monai.networks.nets import resnet18, resnet34

        if backbone == "resnet18":
            net = resnet18(pretrained=pretrained, spatial_dims=3, n_input_channels=in_channels)
            # MONAI ResNet-18 final layer is fc(in_features, out_features)
            # We need to replace the head
            in_feat = net.fc.in_features
            net.fc = nn.Identity()  # remove classification head
            actual_embed_dim = in_feat
        else:
            net = resnet34(pretrained=pretrained, spatial_dims=3, n_input_channels=in_channels)
            in_feat = net.fc.in_features
            net.fc = nn.Identity()
            actual_embed_dim = in_feat

    elif backbone == "densenet121":
        from monai.networks.nets import DenseNet121

        net = DenseNet121(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,  # dummy, we'll replace the head
            pretrained=pretrained,
        )
        # MONAI DenseNet121 uses net.class_layers.out as the final Linear
        in_feat = net.class_layers.out.in_features
        net.class_layers.out = nn.Identity()
        actual_embed_dim = in_feat

    else:
        raise ValueError(f"Unknown backbone: {backbone}")

    # Wrap to return both logits and embeddings
    model = ModelWithEmbedding(net, actual_embed_dim, embed_dim)
    return model


class ModelWithEmbedding(nn.Module):
    """Wraps backbone to expose both embedding and logit output."""

    def __init__(self, backbone: nn.Module, backbone_dim: int, embed_dim: int):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(backbone_dim, embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, x):
        emb = self.backbone(x)
        logits = self.head(emb).squeeze(-1)
        return logits, emb


# ─── Training ───────────────────────────────────────────────────────────

@dataclass
class TrainState:
    """Tracks training state for early stopping and LR scheduling."""
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    current_epoch: int = 0
    peak_lr_reached: bool = False
    best_model_state: Optional[dict] = None


def train_fold(
    cfg: Dict[str, Any],
    fold: int,
    train_uids: List[str],
    train_labels: List[int],
    val_uids: List[str],
    val_labels: List[int],
    seed: int = 42,
) -> Dict[str, Any]:
    """Train CNN for one fold, return metrics + embeddings.

    Args:
        cfg: Pipeline config
        fold: fold index
        train_uids, train_labels: training split
        val_uids, val_labels: validation split
        seed: random seed

    Returns:
        Dict with val_metrics, train_embeddings, val_embeddings, model_path
    """
    cnn_cfg = cfg["cnn"]
    dirs = ensure_dirs(cfg)
    cache_dir = dirs["cache"]

    # Seed
    set_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Fold {fold}, seed {seed}, device {device}")

    # Data
    train_ds = VolumeDataset(cache_dir, train_uids, train_labels, augment=True)
    val_ds = VolumeDataset(cache_dir, val_uids, val_labels, augment=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=cnn_cfg.get("batch_size", 16),
        shuffle=True,
        num_workers=cnn_cfg.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cnn_cfg.get("batch_size", 16),
        shuffle=False,
        num_workers=cnn_cfg.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    model = build_model(
        backbone=cnn_cfg.get("backbone", "resnet18"),
        in_channels=cnn_cfg.get("in_channels", 1),
        embed_dim=cnn_cfg.get("embed_dim", 512),
        pretrained=cnn_cfg.get("pretrained", True),
    ).to(device)

    log.info(f"Model params: {sum(p.numel() for p in model.parameters()):,}")

    # Gradient checkpointing (saves ~30% VRAM)
    if cnn_cfg.get("gradient_checkpointing", False):
        from torch.utils.checkpoint import checkpoint
        log.info("Gradient checkpointing enabled")
        # Wrap forward pass to use checkpointing
        original_forward = model.forward

        def checkpointed_forward(x):
            emb = checkpoint(model.backbone, x, use_reentrant=False)
            logits = model.head(emb).squeeze(-1)
            return logits, emb

        model.forward = checkpointed_forward

    # Optimizer + Scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cnn_cfg.get("lr", 1e-4),
        weight_decay=cnn_cfg.get("weight_decay", 1e-4),
    )

    n_epochs = cnn_cfg.get("epochs", 100)
    warmup_epochs = cnn_cfg.get("warmup_epochs", 5)

    # OneCycleLR: warmup for warmup_epochs, then cosine decay
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=cnn_cfg.get("lr", 1e-4),
        total_steps=n_epochs * len(train_loader),
        pct_start=warmup_epochs / n_epochs,
        anneal_strategy="cos",
        div_factor=25,
        final_div_factor=1000,
    )

    # Loss — plain BCE, no label smoothing
    criterion = nn.BCEWithLogitsLoss()

    # AMP
    use_amp = cnn_cfg.get("amp", True) and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    # EMA
    ema_decay = cnn_cfg.get("ema_decay", 0.999)
    ema_state = {k: v.clone() for k, v in model.state_dict().items()}

    # Training loop
    state = TrainState()
    patience = cnn_cfg.get("early_stop_patience", 15)
    n_train_batches = len(train_loader)

    for epoch in range(n_epochs):
        state.current_epoch = epoch
        model.train()

        epoch_loss = 0.0
        n_batches = 0

        for batch_idx, (volumes, labels) in enumerate(train_loader):
            volumes, labels = volumes.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=use_amp):
                logits, _ = model(volumes)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # EMA update
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    ema_state[k] = ema_decay * ema_state[k] + (1 - ema_decay) * v

            epoch_loss += loss.item()
            n_batches += 1

            # Check if we're past warmup phase
            current_lr = scheduler.get_last_lr()[0]
            if epoch >= warmup_epochs and not state.peak_lr_reached:
                state.peak_lr_reached = True
                log.info(f"  Epoch {epoch}: warmup complete, LR={current_lr:.6f}")

        avg_train_loss = epoch_loss / max(n_batches, 1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_true = []

        with torch.no_grad():
            for volumes, labels in val_loader:
                volumes, labels = volumes.to(device), labels.to(device)
                with autocast(enabled=use_amp):
                    logits, _ = model(volumes)
                    loss = criterion(logits, labels)
                val_loss += loss.item()
                val_preds.extend(torch.sigmoid(logits).cpu().numpy())
                val_true.extend(labels.cpu().numpy())

        avg_val_loss = val_loss / max(len(val_loader), 1)

        # Metrics
        val_preds_arr = np.array(val_preds)
        val_true_arr = np.array(val_true)
        val_auc = _compute_auc(val_true_arr, val_preds_arr)

        # Early stopping check
        # ONLY check after warmup is complete
        if state.peak_lr_reached:
            if avg_val_loss < state.best_val_loss:
                state.best_val_loss = avg_val_loss
                state.patience_counter = 0
                # Save best model (EMA weights)
                state.best_model_state = {k: v.clone() for k, v in ema_state.items()}
            else:
                state.patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(
                f"  Epoch {epoch:3d}: train={avg_train_loss:.4f} val={avg_val_loss:.4f} "
                f"AUC={val_auc:.4f} lr={scheduler.get_last_lr()[0]:.6f} "
                f"patience={state.patience_counter}/{patience} "
                f"peak_lr_reached={state.peak_lr_reached}"
            )

        # Checkpoint
        if (epoch + 1) % cnn_cfg.get("checkpoint_every", 5) == 0:
            ckpt_path = dirs["models"] / f"cnn_fold{fold}_seed{seed}_ep{epoch}.pt"
            torch.save(model.state_dict(), ckpt_path)

        # Early stopping
        if state.patience_counter >= patience:
            log.info(f"  Early stopping at epoch {epoch} (patience={patience})")
            break

    # Load best model
    if state.best_model_state is not None:
        model.load_state_dict(state.best_model_state)
    model_path = dirs["models"] / f"cnn_fold{fold}_seed{seed}_best.pt"
    torch.save(model.state_dict(), model_path)
    log.info(f"Best model saved: {model_path} (val_loss={state.best_val_loss:.4f})")

    # Extract embeddings
    train_embeddings = _extract_embeddings(model, train_loader, device, use_amp)
    val_embeddings = _extract_embeddings(model, val_loader, device, use_amp)

    return {
        "fold": fold,
        "seed": seed,
        "val_loss": state.best_val_loss,
        "val_auc": val_auc,
        "model_path": str(model_path),
        "train_embeddings": train_embeddings,
        "val_embeddings": val_embeddings,
        "train_uids": train_uids,
        "val_uids": val_uids,
    }


def _compute_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUROC, handle edge cases."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return 0.5


def _extract_embeddings(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict[str, np.ndarray]:
    """Extract embeddings for all samples in loader.

    Returns:
        Dict mapping uid -> embedding array
    """
    model.eval()
    embeddings = {}
    uids_list = []

    with torch.no_grad():
        for volumes, uids in loader:
            volumes = volumes.to(device)
            with autocast(enabled=use_amp):
                _, emb = model(volumes)
            emb_np = emb.cpu().numpy()
            for i, uid in enumerate(uids):
                embeddings[uid] = emb_np[i]
            uids_list.extend(uids)

    return embeddings


# ─── Cross-Validation Runner ────────────────────────────────────────────

def run_cnn_cv(cfg: Dict[str, Any]) -> pd.DataFrame:
    """Run full CNN training with cross-validation.

    Returns:
        DataFrame with columns [uid, fold, seed, embedding_dim_0, ...]
    """
    dirs = ensure_dirs(cfg)
    meta = make_meta(cfg, "cnn_embed")
    meta.save(dirs["base"] / "cnn_meta.json")

    cnn_cfg = cfg["cnn"]
    cv_cfg = cfg["cv"]

    # Load data
    data_dir = Path(cfg["paths"]["data_dir"])
    from .utils import load_labels_and_groups
    df, groups = load_labels_and_groups(data_dir)

    # Load preprocessing metadata for scanner groups
    cache_dir = dirs["cache"]
    pp_meta_path = cache_dir / "preprocess_metadata.csv"
    if pp_meta_path.exists():
        pp_meta = pd.read_csv(pp_meta_path)
        df = df.merge(pp_meta[["uid", "scanner_group"]], on="uid", how="left", suffixes=("", "_pp"))
        if "scanner_group_pp" in df.columns:
            df["scanner_group"] = df["scanner_group_pp"]
            df.drop(columns=["scanner_group_pp"], inplace=True)

    # Filter to available volumes
    available_uids = [p.stem for p in cache_dir.glob("*.npy")]
    df = df[df["uid"].isin(available_uids)].reset_index(drop=True)
    log.info(f"CNN training: {len(df)} scans with harmonized volumes")

    # Stratified Group K-Fold
    from sklearn.model_selection import StratifiedGroupKFold
    n_folds = cv_cfg.get("n_folds", 5)
    skf = StratifiedGroupKFold(n_splits=n_folds, shuffle=cv_cfg.get("shuffle", True), random_state=cv_cfg.get("random_state", 42))

    all_embeddings = []
    all_metadata = []

    n_seeds = cnn_cfg.get("n_seeds", 3)
    seeds = [42 + i * 1000 for i in range(n_seeds)]

    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df["is_pathologic"], df["scanner_group"])):
        log.info(f"\n{'='*60}")
        log.info(f"FOLD {fold}/{n_folds-1}")
        log.info(f"  Train: {len(train_idx)}, Val: {len(val_idx)}")
        log.info(f"  Train label dist: {df.iloc[train_idx]['is_pathologic'].value_counts().to_dict()}")
        log.info(f"  Val label dist: {df.iloc[val_idx]['is_pathologic'].value_counts().to_dict()}")
        log.info(f"  Val scanner groups: {df.iloc[val_idx]['scanner_group'].value_counts().to_dict()}")

        train_uids = df.iloc[train_idx]["uid"].tolist()
        train_labels = df.iloc[train_idx]["is_pathologic"].tolist()
        val_uids = df.iloc[val_idx]["uid"].tolist()
        val_labels = df.iloc[val_idx]["is_pathologic"].tolist()

        for seed in seeds:
            log.info(f"\n  --- Seed {seed} ---")
            result = train_fold(cfg, fold, train_uids, train_labels, val_uids, val_labels, seed)

            # Store embeddings
            for uid, emb in result["train_embeddings"].items():
                row = {"uid": uid, "fold": fold, "seed": seed}
                for i in range(emb.shape[0]):
                    row[f"emb_{i}"] = float(emb[i])
                all_metadata.append(row)

            for uid, emb in result["val_embeddings"].items():
                row = {"uid": uid, "fold": fold, "seed": seed}
                for i in range(emb.shape[0]):
                    row[f"emb_{i}"] = float(emb[i])
                all_metadata.append(row)

    # Save embeddings
    emb_df = pd.DataFrame(all_metadata)
    emb_path = dirs["oof"] / "cnn_embeddings.csv"
    emb_df.to_csv(emb_path, index=False)
    log.info(f"CNN embeddings saved: {emb_path} ({emb_df.shape})")

    # Save OOF predictions (averaged across seeds per fold)
    # ... (implemented in the full version)

    return emb_df


# ─── Extract-Only Mode ──────────────────────────────────────────────────

def extract_embeddings_only(
    cfg: Dict[str, Any],
    model_paths: Dict[int, str],
) -> pd.DataFrame:
    """Extract embeddings using pre-trained models (no training).

    Args:
        cfg: Pipeline config
        model_paths: dict mapping fold -> model checkpoint path

    Returns:
        DataFrame with uid + embedding columns
    """
    dirs = ensure_dirs(cfg)
    cache_dir = dirs["cache"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = Path(cfg["paths"]["data_dir"])
    from .utils import load_labels_and_groups
    df, _ = load_labels_and_groups(data_dir)

    available_uids = [p.stem for p in cache_dir.glob("*.npy")]
    df = df[df["uid"].isin(available_uids)].reset_index(drop=True)

    all_embeddings = []

    for fold, model_path in model_paths.items():
        log.info(f"Loading fold {fold} model: {model_path}")
        model = build_model(
            backbone=cfg["cnn"].get("backbone", "resnet18"),
            in_channels=cfg["cnn"].get("in_channels", 1),
            embed_dim=cfg["cnn"].get("embed_dim", 512),
            pretrained=False,
        ).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        # Load all volumes
        ds = VolumeDataset(cache_dir, df["uid"].tolist(), augment=False)
        loader = DataLoader(ds, batch_size=cfg["cnn"].get("batch_size", 16), shuffle=False, num_workers=cfg["cnn"].get("num_workers", 4))

        emb_dict = _extract_embeddings(model, loader, device, cfg["cnn"].get("amp", True))

        for uid, emb in emb_dict.items():
            row = {"uid": uid, "fold": fold}
            for i in range(emb.shape[0]):
                row[f"emb_{i}"] = float(emb[i])
            all_embeddings.append(row)

    emb_df = pd.DataFrame(all_embeddings)
    return emb_df


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Stage 3: Train CNN + extract embeddings")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Path to config YAML")
    parser.add_argument("--fold", type=int, default=-1, help="Specific fold to train (-1 = all)")
    parser.add_argument("--seed", type=int, default=-1, help="Specific seed (-1 = all)")
    parser.add_argument("--extract-only", action="store_true", help="Extract embeddings only (no training)")
    parser.add_argument("--model-paths", type=str, default="", help="JSON: {fold: path} for extract-only")
    args = parser.parse_args()

    cfg = load_config(args.config)
    setup_logging(Path(cfg["paths"]["output_dir"]) / "logs", "cnn")

    if args.extract_only:
        if not args.model_paths:
            log.error("--model-paths required for --extract-only")
            return
        model_paths = json.loads(args.model_paths)
        extract_embeddings_only(cfg, {int(k): v for k, v in model_paths.items()})
    else:
        run_cnn_cv(cfg)


if __name__ == "__main__":
    main()

"""Shared utilities for the DaT-SPECT pipeline."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

log = logging.getLogger(__name__)


# ─── Config ─────────────────────────────────────────────────────────────

def load_config(path: str | Path = "pipeline/config.yaml") -> Dict[str, Any]:
    """Load YAML config, return nested dict."""
    with open(path) as f:
        return yaml.safe_load(f)


def config_hash(cfg: Dict) -> str:
    """Short hash of config for traceability."""
    return hashlib.md5(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def get_git_hash() -> str:
    """Current git commit hash, or 'nogit'."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


@dataclass
class RunMeta:
    """Metadata logged with every run."""
    config_hash: str = ""
    git_hash: str = ""
    timestamp: str = ""
    stage: str = ""
    fold: int = -1
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2, default=str)


def make_meta(cfg: Dict, stage: str, fold: int = -1, **extra) -> RunMeta:
    return RunMeta(
        config_hash=config_hash(cfg),
        git_hash=get_git_hash(),
        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
        stage=stage,
        fold=fold,
        extra=extra,
    )


# ─── Scanner groups ─────────────────────────────────────────────────────

def derive_scanner_group(row: pd.Series) -> str:
    """Derive scanner group string from voxel geometry."""
    sx, sy, sz = row.get("sx", 0), row.get("sy", 0), row.get("sz", 0)
    # Round to nearest 0.1 to group similar spacings
    return f"{sx:.1f}x{sy:.1f}x{sz:.1f}"


def load_labels_and_groups(data_dir: str | Path) -> Tuple[pd.DataFrame, List[str]]:
    """Load labels CSV and derive scanner groups.

    Returns:
        DataFrame with columns [uid, is_pathologic, scanner_group, ...]
        List of unique scanner group names
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "train_labels.csv")

    # Try to load site labels for voxel geometry
    site_path = data_dir / "site_labels.csv"
    if site_path.exists():
        site_df = pd.read_csv(site_path)
        df = df.merge(site_df, on="uid", how="left")
    else:
        # Try voxel_geometry.csv
        vg_path = data_dir / "voxel_geometry.csv"
        if vg_path.exists():
            vg = pd.read_csv(vg_path)
            df = df.merge(vg, on="uid", how="left")

    # Derive scanner groups
    if "sx" in df.columns:
        df["scanner_group"] = df.apply(derive_scanner_group, axis=1)
    else:
        df["scanner_group"] = "unknown"

    groups = sorted(df["scanner_group"].unique().tolist())
    log.info(f"Loaded {len(df)} scans, {len(groups)} scanner groups")
    return df, groups


# ─── Path helpers ───────────────────────────────────────────────────────

def ensure_dirs(cfg: Dict) -> Dict[str, Path]:
    """Create output directories, return as Path dict."""
    base = Path(cfg["paths"]["output_dir"])
    dirs = {
        "base": base,
        "cache": base / "cache",
        "oof": base / "oof",
        "models": base / "models",
        "submission": base / "submission",
        "logs": base / "logs",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


# ─── Seed ───────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    import random
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ─── Device ─────────────────────────────────────────────────────────────

def get_device(cfg: Dict) -> "torch.device":
    """Resolve device from config."""
    import torch
    dev = cfg.get("env", {}).get("device", "auto")
    if dev == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(dev)


# ─── Logging ────────────────────────────────────────────────────────────

def setup_logging(log_dir: str | Path, stage: str = "pipeline") -> logging.Logger:
    """Configure file + console logging."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{stage}_{time.strftime('%Y%m%d_%H%M%S')}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return logging.getLogger(__name__)


# ─── NIfTI I/O ─────────────────────────────────────────────────────────

def load_nifti_data(nifti_path: str | Path) -> Tuple[np.ndarray, Any]:
    """Load NIfTI file, return (data_array, affine).

    Uses nibabel for loading; returns raw voxel data + affine matrix.
    """
    import nibabel as nib
    img = nib.load(str(nifti_path))
    data = np.asarray(img.dataobj, dtype=np.float32)
    return data, img.affine


def save_nifti(data: np.ndarray, affine: np.ndarray, path: str | Path):
    """Save array as NIfTI."""
    import nibabel as nib
    img = nib.Nifti1Image(data, affine)
    nib.save(img, str(path))

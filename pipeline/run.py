"""Main Pipeline Orchestrator.

Runs all stages in sequence or individually.
Supports both local and Kaggle execution modes.

Usage:
    # Full pipeline
    python -m pipeline.run --config pipeline/config.yaml --stage all

    # Individual stages
    python -m pipeline.run --config pipeline/config.yaml --stage preprocess
    python -m pipeline.run --config pipeline/config.yaml --stage sbr
    python -m pipeline.run --config pipeline/config.yaml --stage cnn
    python -m pipeline.run --config pipeline/config.yaml --stage gbm
    python -m pipeline.run --config pipeline/config.yaml --stage calibrate
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

from .utils import ensure_dirs, get_git_hash, load_config, setup_logging

log = logging.getLogger(__name__)


STAGES = ["preprocess", "sbr", "cnn", "gbm", "calibrate"]


def run_stage(
    stage: str,
    cfg: Dict[str, Any],
    **kwargs,
) -> Any:
    """Run a single pipeline stage.

    Args:
        stage: stage name
        cfg: pipeline config
        **kwargs: additional arguments passed to stage runner

    Returns:
        Stage output (varies by stage)
    """
    t0 = time.time()
    log.info(f"\n{'='*60}")
    log.info(f"STAGE: {stage.upper()}")
    log.info(f"{'='*60}")

    if stage == "preprocess":
        from .preprocess import run_preprocessing
        result = run_preprocessing(cfg, n_workers=kwargs.get("n_workers", 1))

    elif stage == "sbr":
        from .sbr_features import extract_sbr_features_batch
        from .utils import load_labels_and_groups
        # Load preprocessing metadata
        cache_dir = Path(cfg["paths"]["output_dir"]) / "cache"
        meta_path = cache_dir / "preprocess_metadata.csv"
        if not meta_path.exists():
            log.error("Run preprocessing first")
            return None
        import pandas as pd
        meta_df = pd.read_csv(meta_path)
        result = extract_sbr_features_batch(cfg, meta_df)

    elif stage == "cnn":
        from .cnn_embed import run_cnn_cv
        result = run_cnn_cv(cfg)

    elif stage == "gbm":
        from .train_gbm import run_gbm_cv
        result = run_gbm_cv(cfg)

    elif stage == "calibrate":
        from .calibrate import run_calibration
        result = run_calibration(cfg)

    else:
        raise ValueError(f"Unknown stage: {stage}")

    elapsed = time.time() - t0
    log.info(f"Stage {stage} completed in {elapsed:.1f}s")
    return result


def run_pipeline(
    cfg: Dict[str, Any],
    stages: List[str] = None,
    **kwargs,
) -> Dict[str, Any]:
    """Run full pipeline or selected stages.

    Args:
        cfg: pipeline config
        stages: list of stages to run (default: all)
        **kwargs: additional arguments

    Returns:
        Dict of stage_name -> stage_output
    """
    if stages is None:
        stages = STAGES

    results = {}
    total_t0 = time.time()

    for stage in stages:
        if stage not in STAGES:
            log.warning(f"Unknown stage '{stage}', skipping")
            continue
        try:
            result = run_stage(stage, cfg, **kwargs)
            results[stage] = result
        except Exception as e:
            log.error(f"Stage {stage} failed: {e}")
            results[stage] = None
            # Stop pipeline on failure
            if stage != stages[-1]:
                log.error(f"Stopping pipeline after {stage} failure")
                break

    total_elapsed = time.time() - total_t0
    log.info(f"\nPipeline completed in {total_elapsed:.1f}s")
    log.info(f"Stages run: {list(results.keys())}")

    return results


# ─── Status ─────────────────────────────────────────────────────────────

def pipeline_status(cfg: Dict[str, Any]) -> Dict[str, bool]:
    """Check which stages have been completed.

    Returns:
        Dict of stage_name -> is_complete
    """
    dirs = ensure_dirs(cfg)
    status = {}

    # Preprocess
    meta_path = dirs["cache"] / "preprocess_metadata.csv"
    status["preprocess"] = meta_path.exists()

    # SBR
    sbr_path = dirs["base"] / cfg.get("sbr", {}).get("output_csv", "sbr_features.csv")
    status["sbr"] = sbr_path.exists()

    # CNN
    emb_path = dirs["oof"] / "cnn_embeddings.csv"
    status["cnn"] = emb_path.exists()

    # GBM
    gbm_path = dirs["oof"] / "gbm_oof_predictions.csv"
    status["gbm"] = gbm_path.exists()

    # Calibrate
    cal_path = dirs["oof"] / "gbm_oof_calibrated.csv"
    status["calibrate"] = cal_path.exists()

    return status


# ─── CLI ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DaT-SPECT Pipeline Orchestrator")
    parser.add_argument("--config", default="pipeline/config.yaml", help="Path to config YAML")
    parser.add_argument(
        "--stage",
        default="all",
        choices=STAGES + ["all", "status"],
        help="Stage to run (default: all)",
    )
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers for preprocessing")
    parser.add_argument("--force", action="store_true", help="Force re-run (clear cache)")
    args = parser.parse_args()

    cfg = load_config(args.config)

    # Setup logging
    log_dir = Path(cfg["paths"]["output_dir"]) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(log_dir, "pipeline")

    # Log run info
    log.info(f"Config: {args.config}")
    log.info(f"Git hash: {get_git_hash()}")
    log.info(f"Stage: {args.stage}")

    if args.force:
        import shutil
        cache_dir = Path(cfg["paths"]["output_dir"]) / "cache"
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
            log.info(f"Cleared cache: {cache_dir}")

    if args.stage == "status":
        status = pipeline_status(cfg)
        for stage, complete in status.items():
            mark = "✓" if complete else "✗"
            log.info(f"  {mark} {stage}")
        return

    stages = STAGES if args.stage == "all" else [args.stage]
    results = run_pipeline(cfg, stages, n_workers=args.workers)

    # Save final summary
    summary = {
        "config": args.config,
        "git_hash": get_git_hash(),
        "stages_completed": list(results.keys()),
        "status": pipeline_status(cfg),
    }
    summary_path = Path(cfg["paths"]["output_dir"]) / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary saved: {summary_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/run_teacher_build.py
==============================
Entry point cho POC2 Giai đoạn 1: Teacher Data Generation.

Cách chạy:
    # Full dataset (single_train.json)
    python scripts/run_teacher_build.py --config configs/teacher_data/default.yaml

    # Test nhanh 20 mẫu
    python scripts/run_teacher_build.py --config configs/teacher_data/default.yaml --subset 20

    # Custom run ID
    python scripts/run_teacher_build.py --config configs/teacher_data/default.yaml --run-id my_test_run
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# ── sys.path ─────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.pipelines.teacher_data.config import TeacherConfig
from src.pipelines.teacher_data.orchestrator import TeacherDataOrchestrator


def _setup_logging(run_dir: Path) -> None:
    """Setup logging: file + console."""
    log_dir = run_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"teacher_build_{ts}.log"

    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    file_h = logging.FileHandler(log_file, encoding="utf-8")
    file_h.setLevel(logging.INFO)
    file_h.setFormatter(formatter)

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setLevel(logging.INFO)
    console_h.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    if root.hasHandlers():
        root.handlers.clear()
    root.addHandler(file_h)
    root.addHandler(console_h)

    logging.info(f"Log file: {log_file}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="POC2 GĐ1: Build teacher dataset cho ViInfographicVQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path tới config YAML (e.g., configs/teacher_data/default.yaml)",
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Override run_id (default: auto-generate từ timestamp)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Chỉ chạy N mẫu đầu tiên (e.g., --subset 20 để test)",
    )
    args = parser.parse_args()

    # Load config
    config = TeacherConfig.from_yaml(args.config)

    # Override CLI args
    if args.run_id:
        config.run_id = args.run_id
    if not config.run_id:
        config.run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}_qwen32b"
    if args.subset is not None:
        config.subset = args.subset

    # Setup logging
    run_dir = Path(config.output_dir) / config.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    _setup_logging(run_dir)

    log = logging.getLogger("teacher_data")

    log.info("=" * 60)
    log.info("POC2 Giai đoạn 1: Teacher Data Generation")
    log.info("=" * 60)
    log.info(f"Config:         {args.config}")
    log.info(f"Run ID:         {config.run_id}")
    log.info(f"Model:          {config.model_name}")
    log.info(f"Base URL:       {config.base_url}")
    log.info(f"Stream:         {config.stream}")
    log.info(f"Max concurrent: {config.max_concurrent}")
    log.info(f"Data root:      {config.data_root}")
    log.info(f"JSON files:     {config.json_files}")
    log.info(f"Subset:         {config.subset or 'full dataset'}")
    log.info(f"Output dir:     {run_dir}")
    log.info("=" * 60)

    # Chạy orchestrator
    orchestrator = TeacherDataOrchestrator(config)
    asyncio.run(orchestrator.run())

    log.info("")
    log.info("=" * 60)
    log.info("✅ Hoàn thành!")
    log.info(f"Output: {run_dir}")
    log.info("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())

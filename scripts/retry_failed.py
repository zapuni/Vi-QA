#!/usr/bin/env python3
"""
scripts/retry_failed.py
=========================
Retry các sample đã thất bại trong run trước.

Cách chạy:
    python scripts/retry_failed.py \
        --config configs/teacher_data/default.yaml \
        --run-dir data/teacher_runs/run_20260427_qwen32b
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT))

from src.pipelines.teacher_data.config import TeacherConfig
from src.pipelines.teacher_data.orchestrator import TeacherDataOrchestrator


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Retry các sample thất bại")
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path tới run directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path tới config YAML gốc",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    progress_file = run_dir / "checkpoints" / "progress.json"

    if not progress_file.exists():
        print(f"❌ Không tìm thấy progress file: {progress_file}")
        return 1

    with open(progress_file, encoding="utf-8") as f:
        progress = json.load(f)

    failed_ids = set(progress.get("failed_ids", []))
    if not failed_ids:
        print("✅ Không có sample nào thất bại!")
        return 0

    print(f"🔄 Tìm thấy {len(failed_ids)} failed samples")

    # Load config với cùng run_id (sẽ resume vào cùng output dir)
    config = TeacherConfig.from_yaml(args.config)
    config.run_id = run_dir.name  # Giữ nguyên run_id

    # Xóa failed_ids khỏi progress để orchestrator retry chúng
    progress["failed_ids"] = []
    # Loại bỏ failed_ids khỏi completed_ids (nếu có)
    progress["completed_ids"] = [
        cid for cid in progress["completed_ids"] if cid not in failed_ids
    ]
    progress["stats"]["failed"] = 0

    with open(progress_file, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)

    print(f"🔄 Cleared {len(failed_ids)} failed IDs from progress, retrying...")

    # Chạy lại orchestrator — nó sẽ tự skip completed và process failed
    orchestrator = TeacherDataOrchestrator(config)
    asyncio.run(orchestrator.run())

    print("✅ Retry hoàn tất!")
    return 0


if __name__ == "__main__":
    sys.exit(main())

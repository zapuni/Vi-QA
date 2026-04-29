"""
src/pipelines/teacher_data/checkpoint.py
=========================================
Quản lý checkpoint và resume.
Dùng set cho completed_ids để lookup O(1).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

log = logging.getLogger("teacher_data")


class CheckpointManager:
    """Quản lý checkpoint và resume cho teacher data generation."""

    def __init__(self, output_dir: Path, run_id: str) -> None:
        self.output_dir = output_dir
        self.run_id = run_id

        self.checkpoint_dir = output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.progress_file = self.checkpoint_dir / "progress.json"

        self.failed_dir = output_dir / "failed"
        self.failed_dir.mkdir(parents=True, exist_ok=True)

        self.progress = self._load_progress()
        # Set cho O(1) lookup
        self._completed_set: set[str] = set(self.progress["completed_ids"])

    def _load_progress(self) -> dict:
        """Load progress từ file hoặc tạo mới."""
        if self.progress_file.exists():
            with open(self.progress_file, encoding="utf-8") as f:
                data = json.load(f)
            log.info(
                f"[checkpoint] Resumed: {data['stats']['success']} done, "
                f"{data['stats']['failed']} failed"
            )
            return data

        return {
            "run_id": self.run_id,
            "started_at": datetime.now().isoformat(),
            "completed_ids": [],
            "failed_ids": [],
            "stats": {"success": 0, "failed": 0},
        }

    def _save_progress(self) -> None:
        """Lưu progress ra file."""
        self.progress["updated_at"] = datetime.now().isoformat()
        with open(self.progress_file, "w", encoding="utf-8") as f:
            json.dump(self.progress, f, indent=2, ensure_ascii=False)

    def is_done(self, sample_id: str) -> bool:
        """Kiểm tra sample đã xử lý chưa (O(1))."""
        return sample_id in self._completed_set

    def mark_done(self, sample_id: str) -> None:
        """Đánh dấu sample hoàn thành."""
        if sample_id not in self._completed_set:
            self._completed_set.add(sample_id)
            self.progress["completed_ids"].append(sample_id)
            self.progress["stats"]["success"] += 1
            self._save_progress()

    def mark_failed(self, sample_id: str, error: str) -> None:
        """Đánh dấu sample thất bại + ghi chi tiết lỗi."""
        if sample_id not in self.progress["failed_ids"]:
            self.progress["failed_ids"].append(sample_id)
            self.progress["stats"]["failed"] += 1

            # Ghi chi tiết lỗi vào jsonl riêng
            failed_file = self.failed_dir / "failed_samples.jsonl"
            with open(failed_file, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sample_id": sample_id,
                            "error": error,
                            "timestamp": datetime.now().isoformat(),
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            self._save_progress()

"""
src/pipelines/teacher_data/writer.py
=====================================
Ghi output ra .jsonl shards.
Mỗi shard chứa tối đa shard_size records.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from src.pipelines.teacher_data.models import TeacherOutput

log = logging.getLogger("teacher_data")


class ShardWriter:
    """Ghi TeacherOutput ra .jsonl shards."""

    def __init__(self, output_dir: Path, shard_size: int = 100) -> None:
        self.output_dir = output_dir
        self.shard_size = shard_size

        self.shard_dir = output_dir / "shards"
        self.shard_dir.mkdir(parents=True, exist_ok=True)

        # Tìm shard hiện tại (resume-safe)
        existing = sorted(self.shard_dir.glob("shard_*.jsonl"))
        if existing:
            last = existing[-1]
            self.current_shard_idx = int(last.stem.split("_")[1])
            self.current_shard_count = sum(1 for _ in open(last))
            # Nếu shard cuối đã đầy, tạo shard mới
            if self.current_shard_count >= self.shard_size:
                self.current_shard_idx += 1
                self.current_shard_count = 0
        else:
            self.current_shard_idx = 0
            self.current_shard_count = 0

    @property
    def current_shard_path(self) -> Path:
        return self.shard_dir / f"shard_{self.current_shard_idx:04d}.jsonl"

    def write(self, output: TeacherOutput) -> None:
        """Atomic write một record vào shard hiện tại."""
        # Rotate shard nếu đầy
        if self.current_shard_count >= self.shard_size:
            self.current_shard_idx += 1
            self.current_shard_count = 0
            log.info(f"[writer] New shard: {self.current_shard_path.name}")

        # Serialize, bỏ raw_response để giảm kích thước
        record = output.model_dump()
        # Giữ raw_response nhưng chỉ lấy usage + model
        raw = record.get("raw_response", {})
        record["raw_response"] = {
            "model": raw.get("model", ""),
            "usage": raw.get("usage", {}),
        }

        with open(self.current_shard_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            f.flush()

        self.current_shard_count += 1

"""
src/pipelines/teacher_data/loader.py
=====================================
Load dataset từ file JSON local (cấu trúc ViInfographicVQA).
Trả về iterator của RawSample.

Hỗ trợ 2 format JSON:
  Single-image: {"image_path": "images/xxx.jpg", ...}
  Multi-image:  {"image_paths": ["images/a.jpg", "images/b.jpg"], ...}
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterator

from src.pipelines.teacher_data.models import RawSample

log = logging.getLogger("teacher_data")


class DatasetLoader:
    """Load ViInfographicVQA JSON files thành RawSample objects."""

    def __init__(self, data_root: str, json_files: list[str]) -> None:
        self.data_root = Path(data_root)
        self.json_files = json_files

    def iter_samples(self, subset: int | None = None) -> Iterator[RawSample]:
        """
        Iterate qua tất cả samples từ JSON files.

        Args:
            subset: Nếu set, chỉ trả N samples đầu tiên.
        """
        count = 0
        single_count = 0
        multi_count = 0

        for jf in self.json_files:
            json_path = self.data_root / "data" / jf
            if not json_path.exists():
                log.warning(f"[loader] JSON file not found: {json_path}, skipping")
                continue

            with open(json_path, encoding="utf-8") as f:
                records = json.load(f)

            log.info(f"[loader] Loaded {len(records)} records from {json_path.name}")

            for item in records:
                if subset is not None and count >= subset:
                    return

                # ── Xác định danh sách image paths ───────────────────
                # Ưu tiên "image_paths" (list) trước, fallback "image_path" (str)
                if "image_paths" in item:
                    rel_paths = item["image_paths"]
                elif "image_path" in item:
                    rel_paths = [item["image_path"]]
                else:
                    log.warning(
                        f"[loader] Record {item.get('question_id', '?')} "
                        f"thiếu cả 'image_path' lẫn 'image_paths', skipping"
                    )
                    continue

                # Build absolute image paths
                abs_paths = [str(self.data_root / rp) for rp in rel_paths]

                # Track single vs multi
                if len(abs_paths) > 1:
                    multi_count += 1
                else:
                    single_count += 1

                sample = RawSample(
                    sample_id=str(item["question_id"]),
                    image_paths=abs_paths,
                    question=str(item["question"]).strip(),
                    answer=str(item.get("answer", "")).strip(),
                    answer_source=str(item.get("answer_source", "unknown")),
                    metadata={
                        "image_type": item.get("image_type", "unknown"),
                        "element": item.get("element", "unknown"),
                        "json_file": jf,
                        "num_images": len(abs_paths),
                    },
                )
                yield sample
                count += 1

        log.info(
            f"[loader] Total samples yielded: {count} "
            f"(single={single_count}, multi={multi_count})"
        )

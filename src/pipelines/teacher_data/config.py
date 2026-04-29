"""
src/pipelines/teacher_data/config.py
=====================================
Config cho teacher data generation pipeline.
Load từ YAML file, hỗ trợ override qua CLI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TeacherConfig:
    """Config cho teacher data generation"""

    # Teacher model
    model_name: str = "Qwen/Qwen3-VL-32B-Thinking"
    base_url: str = "http://localhost:8000/v1"
    api_key: str = "EMPTY"

    # Generation params
    temperature: float = 0.0
    max_completion_tokens: int = 4096
    enable_thinking: bool = True
    stream: bool = True  # Stream response để tránh lỗi 524 timeout từ reverse proxy

    # Concurrency
    max_concurrent: int = 4
    retry_attempts: int = 3
    retry_backoff: float = 2.0
    timeout_seconds: float = 120.0

    # Paths — tương thích với cấu trúc hiện tại: data/ViInfographicVQA/
    data_root: str = "data/ViInfographicVQA"
    json_files: list[str] = field(default_factory=lambda: ["single_train.json"])
    output_dir: str = "data/teacher_runs"

    # Checkpoint
    checkpoint_interval: int = 10
    shard_size: int = 100

    # Run metadata
    run_id: str = ""
    seed: int = 42
    subset: int | None = None   # None = full dataset, int = lấy N mẫu đầu

    @classmethod
    def from_yaml(cls, path: str | Path) -> "TeacherConfig":
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Flatten nested keys if present
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})

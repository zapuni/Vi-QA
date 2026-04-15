"""
modules/data/loader.py
======================
Trách nhiệm DUY NHẤT: Load dataset từ file JSON local + ảnh local,
chuẩn hóa thành list[dict] với keys nhất quán.

Dataset được tải thủ công bằng `hf download` vào thư mục `data/`.
Cấu trúc thư mục:
    data/
    ├── data/
    │   ├── single_test.json
    │   ├── single_train.json
    │   ├── multi_test.json
    │   └── multi_train.json
    └── images/
        ├── 4813.jpg
        ├── 5707.jpg
        └── ...

Mỗi entry trong JSON có dạng:
    {
        "question_id": "47438",
        "image_type": "Dự báo thời tiết",      ← domain
        "answer_source": "multi-span",
        "element": "Visual/Layout",
        "question": "...",
        "answer": "...",
        "image_path": "images/4813.jpg"         ← relative path
    }

Keys trả ra (chuẩn hóa):
    id            – str  (question_id)
    image         – PIL.Image.Image (RGB)
    question      – str
    answer        – str  (ground truth)
    answer_source – str  ("image-span" | "question-span" | "multi-span" | "non-extractive")
    domain        – str  (image_type)
    image_path    – str  (đường dẫn tuyệt đối tới ảnh)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from PIL import Image

log = logging.getLogger("poc1")


def _normalize(item: dict, data_root: Path) -> dict:
    """Chuẩn hóa 1 sample từ JSON dict thành dict chuẩn."""
    # Xây dựng đường dẫn ảnh tuyệt đối
    # image_path trong JSON là relative, ví dụ: "images/4813.jpg"
    # data_root là thư mục gốc chứa data/, images/ (ví dụ: poc1/data/)
    image_rel = item["image_path"]
    image_abs = data_root / image_rel

    if not image_abs.exists():
        raise FileNotFoundError(f"Image not found: {image_abs}")

    image = Image.open(image_abs)
    if image.mode != "RGB":
        image = image.convert("RGB")

    return {
        "id":            str(item["question_id"]),
        "image":         image,
        "question":      str(item["question"]).strip(),
        "answer":        str(item["answer"]).strip(),
        "answer_source": str(item.get("answer_source", "unknown")),
        "domain":        str(item.get("image_type", "unknown")),
        "image_path":    str(image_abs),
    }


def load(cfg: dict) -> list[dict]:
    """
    Load và chuẩn hóa dataset theo config.

    Đọc file JSON từ thư mục local (tải thủ công bằng `hf download`).

    Args:
        cfg: full config dict (đọc từ poc1_config.yaml)

    Returns:
        list[dict] sau khi chuẩn hóa.
    """
    data_root = Path(cfg["dataset"]["data_root"])
    split     = cfg["split"]  # "test" hoặc "train"

    if not data_root.exists():
        raise FileNotFoundError(
            f"[data/loader] Data root not found: {data_root}\n"
            f"Hãy tải dataset: hf download duytranus/ViInfographicVQA "
            f"--repo-type=dataset --local-dir {data_root}"
        )

    # Xác định file JSON cần đọc dựa trên split
    # Mặc định đọc single_test.json cho split="test"
    json_files = cfg["dataset"].get("json_files")
    if json_files is None:
        json_files = [f"single_{split}.json"]

    all_records: list[dict] = []
    for jf in json_files:
        json_path = data_root / "data" / jf
        if not json_path.exists():
            log.warning(f"[data/loader] JSON file not found: {json_path}, skipping")
            continue

        with open(json_path, encoding="utf-8") as f:
            records = json.load(f)

        log.info(f"[data/loader] ✔ Loaded {len(records)} records from {json_path.name}")
        all_records.extend(records)

    if not all_records:
        raise FileNotFoundError(
            f"[data/loader] No data files found in {data_root / 'data'}. "
            f"Expected files: {json_files}"
        )

    # Chuẩn hóa tất cả records
    samples = []
    skipped = 0
    for i, item in enumerate(all_records):
        try:
            samples.append(_normalize(item, data_root))
        except Exception as e:
            log.warning(f"[data/loader] Skip record {i}: {e}")
            skipped += 1

    if skipped > 0:
        log.warning(f"[data/loader] ⚠ Skipped {skipped}/{len(all_records)} records")

    log.info(
        f"[data/loader] ✔ {len(samples)} samples loaded "
        f"(split='{split}', files={json_files})"
    )
    return samples

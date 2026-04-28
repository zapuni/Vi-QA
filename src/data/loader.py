"""
src/data/loader.py
======================
Trách nhiệm DUY NHẤT: Load dataset từ file JSON local,
chuẩn hóa thành list[dict] với keys nhất quán.

**QUAN TRỌNG**: Module này KHÔNG load ảnh PIL vào RAM.
Chỉ lưu đường dẫn tuyệt đối (image_paths) để downstream
code (runner.py) load on-demand qua ImageProvider.

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

Hỗ trợ 2 format JSON:

  Single-image (single_*.json):
    {
        "question_id": "47438",
        "image_type": "Dự báo thời tiết",
        "answer_source": "multi-span",
        "element": "Visual/Layout",
        "question": "...",
        "answer": "...",
        "image_path": "images/4813.jpg"         ← string đơn
    }

  Multi-image (multi_*.json):
    {
        "question_id": "5560",
        "image_paths": [                         ← list of strings
            "images/18777.jpg",
            "images/18779.jpg",
            "images/18787.jpg"
        ],
        "image_type": "Thể thao - Nghệ thuật",
        "answer_source": "Cross-Image Synthesis",
        "question": "...",
        "answer": "..."
    }

Keys trả ra (chuẩn hóa):
    id            – str  (question_id)
    image_paths   – list[str]  (đường dẫn tuyệt đối tới từng ảnh)
    question      – str
    answer        – str  (ground truth)
    answer_source – str
    domain        – str  (image_type)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

log = logging.getLogger("poc1")


def _validate_image_paths(image_paths: list[str], data_root: Path) -> list[str]:
    """
    Validate và build absolute paths cho danh sách ảnh.

    KHÔNG load PIL Images — chỉ kiểm tra file tồn tại.
    Ảnh sẽ được load on-demand bởi ImageProvider khi cần.

    Args:
        image_paths: Danh sách relative paths (vd: ["images/4813.jpg"])
        data_root:   Thư mục gốc chứa images/

    Returns:
        list[str] absolute paths
    """
    abs_paths = []
    for rel_path in image_paths:
        abs_path = data_root / rel_path
        if not abs_path.exists():
            raise FileNotFoundError(f"Image not found: {abs_path}")
        abs_paths.append(str(abs_path))
    return abs_paths


def _normalize(item: dict, data_root: Path) -> dict:
    """Chuẩn hóa 1 sample từ JSON dict thành dict chuẩn.

    Tự động phát hiện format:
      - Nếu có key "image_paths" (list) → multi-image
      - Nếu có key "image_path" (str)  → single-image, wrap thành list

    LƯU Ý: Trả về image_paths (strings) thay vì PIL Images.
    Ảnh sẽ được load lazy bởi ImageProvider trong runner.py.
    """
    # Xác định danh sách relative image paths
    if "image_paths" in item:
        # Multi-image format: image_paths là list
        rel_paths = item["image_paths"]
    elif "image_path" in item:
        # Single-image format: image_path là string → wrap thành list
        rel_paths = [item["image_path"]]
    else:
        raise KeyError(
            f"Record {item.get('question_id', '?')} thiếu cả "
            f"'image_path' lẫn 'image_paths'"
        )

    abs_paths = _validate_image_paths(rel_paths, data_root)

    return {
        "id":            str(item["question_id"]),
        "image_paths":   abs_paths,     # list[str] — chỉ lưu paths, KHÔNG load PIL
        "question":      str(item["question"]).strip(),
        "answer":        str(item["answer"]).strip(),
        "answer_source": str(item.get("answer_source", "unknown")),
        "domain":        str(item.get("image_type", "unknown")),
    }


def load(cfg: dict) -> list[dict]:
    """
    Load và chuẩn hóa dataset theo config.

    Đọc file JSON từ thư mục local (tải thủ công bằng `hf download`).

    Args:
        cfg: full config dict (đọc từ benchmark.yaml)

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
    single_count = 0
    multi_count = 0
    for i, item in enumerate(all_records):
        try:
            sample = _normalize(item, data_root)
            samples.append(sample)
            if len(sample["image_paths"]) == 1:
                single_count += 1
            else:
                multi_count += 1
        except Exception as e:
            log.warning(f"[data/loader] Skip record {i}: {e}")
            skipped += 1

    if skipped > 0:
        log.warning(f"[data/loader] ⚠ Skipped {skipped}/{len(all_records)} records")

    log.info(
        f"[data/loader] ✔ {len(samples)} samples loaded "
        f"(single={single_count}, multi={multi_count}, "
        f"split='{split}', files={json_files})"
    )
    return samples

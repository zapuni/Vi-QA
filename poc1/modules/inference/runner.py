"""
modules/inference/runner.py
============================
Trách nhiệm DUY NHẤT: Chạy inference cho 1 model trên toàn bộ dataset.

Tính năng quan trọng:
  - Resume-safe: ghi kết quả append từng dòng vào .jsonl
    → nếu bị kill giữa chừng, chạy lại sẽ skip các mẫu đã xong
  - Đo latency mỗi sample (wall-clock time perf_counter)
  - Dynamic import adapter → không load model không cần dùng

Output: results/{model_key}/predictions.jsonl
"""

from __future__ import annotations

import logging

import json
import time
from pathlib import Path

from tqdm import tqdm

from modules.inference.base import VLMAdapter, release_torch_memory

# ── Registry: map type string → adapter class path ───────────────────
# Thêm model mới: chỉ cần thêm 1 dòng ở đây
_ADAPTER_REGISTRY: dict[str, str] = {
    "vintern": "modules.inference.adapters.vintern.VinternAdapter",
    "qwen":    "modules.inference.adapters.qwen.QwenVLAdapter",
}


def _import_adapter(adapter_type: str):
    """
    Dynamic import adapter class để tránh load dependencies không cần thiết.
    Ví dụ: khi chỉ chạy vintern, Qwen sẽ không được import.
    """
    if adapter_type not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"[runner] Unknown adapter type: '{adapter_type}'. "
            f"Available: {list(_ADAPTER_REGISTRY.keys())}"
        )
    module_path, class_name = _ADAPTER_REGISTRY[adapter_type].rsplit(".", 1)
    module = __import__(module_path, fromlist=[class_name])
    return getattr(module, class_name)


def _load_done_ids(jsonl_path: Path) -> set[str]:
    """Đọc file jsonl hiện tại, trả về set id đã xử lý (để resume)."""
    if not jsonl_path.exists():
        return set()
    done = set()
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                done.add(str(obj["id"]))
            except (json.JSONDecodeError, KeyError):
                pass
    return done


def run(
    model_key:  str,
    model_cfg:  dict,
    samples:    list[dict],
    cfg:        dict,
) -> str:
    """
    Chạy batch inference cho 1 model.

    Args:
        model_key:  Tên key model (vd: "vintern-1b").
        model_cfg:  cfg["models"][model_key] từ config yaml.
        samples:    list[dict] đã chuẩn hóa từ loader.
        cfg:        Full config dict.

    Returns:
        Đường dẫn tuyệt đối đến file predictions.jsonl.
    """
    out_dir  = Path(cfg["output_dir"]) / model_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "predictions.jsonl"

    # ── Resume: xác định samples chưa xử lý ─────────────────────────
    done_ids = _load_done_ids(out_path)
    if done_ids:
        logging.getLogger("poc1").info(f"[runner/{model_key}] Resume: {len(done_ids)} done, skipping…")

    pending = [
        s for s in samples
        if str(s["id"]) not in done_ids
    ]

    if not pending:
        logging.getLogger("poc1").info(f"[runner/{model_key}] ✔ All {len(samples)} samples already done.")
        return str(out_path)

    # ── Load model ───────────────────────────────────────────────────
    release_torch_memory(tag=f"{model_key}/before-load")
    AdapterClass = _import_adapter(model_cfg["type"])
    adapter: VLMAdapter = AdapterClass(
        model_key=model_key,
        hf_id=model_cfg["hf_id"],
        cfg=cfg,
    )

    # ── Inference loop ───────────────────────────────────────────────
    errors = 0
    try:
        adapter.load()
        with open(out_path, "a", encoding="utf-8") as fout:
            for sample in tqdm(pending, desc=f"[{model_key}]", unit="sample"):
                t0 = time.perf_counter()
                try:
                    prediction = adapter.infer(sample["image"], sample["question"])
                    error_msg  = None
                except Exception as exc:
                    prediction = ""
                    error_msg  = str(exc)
                    errors += 1

                latency = round(time.perf_counter() - t0, 4)

                record = {
                    "id":            sample["id"],
                    "question":      sample["question"],
                    "ground_truth":  sample["answer"],
                    "prediction":    prediction,
                    "latency":       latency,
                    "answer_source": sample["answer_source"],
                    "domain":        sample["domain"],
                    "error":         error_msg,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()   # flush sau mỗi dòng để không mất data nếu crash
    finally:
        adapter.unload()
        release_torch_memory(tag=f"{model_key}/after-unload")

    total = len(done_ids) + len(pending)
    logging.getLogger("poc1").info(
        f"[runner/{model_key}] ✔ Done: {total} samples "
        f"(errors: {errors}) → {out_path}"
    )
    return str(out_path)

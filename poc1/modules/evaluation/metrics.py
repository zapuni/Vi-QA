"""
modules/evaluation/metrics.py
==============================
Trách nhiệm DUY NHẤT: Tính toán metrics cho từng cặp (prediction, ground_truth).

Metrics:
  - ANLS (Average Normalized Levenshtein Similarity):
      score = Levenshtein.ratio(pred, gt) nếu score >= threshold, else 0
      → đo độ tương đồng chuỗi, mạnh hơn EM khi answer gần đúng
  - Exact Match (EM):
      1 nếu pred == gt (sau khi normalize), 0 nếu không

Module này KHÔNG biết về file, dataset, hay model.
Chỉ nhận 2 string, trả ra số.
"""

from __future__ import annotations

import Levenshtein


# ── Text normalization ───────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Chuẩn hóa text trước khi so sánh.
    Áp dụng cho cả prediction và ground truth.
    """
    return (
        text
        .lower()
        .strip()
        .rstrip(".")           # bỏ dấu chấm cuối
        .replace('"', "")      # bỏ nháy kép
        .replace("\n", " ")    # xuống dòng → space
        .strip()
    )


# ── Core metrics ─────────────────────────────────────────────────────

def anls(pred: str, gt: str, threshold: float = 0.5) -> float:
    """
    Average Normalized Levenshtein Similarity.

    Args:
        pred:      Câu trả lời của model (raw).
        gt:        Ground truth (raw).
        threshold: Nếu similarity < threshold → score = 0.

    Returns:
        float trong [0.0, 1.0]
    """
    p = normalize(pred)
    g = normalize(gt)
    score = Levenshtein.ratio(p, g)
    return float(score) if score >= threshold else 0.0


def exact_match(pred: str, gt: str) -> int:
    """
    Exact Match sau khi normalize.

    Returns:
        1 nếu khớp hoàn toàn, 0 nếu không.
    """
    return int(normalize(pred) == normalize(gt))

"""
src/evaluation/analyzer.py
================================
Trách nhiệm DUY NHẤT: Đọc predictions.jsonl, tính metrics,
trả về structured report dict.

Report gồm:
  overall      – ANLS%, EM%, n
  by_domain    – breakdown theo domain
  by_source    – breakdown theo answer_source (extractive / non-extractive…)
  latency      – mean, p50, p95 (giây)
  failure_cases – list records có ANLS == 0 (dùng cho error analysis)

Module này KHÔNG vẽ chart, KHÔNG print ra màn hình quá nhiều.
Nó chỉ tính và trả dict.
"""

from __future__ import annotations

import logging

import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from src.evaluation.metrics import anls, exact_match


# ── Helpers ──────────────────────────────────────────────────────────

def _load_predictions(jsonl_path: str) -> list[dict]:
    """Đọc toàn bộ file .jsonl thành list[dict]."""
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"[analyzer] Predictions file not found: {path}")

    records = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.getLogger("poc1").info(f"[analyzer] Warning: skip malformed line {i}: {e}")
    return records


def _empty_stats() -> dict[str, Any]:
    return {"anls_sum": 0.0, "em_sum": 0, "n": 0}


def _finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    """Tính trung bình phần trăm từ tổng."""
    n = stats["n"]
    if n == 0:
        return {"anls": 0.0, "exact_match": 0.0, "n": 0}
    return {
        "anls":        round(stats["anls_sum"] / n * 100, 2),
        "exact_match": round(stats["em_sum"]   / n * 100, 2),
        "n":           n,
    }


def _latency_stats(latencies: list[float]) -> dict[str, float]:
    """Tính mean, median (p50), p95 của danh sách latency."""
    if not latencies:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    s = sorted(latencies)
    n = len(s)
    return {
        "mean": round(statistics.mean(s), 3),
        "p50":  round(s[int(n * 0.50)], 3),
        "p95":  round(s[min(int(n * 0.95), n - 1)], 3),
    }


# ── Main analysis function ───────────────────────────────────────────

def analyze(jsonl_path: str) -> dict:
    """
    Phân tích đầy đủ predictions.jsonl.

    Args:
        jsonl_path: Đường dẫn file predictions.jsonl.

    Returns:
        Dict report với structure:
        {
          "overall":       {"anls": float, "exact_match": float, "n": int},
          "by_domain":     {domain: {...}},
          "by_source":     {source: {...}},
          "latency":       {"mean": float, "p50": float, "p95": float},
          "failure_cases": [list of records with anls==0]
        }
    """
    records = _load_predictions(jsonl_path)
    logging.getLogger("poc1").info(f"[analyzer] Loaded {len(records)} records from {jsonl_path}")

    overall   = _empty_stats()
    by_domain = defaultdict(_empty_stats)
    by_source = defaultdict(_empty_stats)
    latencies = []
    failures  = []
    skipped   = 0

    for rec in records:
        # Bỏ qua các record lỗi inference
        if rec.get("error"):
            skipped += 1
            continue

        gt   = rec.get("ground_truth", "")
        pred = rec.get("prediction",   "")

        a  = anls(pred, gt)
        em = exact_match(pred, gt)
        l  = rec.get("latency", 0.0)
        latencies.append(l)

        # Accumulate
        for store in (overall, by_domain[rec.get("domain", "unknown")],
                      by_source[rec.get("answer_source", "unknown")]):
            store["anls_sum"] += a
            store["em_sum"]   += em
            store["n"]        += 1

        if a == 0.0:
            failures.append(rec)

    if skipped:
        logging.getLogger("poc1").info(f"[analyzer] ⚠ Skipped {skipped} error records")

    report = {
        "overall":       _finalize_stats(overall),
        "by_domain":     {k: _finalize_stats(v) for k, v in sorted(by_domain.items())},
        "by_source":     {k: _finalize_stats(v) for k, v in sorted(by_source.items())},
        "latency":       _latency_stats(latencies),
        "failure_cases": failures,
    }

    logging.getLogger("poc1").info(
        f"[analyzer] ✔ ANLS={report['overall']['anls']:.1f}%  "
        f"EM={report['overall']['exact_match']:.1f}%  "
        f"n={report['overall']['n']}  "
        f"failures={len(failures)}"
    )
    return report

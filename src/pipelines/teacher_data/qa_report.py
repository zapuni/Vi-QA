"""
src/pipelines/teacher_data/qa_report.py
========================================
Tạo báo cáo QA cho run: phân bố profile, bbox coverage,
PoT rate, reasoning length, v.v.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

log = logging.getLogger("teacher_data")


def generate_qa_report(run_dir: str | Path) -> dict:
    """
    Tạo báo cáo QA cho một run.

    Args:
        run_dir: Path tới run directory chứa shards/.

    Returns:
        Dict report với stats + metrics.
    """
    run_dir = Path(run_dir)
    shards_dir = run_dir / "shards"

    if not shards_dir.exists():
        log.error(f"[qa_report] Shards dir not found: {shards_dir}")
        return {}

    total = 0
    profile_dist: Counter = Counter()
    reasoning_lengths: list[int] = []
    bbox_missing = 0
    pot_present = 0

    for shard_file in sorted(shards_dir.glob("*.jsonl")):
        with open(shard_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                total += 1
                profile_dist[record.get("output_profile", "unknown")] += 1

                reasoning = record.get("reasoning")
                if reasoning:
                    reasoning_lengths.append(len(reasoning))

                if not record.get("grounding_boxes"):
                    bbox_missing += 1

                if record.get("python_code"):
                    pot_present += 1

    report = {
        "run_dir": str(run_dir),
        "total_samples": total,
        "profile_distribution": dict(profile_dist),
        "avg_reasoning_length": (
            round(sum(reasoning_lengths) / len(reasoning_lengths), 1)
            if reasoning_lengths
            else 0.0
        ),
        "bbox_coverage": (
            round((total - bbox_missing) / total * 100, 1) if total > 0 else 0.0
        ),
        "bbox_missing_count": bbox_missing,
        "pot_present_rate": round(pot_present / total * 100, 1) if total > 0 else 0.0,
        "pot_present_count": pot_present,
    }

    # Load progress stats
    progress_file = run_dir / "checkpoints" / "progress.json"
    if progress_file.exists():
        with open(progress_file, encoding="utf-8") as f:
            progress = json.load(f)
        report["checkpoint_stats"] = progress.get("stats", {})

    # Ghi report
    report_dir = run_dir / "reports"
    report_dir.mkdir(exist_ok=True)
    report_path = report_dir / "qa_summary.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    log.info(f"[qa_report] Report saved: {report_path}")
    log.info(f"[qa_report] Total: {total}, BBox coverage: {report['bbox_coverage']}%, "
             f"PoT rate: {report['pot_present_rate']}%")

    return report

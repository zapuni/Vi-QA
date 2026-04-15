"""
modules/reporting/table.py
===========================
Trách nhiệm DUY NHẤT: Tổng hợp reports của tất cả models thành
bảng so sánh Markdown và lưu file.

Input:  dict {model_key: report_dict}  (từ analyzer.analyze())
Output: comparison_table.md + comparison_report.json
"""

from __future__ import annotations

import logging

import json
from pathlib import Path


def _format_row(model_key: str, report: dict) -> str:
    """Tạo 1 dòng trong bảng Markdown."""
    o   = report.get("overall", {})
    lat = report.get("latency", {})
    return (
        f"| {model_key:<20} "
        f"| {o.get('anls', 0):>8.1f} "
        f"| {o.get('exact_match', 0):>6.1f} "
        f"| {o.get('n', 0):>7} "
        f"| {lat.get('mean', 0):>12.3f} "
        f"| {lat.get('p95', 0):>11.3f} |"
    )


def build_summary_table(reports: dict[str, dict]) -> str:
    """
    Tạo bảng Markdown so sánh tất cả models.

    Args:
        reports: {model_key: analyzer report dict}

    Returns:
        Chuỗi Markdown table.
    """
    lines = [
        "# POC-1: VLM Benchmark Results\n",
        "| Model                | ANLS (%) | EM (%) | Samples | Lat mean (s) | Lat p95 (s) |",
        "|----------------------|----------|--------|---------|--------------|-------------|",
    ]
    for model_key in sorted(reports.keys()):
        lines.append(_format_row(model_key, reports[model_key]))

    # Thêm breakdown section nếu có nhiều hơn 1 model
    if len(reports) > 1:
        lines.append("\n## Breakdown by Answer Source\n")
        lines.append("| Model                | Answer Source      | ANLS (%) | EM (%) | n      |")
        lines.append("|----------------------|--------------------|----------|--------|--------|")
        for model_key in sorted(reports.keys()):
            by_src = reports[model_key].get("by_source", {})
            for src, stats in sorted(by_src.items()):
                lines.append(
                    f"| {model_key:<20} "
                    f"| {src:<18} "
                    f"| {stats.get('anls', 0):>8.1f} "
                    f"| {stats.get('exact_match', 0):>6.1f} "
                    f"| {stats.get('n', 0):>6} |"
                )

    return "\n".join(lines)


def save_report(reports: dict[str, dict], out_dir: str) -> None:
    """
    Lưu:
      - comparison_report.json  (full data, dùng lập trình sau)
      - comparison_table.md     (dễ đọc, copy vào báo cáo)

    Args:
        reports:  {model_key: report dict}
        out_dir:  thư mục output (tự tạo nếu chưa có)
    """
    p = Path(out_dir)
    p.mkdir(parents=True, exist_ok=True)

    # 1. JSON dump (bỏ failure_cases để file nhỏ hơn)
    slim_reports = {
        k: {kk: vv for kk, vv in v.items() if kk != "failure_cases"}
        for k, v in reports.items()
    }
    json_path = p / "comparison_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(slim_reports, f, indent=2, ensure_ascii=False)

    # 2. Markdown table
    table = build_summary_table(reports)
    md_path = p / "comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")

    # 3. In ra stdout để thấy ngay
    logging.getLogger("poc1").info("\n" + table)
    logging.getLogger("poc1").info(f"\n[reporting] ✔ Saved → {json_path}")
    logging.getLogger("poc1").info(f"[reporting] ✔ Saved → {md_path}")

#!/usr/bin/env python3
"""
run_benchmark.py — Benchmark VLMs trên ViInfographicVQA
=======================================================

Entry point duy nhất. Ghép các module lại theo thứ tự pipeline:
    Step 1: Load data          (src/data/loader.py)
    Step 2: Batch inference    (src/inference/runner.py)
    Step 3: Evaluate           (src/evaluation/analyzer.py)
    Step 4: Report             (src/reporting/table.py + charts.py)

Cách chạy:
    # Dev mode (20 mẫu, 1 model — test pipeline nhanh)
    python run_benchmark.py --dev --models vintern-1b

    # Chạy 1 model cụ thể (full dataset)
    python run_benchmark.py --models qwen25vl-7b

    # Chạy tất cả models enabled trong config
    python run_benchmark.py

    # Đã có predictions rồi, chỉ evaluate + report lại
    python run_benchmark.py --skip-inference

    # Dùng config khác
    python run_benchmark.py --config configs/my_config.yaml
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import yaml

# ── sys.path: đảm bảo import src/ từ thư mục project root ───────────
_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

from src.data       import loader, sampler
from src.inference  import runner
from src.evaluation import analyzer
from src.reporting  import table, charts


# ── Logging ──────────────────────────────────────────────────────────

def _setup_logging(log_dir: str) -> None:
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"run_benchmark_{ts}.log"

    # Định dạng chung cho log
    log_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S"
    )

    # 1. File Handler (ghi mọi thứ từ INFO trở lên)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(log_formatter)

    # 2. Console Handler (chỉ hiện WARNING và ERROR ra terminal)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.WARNING)
    console_handler.setFormatter(log_formatter)

    # Reset root logger với cấu hình mới
    root_log = logging.getLogger()
    root_log.setLevel(logging.INFO)
    # Xóa handler cũ nếu có để tránh log đôi
    if root_log.hasHandlers():
        root_log.handlers.clear()
        
    root_log.addHandler(file_handler)
    root_log.addHandler(console_handler)

    logging.info(f"Log file: {log_file}")


# ── Config ────────────────────────────────────────────────────────────

def _load_config(path: str) -> dict:
    cfg_path = Path(path)
    if not cfg_path.exists():
        sys.exit(f"[benchmark] Config file not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── CLI ───────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Benchmark VLMs trên ViInfographicVQA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--config",
        default=str(_ROOT / "configs" / "benchmark.yaml"),
        help="Đường dẫn file config YAML (default: configs/benchmark.yaml)",
    )
    p.add_argument(
        "--models",
        nargs="+",
        default=None,
        metavar="MODEL_KEY",
        help="Chỉ chạy các model keys này (vd: --models vintern-1b qwen25vl-7b)",
    )
    p.add_argument(
        "--dev",
        action="store_true",
        help=f"Dev mode: chỉ dùng N mẫu nhỏ (mặc định: dev_samples trong config)",
    )
    p.add_argument(
        "--skip-inference",
        action="store_true",
        help="Bỏ qua bước inference, dùng predictions.jsonl đã có để evaluate",
    )
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────

def _section(log: logging.Logger, title: str) -> None:
    log.info("=" * 65)
    log.info(f"  {title}")
    log.info("=" * 65)


def _save_failure_csv(records: list[dict], out_path: Path) -> None:
    """Lưu failure cases ra CSV để phân tích thủ công sau."""
    if not records:
        return
    fields = ["id", "question", "ground_truth", "prediction",
              "domain", "answer_source", "latency"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


# ── Main pipeline ─────────────────────────────────────────────────────

def main() -> int:
    args = _parse_args()
    cfg  = _load_config(args.config)

    # Áp dụng override từ CLI
    if args.dev:
        cfg["dev_mode"] = True
    if args.models:
        for k in cfg["models"]:
            cfg["models"][k]["enabled"] = (k in args.models)

    _setup_logging(cfg["log_dir"])
    log = logging.getLogger("poc1")

    log.info("Benchmark START — VLM Benchmark on ViInfographicVQA")
    log.info(f"Config: {args.config}")
    log.info(f"Dev mode: {cfg.get('dev_mode', False)}")

    # ────────────────────────────────────────────────────────────────
    # STEP 1 — Load Dataset
    # ────────────────────────────────────────────────────────────────
    _section(log, "STEP 1 — Load Dataset")
    samples = loader.load(cfg)

    if cfg.get("dev_mode"):
        n = cfg.get("dev_samples", 20)
        samples = sampler.sample(samples, n, cfg.get("seed", 42))
        log.info(f"[dev_mode] Using {len(samples)} samples")

    # ────────────────────────────────────────────────────────────────
    # STEP 2 — Inference
    # ────────────────────────────────────────────────────────────────
    enabled = {k: v for k, v in cfg["models"].items() if v.get("enabled", False)}
    if not enabled:
        log.error("No models enabled in config. Set enabled: true for at least one model.")
        return 1

    log.info(f"Models enabled: {list(enabled.keys())}")
    jsonl_paths: dict[str, str] = {}

    if not args.skip_inference:
        _section(log, "STEP 2 — Batch Inference")
        for model_key, model_cfg in enabled.items():
            log.info(f"\n>>> Inference: {model_key} ({model_cfg['hf_id']})")
            try:
                path = runner.run(model_key, model_cfg, samples, cfg)
                jsonl_paths[model_key] = path
            except Exception as exc:
                log.error(f"[{model_key}] FAILED during inference: {exc}", exc_info=True)
                # Tiếp tục với model tiếp theo thay vì dừng hẳn
    else:
        log.info("[skip-inference] Loading existing prediction files…")
        for model_key in enabled:
            p = Path(cfg["output_dir"]) / model_key / "predictions.jsonl"
            if p.exists():
                jsonl_paths[model_key] = str(p)
                log.info(f"  ✔ Found: {p}")
            else:
                log.warning(f"  ✗ Not found: {p} (skipping {model_key})")

    if not jsonl_paths:
        log.error("No prediction files available for evaluation.")
        return 1

    # ────────────────────────────────────────────────────────────────
    # STEP 3 — Evaluation
    # ────────────────────────────────────────────────────────────────
    _section(log, "STEP 3 — Evaluation & Error Analysis")
    reports: dict[str, dict] = {}

    for model_key, jsonl_path in jsonl_paths.items():
        log.info(f"\n>>> Evaluating: {model_key}")
        try:
            report = analyzer.analyze(jsonl_path)
            reports[model_key] = report

            # Lưu evaluation report riêng từng model
            model_dir = Path(cfg["output_dir"]) / model_key
            model_dir.mkdir(parents=True, exist_ok=True)

            eval_path = model_dir / "evaluation_report.json"
            slim = {k: v for k, v in report.items() if k != "failure_cases"}
            with open(eval_path, "w", encoding="utf-8") as f:
                json.dump(slim, f, indent=2, ensure_ascii=False)
            log.info(f"  ✔ Saved evaluation report → {eval_path}")

            # Lưu failure cases
            fc = report.get("failure_cases", [])
            if fc:
                fc_path = model_dir / "failure_cases.csv"
                _save_failure_csv(fc, fc_path)
                log.info(f"  ✔ {len(fc)} failure cases → {fc_path}")

        except Exception as exc:
            log.error(f"[{model_key}] Evaluation failed: {exc}", exc_info=True)

    if not reports:
        log.error("No evaluation results to report.")
        return 1

    # ────────────────────────────────────────────────────────────────
    # STEP 4 — Reporting
    # ────────────────────────────────────────────────────────────────
    _section(log, "STEP 4 — Reporting & Visualization")
    summary_dir = str(Path(cfg["output_dir"]) / "_summary")

    table.save_report(reports, summary_dir)
    charts.generate_all(reports, summary_dir)

    # ────────────────────────────────────────────────────────────────
    # DONE
    # ────────────────────────────────────────────────────────────────
    log.info("\n" + "=" * 65)
    log.info("  ✅  BENCHMARK COMPLETE")
    log.info("=" * 65)
    log.info(f"Results saved in: {cfg['output_dir']}/")
    log.info(f"Summary:          {summary_dir}/comparison_table.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())

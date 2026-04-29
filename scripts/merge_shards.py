#!/usr/bin/env python3
"""
scripts/merge_shards.py
=========================
Merge tất cả shard .jsonl thành một file duy nhất.

Cách chạy:
    python scripts/merge_shards.py \
        --run-dir data/teacher_runs/run_20260427_qwen32b \
        --output data/teacher_runs/run_20260427_qwen32b/merged_output.jsonl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge shard files")
    parser.add_argument("--run-dir", type=str, required=True, help="Path tới run dir")
    parser.add_argument("--output", type=str, default=None, help="Output file path")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    shards_dir = run_dir / "shards"

    if not shards_dir.exists():
        print(f"❌ Shards dir not found: {shards_dir}")
        return 1

    shard_files = sorted(shards_dir.glob("shard_*.jsonl"))
    if not shard_files:
        print(f"❌ No shard files found in {shards_dir}")
        return 1

    output_path = Path(args.output) if args.output else run_dir / "merged_output.jsonl"

    total = 0
    with open(output_path, "w", encoding="utf-8") as fout:
        for shard in shard_files:
            with open(shard, encoding="utf-8") as fin:
                for line in fin:
                    line = line.strip()
                    if line:
                        fout.write(line + "\n")
                        total += 1
            print(f"  ✔ Merged {shard.name}")

    print(f"\n✅ Merged {total} records → {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""
scripts/run_verify_gui.py
==========================
Launcher script cho Human Verification GUI.

Usage:
    # Auto-detect latest run:
    python scripts/run_verify_gui.py

    # Chỉ định run directory:
    python scripts/run_verify_gui.py --run-dir data/teacher_runs_20/run_xxx

    # Chỉ định port (mặc định 8501):
    python scripts/run_verify_gui.py --port 8502
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def find_latest_run() -> Path | None:
    """Tìm run directory mới nhất."""
    for search_dir in ["data/teacher_runs_20", "data/teacher_runs"]:
        p = PROJECT_ROOT / search_dir
        if p.exists():
            runs = sorted(
                [d for d in p.iterdir() if d.is_dir() and (d / "shards").exists()],
                key=lambda d: d.name,
            )
            if runs:
                return runs[-1]
    return None


def main():
    parser = argparse.ArgumentParser(
        description="Launch Human Verification GUI for teacher data"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to teacher run directory (containing shards/)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port for Streamlit server (default: 8501)",
    )
    args = parser.parse_args()

    # Resolve run dir
    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        run_dir = find_latest_run()

    if not run_dir or not run_dir.exists():
        print("❌ Không tìm thấy run directory!")
        print("   Truyền --run-dir hoặc đảm bảo có data/teacher_runs*/run_*/shards/")
        sys.exit(1)

    print(f"🔍 Run dir: {run_dir}")
    print(f"🌐 Starting GUI on port {args.port}...")
    print(f"   URL: http://localhost:{args.port}")
    print()

    # Launch streamlit
    import shutil
    app_path = PROJECT_ROOT / "src" / "gui" / "verify" / "app.py"

    # Tìm streamlit executable
    streamlit_bin = shutil.which("streamlit")
    if streamlit_bin:
        cmd = [
            streamlit_bin, "run",
            str(app_path),
            "--server.port", str(args.port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--",  # Separator cho app args
            "--run-dir", str(run_dir),
        ]
    else:
        cmd = [
            sys.executable, "-m", "streamlit", "run",
            str(app_path),
            "--server.port", str(args.port),
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false",
            "--",  # Separator cho app args
            "--run-dir", str(run_dir),
        ]

    subprocess.run(cmd, cwd=str(PROJECT_ROOT))


if __name__ == "__main__":
    main()

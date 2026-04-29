"""
src/gui/verify/store.py
========================
Backend data store cho verification GUI.

Trách nhiệm:
  - Load JSONL shards từ teacher runs
  - Quản lý trạng thái verification (pending / approved / rejected / skipped)
  - Lưu/Load trạng thái vào verification_state.json (auto-save)
  - Export gold standard dataset

Trạng thái mỗi sample:
  {
    "status": "pending" | "approved" | "rejected" | "skipped",
    "edited_answer": str | None,     # Human override cho final_answer
    "edited_boxes": list | None,     # Human override cho grounding_boxes
    "note": str | None,              # Ghi chú của reviewer
    "reviewed_at": str | None,       # ISO timestamp
    "reviewer": str | None,          # Tên reviewer
  }
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

log = logging.getLogger("verify_gui")

# Verification statuses
STATUS_PENDING = "pending"
STATUS_APPROVED = "approved"
STATUS_REJECTED = "rejected"
STATUS_SKIPPED = "skipped"

ALL_STATUSES = [STATUS_PENDING, STATUS_APPROVED, STATUS_REJECTED, STATUS_SKIPPED]


class VerifyStore:
    """
    Quản lý dữ liệu verification.

    - records: list[dict]  — tất cả teacher outputs (đọc từ JSONL shards)
    - state: dict[str, dict]  — trạng thái verify per sample_id
    """

    def __init__(
        self,
        run_dir: str | Path,
        data_root: str | Path = "data/ViInfographicVQA",
    ) -> None:
        self.run_dir = Path(run_dir)
        self.data_root = Path(data_root)
        self.shard_dir = self.run_dir / "shards"
        self.state_path = self.run_dir / "verification_state.json"
        self.gold_path = self.run_dir / "gold_standard.jsonl"

        # Load data
        self.records: list[dict] = self._load_shards()

        # Enrich với question/answer từ dataset gốc
        self._enrich_from_dataset()

        self.state: dict[str, dict] = self._load_state()

        # Index for fast lookup
        self._id_to_idx: dict[str, int] = {
            r["sample_id"]: i for i, r in enumerate(self.records)
        }

        log.info(
            f"[store] Loaded {len(self.records)} records, "
            f"{self.count_by_status(STATUS_APPROVED)} approved, "
            f"{self.count_by_status(STATUS_PENDING)} pending"
        )

    # ── Load ─────────────────────────────────────────────────────────

    def _load_shards(self) -> list[dict]:
        """Load tất cả JSONL shards, sorted theo sample_id."""
        records = []
        if not self.shard_dir.exists():
            log.warning(f"[store] Shard dir not found: {self.shard_dir}")
            return records

        for shard_file in sorted(self.shard_dir.glob("shard_*.jsonl")):
            with open(shard_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            records.append(json.loads(line))
                        except json.JSONDecodeError as e:
                            log.warning(f"[store] Bad JSON in {shard_file.name}: {e}")
        return records

    def _enrich_from_dataset(self) -> None:
        """
        Enrich records với question/answer từ dataset gốc.

        Teacher JSONL shards chỉ lưu sample_id, final_answer, reasoning, ...
        nhưng KHÔNG lưu question gốc và ground-truth answer.
        Method này load từ original JSON files (single_*.json, multi_*.json)
        và merge vào records.
        """
        if not self.data_root.exists():
            log.warning(f"[store] Data root not found: {self.data_root}, skip enrich")
            return

        # Build lookup: question_id → original record
        lookup: dict[str, dict] = {}
        data_dir = self.data_root / "data"
        if data_dir.exists():
            for jf in sorted(data_dir.glob("*.json")):
                try:
                    with open(jf, encoding="utf-8") as f:
                        items = json.load(f)
                    for item in items:
                        qid = str(item.get("question_id", ""))
                        if qid:
                            lookup[qid] = item
                except Exception as e:
                    log.warning(f"[store] Error loading {jf.name}: {e}")

        if not lookup:
            log.warning("[store] No dataset records found for enrichment")
            return

        enriched = 0
        for record in self.records:
            sid = record["sample_id"]
            if sid in lookup and "question" not in record:
                orig = lookup[sid]
                record["question"] = orig.get("question", "")
                record["answer"] = orig.get("answer", "")
                record["image_type"] = orig.get("image_type", "unknown")
                record["answer_source"] = orig.get("answer_source", "unknown")
                enriched += 1

        log.info(f"[store] Enriched {enriched}/{len(self.records)} records from dataset")

    def _load_state(self) -> dict[str, dict]:
        """Load verification state, khởi tạo pending cho records mới."""
        state: dict[str, dict] = {}

        if self.state_path.exists():
            with open(self.state_path, encoding="utf-8") as f:
                state = json.load(f)

        # Đảm bảo mỗi record đều có state
        for r in self.records:
            sid = r["sample_id"]
            if sid not in state:
                state[sid] = self._default_state()

        return state

    @staticmethod
    def _default_state() -> dict:
        return {
            "status": STATUS_PENDING,
            "edited_answer": None,
            "edited_boxes": None,
            "note": None,
            "reviewed_at": None,
            "reviewer": None,
        }

    # ── Save ─────────────────────────────────────────────────────────

    def save_state(self) -> None:
        """Atomic save verification state."""
        tmp = self.state_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)
        tmp.rename(self.state_path)

    # ── Query ────────────────────────────────────────────────────────

    def get_record(self, sample_id: str) -> dict | None:
        idx = self._id_to_idx.get(sample_id)
        return self.records[idx] if idx is not None else None

    def get_state(self, sample_id: str) -> dict:
        return self.state.get(sample_id, self._default_state())

    def count_by_status(self, status: str) -> int:
        return sum(1 for s in self.state.values() if s["status"] == status)

    def get_ids_by_status(self, status: str | None = None) -> list[str]:
        """Trả về danh sách sample_ids, lọc theo status nếu cần."""
        if status is None:
            return [r["sample_id"] for r in self.records]
        return [
            r["sample_id"] for r in self.records
            if self.state.get(r["sample_id"], {}).get("status") == status
        ]

    def get_stats(self) -> dict[str, int]:
        """Thống kê tổng quan."""
        stats = {s: 0 for s in ALL_STATUSES}
        for s in self.state.values():
            status = s.get("status", STATUS_PENDING)
            stats[status] = stats.get(status, 0) + 1
        stats["total"] = len(self.records)
        return stats

    # ── Update ───────────────────────────────────────────────────────

    def update_status(
        self,
        sample_id: str,
        status: str,
        edited_answer: str | None = None,
        edited_boxes: list | None = None,
        note: str | None = None,
        reviewer: str = "anonymous",
    ) -> None:
        """Cập nhật trạng thái verify cho sample."""
        self.state[sample_id] = {
            "status": status,
            "edited_answer": edited_answer,
            "edited_boxes": edited_boxes,
            "note": note,
            "reviewed_at": datetime.now(timezone.utc).isoformat(),
            "reviewer": reviewer,
        }
        # Auto-save ngay sau mỗi action
        self.save_state()

    # ── Export Gold Standard ──────────────────────────────────────────

    def export_gold(self) -> int:
        """
        Export tất cả approved records thành gold_standard.jsonl.

        Nếu sample có edited_answer hoặc edited_boxes, sẽ dùng bản chỉnh sửa.
        Trả về số records đã export.
        """
        count = 0
        with open(self.gold_path, "w", encoding="utf-8") as f:
            for record in self.records:
                sid = record["sample_id"]
                st = self.state.get(sid, {})
                if st.get("status") != STATUS_APPROVED:
                    continue

                # Build gold record (override nếu human đã sửa)
                gold = {**record}

                # Loại bỏ raw_response để giảm size
                gold.pop("raw_response", None)

                # Override nếu có chỉnh sửa
                if st.get("edited_answer"):
                    gold["final_answer"] = st["edited_answer"]
                if st.get("edited_boxes"):
                    gold["grounding_boxes"] = st["edited_boxes"]

                # Ghi nhận metadata verify
                gold["verification"] = {
                    "status": "approved",
                    "reviewer": st.get("reviewer", "anonymous"),
                    "reviewed_at": st.get("reviewed_at"),
                    "note": st.get("note"),
                }

                f.write(json.dumps(gold, ensure_ascii=False) + "\n")
                count += 1

        log.info(f"[store] Exported {count} gold records → {self.gold_path}")
        return count

"""
src/pipelines/teacher_data/parser.py
=====================================
Parse raw response từ teacher model thành structured dict.
Hỗ trợ:
  - Extract reasoning từ reasoning_content (vLLM Thinking mode)
  - Parse XML tags từ content
  - Parse bounding boxes:
      Single-image: [x1,y1,x2,y2]           — 4 phần tử
      Multi-image:  [img_idx,x1,y1,x2,y2]   — 5 phần tử
  - Parse evidence items cho multi_span / multi_image_spans
"""

from __future__ import annotations

import ast
import json
import logging
import re
from typing import Any

log = logging.getLogger("teacher_data")


class ResponseParser:
    """Parse raw response từ teacher model."""

    def parse(
        self,
        sample_id: str,
        raw_response: dict,
        output_profile: str,
    ) -> dict[str, Any]:
        """Parse response thành structured data."""
        message = raw_response["choices"][0]["message"]

        # vLLM Thinking mode: reasoning nằm trong reasoning_content
        reasoning = self._extract_reasoning(message)
        content = message.get("content", "") or ""

        parsed: dict[str, Any] = {
            "sample_id": sample_id,
            "output_profile": output_profile,
            "reasoning": reasoning or self._extract_tag(content, "reasoning"),
            "python_code": self._extract_tag(content, "python_code"),
            "grounding_boxes": self._parse_boxes(
                self._extract_tag(content, "grounding_boxes")
            ),
            "evidence_text": self._extract_tag(content, "evidence_text"),
            "assumption_note": self._extract_tag(content, "assumption_note"),
            "missing_evidence_reason": self._extract_tag(
                content, "missing_evidence_reason"
            ),
            "final_answer": (
                self._extract_tag(content, "final_answer")
                or self._fallback_final_answer(content)
            ),
            "raw_response": raw_response,
        }

        # Parse evidence_items cho multi_span / multi_image_spans
        if output_profile in ("multi_span_v1", "multi_image_spans_v1"):
            parsed["evidence_items"] = self._parse_evidence_items(content)

        return parsed

    # ── Helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _extract_reasoning(message: dict) -> str | None:
        """Trích reasoning từ vLLM Thinking response."""
        return (
            message.get("reasoning")
            or message.get("reasoning_content")
        )

    @staticmethod
    def _extract_tag(text: str, tag: str) -> str | None:
        """Trích nội dung giữa <tag>...</tag>."""
        pattern = rf"<{tag}>\s*(.+?)\s*</{tag}>"
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        return match.group(1).strip() if match else None

    @staticmethod
    def _parse_boxes(boxes_str: str | None) -> list[list[int]]:
        """
        Parse bbox string thành list of int lists.

        Hỗ trợ cả 2 format:
          - Single: [[x1,y1,x2,y2], ...]             — 4 phần tử
          - Multi:  [[img_idx,x1,y1,x2,y2], ...]     — 5 phần tử
        """
        if not boxes_str:
            return []

        # Try ast.literal_eval first (handles nested lists)
        try:
            boxes = ast.literal_eval(boxes_str.strip())
            if isinstance(boxes, list):
                # Handle single box: [1,2,3,4] → [[1,2,3,4]]
                if boxes and isinstance(boxes[0], (int, float)):
                    boxes = [boxes]
                result = []
                for b in boxes:
                    if isinstance(b, (list, tuple)) and len(b) in (4, 5):
                        result.append([int(x) for x in b])
                if result:
                    return result
        except (ValueError, SyntaxError):
            pass

        # Fallback: regex tìm cả 4-element và 5-element patterns
        # 5-element: [img_idx, x1, y1, x2, y2]
        pattern_5 = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches_5 = re.findall(pattern_5, boxes_str)
        if matches_5:
            return [[int(a), int(b), int(c), int(d), int(e)]
                    for a, b, c, d, e in matches_5]

        # 4-element: [x1, y1, x2, y2]
        pattern_4 = r"\[(\d+),\s*(\d+),\s*(\d+),\s*(\d+)\]"
        matches_4 = re.findall(pattern_4, boxes_str)
        return [[int(a), int(b), int(c), int(d)] for a, b, c, d in matches_4]

    @staticmethod
    def _parse_evidence_items(text: str) -> list[dict[str, Any]]:
        """Parse evidence items cho multi_span / multi_image_spans profile."""
        raw = ResponseParser._extract_tag(text, "evidence_items")
        if not raw:
            return []

        try:
            items = json.loads(raw)
            if isinstance(items, list):
                return items
        except json.JSONDecodeError:
            pass

        try:
            items = ast.literal_eval(raw)
            if isinstance(items, list):
                return items
        except (ValueError, SyntaxError):
            pass

        return []

    @staticmethod
    def _fallback_final_answer(content: str) -> str:
        """
        Fallback khi không tìm thấy <final_answer> tag.
        Trả về toàn bộ content đã stripped (tốt hơn trả empty string).
        """
        # Strip all XML tags
        cleaned = re.sub(r"<[^>]+>", "", content).strip()
        # Lấy dòng cuối cùng non-empty (thường là đáp án)
        lines = [l.strip() for l in cleaned.split("\n") if l.strip()]
        return lines[-1] if lines else content.strip()

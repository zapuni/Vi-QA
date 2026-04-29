"""
src/pipelines/teacher_data/planner.py
=====================================
Quyết định output_profile và question_type cho từng sample
dựa trên answer_source, số lượng ảnh, và nội dung câu hỏi.

Hỗ trợ:
  - Single-image: image-span, question-span, multi-span, non-extractive
  - Multi-image:  Cross-Image Synthesis, Multi-Image Spans, Non-Span
"""

from __future__ import annotations

from src.pipelines.teacher_data.models import RawSample, OutputProfile, QuestionType


class TaskPlanner:
    """Quyết định output profile cho từng sample."""

    # ── Single-image answer_source mapping ────────────────────────────
    # answer_source trong dataset dùng dấu gạch ngang: "image-span", "multi-span", ...
    SINGLE_PROFILE_MAP: dict[str, OutputProfile] = {
        "image-span": "image_span_v1",
        "question-span": "question_span_v1",
        "multi-span": "multi_span_v1",
        "non-extractive": "reasoning_math_v1",
    }

    # ── Multi-image answer_source mapping ─────────────────────────────
    # Multi-image dataset dùng format khác: "Cross-Image Synthesis", etc.
    MULTI_PROFILE_MAP: dict[str, OutputProfile] = {
        "Cross-Image Synthesis": "cross_image_v1",
        "Multi-Image Spans": "multi_image_spans_v1",
        "Non-Span": "cross_image_v1",  # Non-Span vẫn cần cross-image reasoning
    }

    MATH_KEYWORDS = [
        "tính", "bao nhiêu", "tổng", "hiệu", "tỉ lệ", "phần trăm",
        "trung bình", "cao hơn", "thấp hơn", "chênh lệch", "gấp",
        "giảm", "tăng", "so với",
    ]

    HYPOTHETICAL_KEYWORDS = [
        "nếu", "giả sử", "nếu như", "trong trường hợp",
        "giả định", "nếu giả sử",
    ]

    def plan(self, sample: RawSample) -> dict:
        """Trả về output_profile và question_type cho sample."""
        answer_source = sample.answer_source or "unknown"
        question_lower = sample.question.lower()

        # ── Multi-image: ưu tiên xử lý theo logic riêng ─────────────
        if sample.is_multi_image:
            return self._plan_multi_image(sample, answer_source, question_lower)

        # ── Single-image flow (giữ nguyên logic cũ) ─────────────────
        # Hypothetical detection
        if self._is_hypothetical(question_lower):
            return {
                "output_profile": "hypothetical_v1",
                "question_type": "hypothetical",
            }

        # Unanswerable detection
        if self._is_unanswerable(sample):
            return {
                "output_profile": "unanswerable_v1",
                "question_type": "unanswerable",
            }

        # Non-extractive with math → reasoning_math
        if answer_source == "non-extractive" and self._requires_math(question_lower):
            return {
                "output_profile": "reasoning_math_v1",
                "question_type": "reasoning_math",
            }

        # Default mapping từ answer_source
        profile = self.SINGLE_PROFILE_MAP.get(answer_source, "reasoning_math_v1")
        qtype = self._infer_question_type(answer_source)

        return {
            "output_profile": profile,
            "question_type": qtype,
        }

    def _plan_multi_image(
        self, sample: RawSample, answer_source: str, question_lower: str
    ) -> dict:
        """Plan cho multi-image samples."""
        # Map answer_source → profile
        profile = self.MULTI_PROFILE_MAP.get(answer_source, "cross_image_v1")

        # Infer question type
        if answer_source == "Cross-Image Synthesis":
            qtype: QuestionType = "cross_image_synthesis"
        elif answer_source == "Multi-Image Spans":
            qtype = "multi_image_spans"
        elif self._requires_math(question_lower):
            qtype = "cross_image_synthesis"
            profile = "cross_image_v1"  # Math cross-image vẫn dùng reasoning
        else:
            qtype = "cross_image_synthesis"

        return {
            "output_profile": profile,
            "question_type": qtype,
        }

    def _is_hypothetical(self, question_lower: str) -> bool:
        return any(kw in question_lower for kw in self.HYPOTHETICAL_KEYWORDS)

    def _is_unanswerable(self, sample: RawSample) -> bool:
        if sample.answer and "không thể" in sample.answer.lower():
            return True
        return False

    def _requires_math(self, question_lower: str) -> bool:
        return any(kw in question_lower for kw in self.MATH_KEYWORDS)

    def _infer_question_type(self, answer_source: str) -> QuestionType:
        if answer_source in ("image-span", "question-span"):
            return "extractive"
        elif answer_source == "multi-span":
            return "multi_span"
        else:
            return "reasoning_math"

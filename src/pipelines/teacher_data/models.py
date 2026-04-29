"""
src/pipelines/teacher_data/models.py
=====================================
Pydantic schema models + Output Profile Contract.
Đảm bảo đồng bộ schema cho tất cả các loại câu hỏi,
bao gồm cả Single-image và Multi-image.

Bbox format:
  - Single-image: [x1, y1, x2, y2]           — 4 phần tử, normalized [0-1000]
  - Multi-image:  [img_idx, x1, y1, x2, y2]  — 5 phần tử, img_idx bắt đầu từ 0
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


# ── Type aliases ─────────────────────────────────────────────────────

QuestionType = Literal[
    "extractive", "multi_span", "reasoning_math",
    "hypothetical", "unanswerable",
    # Multi-image specific
    "cross_image_synthesis", "multi_image_spans",
]

OutputProfile = Literal[
    # Single-image profiles
    "image_span_v1", "question_span_v1", "multi_span_v1",
    "reasoning_math_v1", "hypothetical_v1", "unanswerable_v1",
    # Multi-image profiles
    "cross_image_v1", "multi_image_spans_v1",
]


# ── Dataclasses cho internal pipeline ────────────────────────────────

@dataclass
class RawSample:
    """Raw sample từ ViInfographicVQA dataset"""
    sample_id: str
    image_paths: list[str]
    question: str
    answer: str | None
    answer_source: str | None  # 'image-span', 'Cross-Image Synthesis', etc.
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_images(self) -> int:
        return len(self.image_paths)

    @property
    def is_multi_image(self) -> bool:
        return len(self.image_paths) > 1


@dataclass
class TeacherRequest:
    """Request cho teacher model"""
    sample: RawSample
    question_type: QuestionType
    output_profile: OutputProfile
    system_prompt: str
    user_prompt: str


# ── Pydantic model cho validated output ──────────────────────────────

class TeacherOutput(BaseModel):
    """Output từ teacher model - validated theo profile"""
    sample_id: str
    model_name: str
    output_profile: OutputProfile

    # Core fields
    final_answer: str
    num_images: int = 1  # Số ảnh trong sample (để audit)
    image_paths: list[str] = Field(default_factory=list)  # Đường dẫn ảnh để map img_idx

    # Optional fields theo profile
    reasoning: str | None = None
    python_code: str | None = None
    grounding_boxes: list[list[int]] = Field(default_factory=list)
    evidence_text: str | None = None
    evidence_items: list[dict[str, Any]] = Field(default_factory=list)
    assumption_note: str | None = None
    missing_evidence_reason: str | None = None

    # Metadata
    raw_response: dict[str, Any] = Field(default_factory=dict)
    validation: dict[str, bool] = Field(default_factory=dict)
    run_meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("grounding_boxes")
    @classmethod
    def validate_bbox_format(cls, v: list[list[int]]) -> list[list[int]]:
        """
        Kiểm tra bbox format:
          - Single-image: [x1, y1, x2, y2]            — 4 phần tử, coords [0-1000]
          - Multi-image:  [img_idx, x1, y1, x2, y2]   — 5 phần tử, img_idx >= 0
        """
        for box in v:
            if len(box) == 4:
                # Single-image bbox
                if not all(0 <= coord <= 1000 for coord in box):
                    raise ValueError(f"Bbox coords phải normalized [0-1000]: {box}")
            elif len(box) == 5:
                # Multi-image bbox: [img_idx, x1, y1, x2, y2]
                img_idx = box[0]
                coords = box[1:]
                if img_idx < 0:
                    raise ValueError(f"img_idx phải >= 0: {box}")
                if not all(0 <= c <= 1000 for c in coords):
                    raise ValueError(f"Bbox coords phải normalized [0-1000]: {box}")
            else:
                raise ValueError(
                    f"Bbox phải có 4 phần tử [x1,y1,x2,y2] hoặc "
                    f"5 phần tử [img_idx,x1,y1,x2,y2], nhận được {len(box)}: {box}"
                )
        return v


# ── Output Profile Contract ──────────────────────────────────────────

OUTPUT_PROFILES: dict[str, dict[str, list[str]]] = {
    # ── Single-image profiles ────────────────────────────────────────
    "image_span_v1": {
        "required": ["final_answer", "grounding_boxes", "evidence_text"],
        "forbidden": ["python_code"],
    },
    "question_span_v1": {
        "required": ["final_answer", "grounding_boxes"],
        "forbidden": ["python_code"],
    },
    "multi_span_v1": {
        "required": ["final_answer", "grounding_boxes", "evidence_items"],
        "forbidden": [],
    },
    "reasoning_math_v1": {
        "required": ["reasoning", "python_code", "grounding_boxes", "final_answer"],
        "forbidden": [],
    },
    "hypothetical_v1": {
        "required": ["reasoning", "assumption_note", "final_answer"],
        "forbidden": [],
    },
    "unanswerable_v1": {
        "required": ["reasoning", "final_answer", "missing_evidence_reason"],
        "forbidden": ["python_code"],
    },
    # ── Multi-image profiles ─────────────────────────────────────────
    "cross_image_v1": {
        "required": ["reasoning", "grounding_boxes", "final_answer"],
        "forbidden": [],
    },
    "multi_image_spans_v1": {
        "required": ["final_answer", "grounding_boxes", "evidence_items"],
        "forbidden": [],
    },
}

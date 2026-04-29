"""
src/pipelines/teacher_data/prompt_builder.py
=============================================
Sinh system prompt và user prompt theo output profile.
Mỗi profile có format output riêng dùng XML tags.

Hỗ trợ:
  - Single-image: bbox format [x1, y1, x2, y2]
  - Multi-image:  bbox format [img_idx, x1, y1, x2, y2]
    + Cross-image reasoning prompt
    + Chỉ rõ dữ liệu nằm ở ảnh thứ mấy
"""

from __future__ import annotations

from src.pipelines.teacher_data.models import (
    OutputProfile,
    RawSample,
    TeacherRequest,
)


class PromptBuilder:
    """Sinh system prompt và user prompt theo output profile."""

    # ── System Prompt: Single-image ──────────────────────────────────

    _SYSTEM_SINGLE = """\
Bạn là AI Teacher chuyên phân tích Infographic tiếng Việt cho dự án ViInfographicVQA.
Nhiệm vụ: Trả lời câu hỏi về infographic với reasoning, visual grounding, và PoT khi cần.

Nguyên tắc:
1. Luôn suy nghĩ từng bước (step-by-step reasoning) trước khi kết luận.
2. Mọi thực thể/số liệu trích xuất từ ảnh PHẢI kèm tọa độ bounding box [x1, y1, x2, y2] normalized [0-1000].
3. Nếu cần tính toán, viết Python code (Program-of-Thought) để giải quyết.
4. Nếu không đủ thông tin, trả lời "không thể xác định" kèm lý do.
5. Ngôn ngữ: Tiếng Việt."""

    # ── System Prompt: Multi-image ───────────────────────────────────

    _SYSTEM_MULTI = """\
Bạn là AI Teacher chuyên phân tích Infographic tiếng Việt cho dự án ViInfographicVQA.
Bạn đang xem một bộ {num_images} ảnh Infographic liên quan đến nhau.

Nhiệm vụ:
1. Phân tích thông tin xuyên suốt các ảnh (Cross-image reasoning).
2. Khi trích xuất dữ liệu, hãy chỉ rõ dữ liệu đó nằm ở ảnh thứ mấy (ví dụ: "Giá trị A ở ảnh 1 là..., B ở ảnh 2 là...").
3. Luôn cung cấp Bounding Box kèm chỉ số ảnh: [index_ảnh, x1, y1, x2, y2] normalized [0-1000], trong đó index_ảnh bắt đầu từ 0.
4. Nếu cần tính toán (so sánh, tổng hợp số liệu xuyên ảnh), viết Python code (PoT).
5. Nếu không đủ thông tin, trả lời "không thể xác định" kèm lý do.
6. Ngôn ngữ: Tiếng Việt."""

    # ── Profile Suffixes ─────────────────────────────────────────────

    _PROFILE_SUFFIXES: dict[str, str] = {
        "image_span_v1": (
            "\n\nProfile: image_span_v1 — Trích xuất trực tiếp câu trả lời từ ảnh. "
            "Bắt buộc kèm bbox của vùng chứa đáp án và nội dung evidence_text."
        ),
        "question_span_v1": (
            "\n\nProfile: question_span_v1 — Câu trả lời nằm trong chính câu hỏi/options. "
            "Bắt buộc kèm bbox chỉ vùng question/option trên ảnh."
        ),
        "multi_span_v1": (
            "\n\nProfile: multi_span_v1 — Câu trả lời cần tổng hợp nhiều vùng evidence. "
            "Mỗi evidence item phải có bbox riêng."
        ),
        "reasoning_math_v1": (
            "\n\nProfile: reasoning_math_v1 — Bài toán cần suy luận + tính toán. "
            "Bắt buộc có reasoning, Python code (PoT), bbox của các số liệu đầu vào, và final answer."
        ),
        "hypothetical_v1": (
            "\n\nProfile: hypothetical_v1 — Câu hỏi giả thuyết. "
            "Bắt buộc có reasoning, ghi chú giả định (assumption_note), và final answer."
        ),
        "unanswerable_v1": (
            "\n\nProfile: unanswerable_v1 — Không thể trả lời. "
            "Bắt buộc giải thích lý do thiếu thông tin và refusal format."
        ),
        # Multi-image profiles
        "cross_image_v1": (
            "\n\nProfile: cross_image_v1 — Suy luận xuyên ảnh (Cross-Image Synthesis). "
            "Bắt buộc có reasoning, bbox kèm index ảnh [img_idx, x1, y1, x2, y2], "
            "và final answer. Nếu cần tính toán, bắt buộc có Python code."
        ),
        "multi_image_spans_v1": (
            "\n\nProfile: multi_image_spans_v1 — Trích xuất evidence từ nhiều ảnh. "
            "Mỗi evidence item phải kèm bbox kèm index ảnh [img_idx, x1, y1, x2, y2]. "
            "Bắt buộc có evidence_items."
        ),
    }

    # ── User Prompt Templates: Single-image ──────────────────────────

    _USER_TEMPLATES: dict[str, str] = {
        "image_span_v1": """\
Câu hỏi: {question}

Hãy trả lời theo format sau:
<reasoning>Suy nghĩ từng bước để tìm đáp án trên ảnh</reasoning>
<grounding_boxes>[[x1,y1,x2,y2], ...]</grounding_boxes>
<evidence_text>Nội dung trích xuất từ vùng bbox trên ảnh</evidence_text>
<final_answer>Đáp án cuối cùng</final_answer>""",

        "question_span_v1": """\
Câu hỏi: {question}

Hãy trả lời theo format sau:
<reasoning>Suy nghĩ từng bước</reasoning>
<grounding_boxes>[[x1,y1,x2,y2], ...]</grounding_boxes>
<final_answer>Đáp án cuối cùng</final_answer>""",

        "multi_span_v1": """\
Câu hỏi: {question}

Hãy trả lời theo format sau:
<reasoning>Suy nghĩ từng bước, liệt kê các vùng evidence</reasoning>
<grounding_boxes>[[x1,y1,x2,y2], ...]</grounding_boxes>
<evidence_items>[{{"text": "...", "bbox": [x1,y1,x2,y2]}}, ...]</evidence_items>
<final_answer>Đáp án cuối cùng</final_answer>""",

        "reasoning_math_v1": """\
Câu hỏi: {question}

Hãy trả lời theo format sau:
<reasoning>Phân tích bài toán từng bước, xác định các số liệu cần lấy từ ảnh</reasoning>
<grounding_boxes>[[x1,y1,x2,y2], ...]</grounding_boxes>
<python_code>
# Code Python tính toán kết quả
# Với mỗi con số lấy từ ảnh, comment bbox tương ứng
result = ...
print(result)
</python_code>
<final_answer>Đáp án cuối cùng</final_answer>""",

        "hypothetical_v1": """\
Câu hỏi: {question}

Hãy trả lời theo format sau:
<reasoning>Phân tích câu hỏi giả thuyết từng bước</reasoning>
<assumption_note>Ghi rõ các giả định đang sử dụng</assumption_note>
<final_answer>Đáp án cuối cùng</final_answer>""",

        "unanswerable_v1": """\
Câu hỏi: {question}

Hãy trả lời theo format sau:
<reasoning>Phân tích tại sao không thể trả lời được</reasoning>
<missing_evidence_reason>Liệt kê thông tin bị thiếu trên ảnh</missing_evidence_reason>
<final_answer>Không thể xác định vì...</final_answer>""",
    }

    # ── User Prompt Templates: Multi-image ───────────────────────────

    _USER_MULTI_TEMPLATES: dict[str, str] = {
        "cross_image_v1": """\
Bạn đang xem {num_images} ảnh (đánh số từ Ảnh 0 đến Ảnh {last_img_idx}).

Câu hỏi: {question}

Hãy trả lời theo format sau:
<reasoning>
Phân tích từng bước, chỉ rõ dữ liệu lấy từ ảnh nào:
- Ảnh 0: ...
- Ảnh 1: ...
So sánh/tổng hợp xuyên ảnh.
</reasoning>
<grounding_boxes>[[img_idx,x1,y1,x2,y2], ...]</grounding_boxes>
<python_code>
# Code Python tính toán kết quả (nếu cần)
# Comment: giá trị X từ Ảnh 0 bbox [0,x1,y1,x2,y2]
result = ...
print(result)
</python_code>
<final_answer>Đáp án cuối cùng</final_answer>""",

        "multi_image_spans_v1": """\
Bạn đang xem {num_images} ảnh (đánh số từ Ảnh 0 đến Ảnh {last_img_idx}).

Câu hỏi: {question}

Hãy trả lời theo format sau:
<reasoning>Suy nghĩ từng bước, chỉ rõ evidence từ ảnh nào</reasoning>
<grounding_boxes>[[img_idx,x1,y1,x2,y2], ...]</grounding_boxes>
<evidence_items>[{{"text": "...", "bbox": [img_idx,x1,y1,x2,y2], "image_index": 0}}, ...]</evidence_items>
<final_answer>Đáp án cuối cùng</final_answer>""",
    }

    # ── Public API ───────────────────────────────────────────────────

    def build(self, sample: RawSample, plan: dict) -> TeacherRequest:
        """Build TeacherRequest cho sample."""
        profile: OutputProfile = plan["output_profile"]
        qtype = plan["question_type"]

        system_prompt = self._build_system_prompt(sample, profile)
        user_prompt = self._build_user_prompt(sample, profile)

        return TeacherRequest(
            sample=sample,
            question_type=qtype,
            output_profile=profile,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _build_system_prompt(self, sample: RawSample, profile: OutputProfile) -> str:
        """Build system prompt, tự chọn single/multi base."""
        if sample.is_multi_image:
            base = self._SYSTEM_MULTI.format(num_images=sample.num_images)
        else:
            base = self._SYSTEM_SINGLE

        suffix = self._PROFILE_SUFFIXES.get(profile, "")
        return base + suffix

    def _build_user_prompt(self, sample: RawSample, profile: OutputProfile) -> str:
        """Build user prompt, tự chọn single/multi template."""
        if sample.is_multi_image and profile in self._USER_MULTI_TEMPLATES:
            template = self._USER_MULTI_TEMPLATES[profile]
            return template.format(
                question=sample.question,
                num_images=sample.num_images,
                last_img_idx=sample.num_images - 1,
            )
        else:
            template = self._USER_TEMPLATES.get(
                profile, self._USER_TEMPLATES["image_span_v1"]
            )
            return template.format(question=sample.question)

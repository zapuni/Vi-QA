"""
src/inference/adapters/qwen.py
===================================
Trách nhiệm DUY NHẤT: Wrap Qwen2.5-VL và Qwen3-VL theo interface VLMAdapter.

Khác biệt quan trọng giữa các phiên bản:
  - Qwen2-VL:   Qwen2VLForConditionalGeneration
  - Qwen2.5-VL: Qwen2_5_VLForConditionalGeneration  (class riêng!)
  - Qwen3-VL:   Qwen3VLForConditionalGeneration

Cả ba đều dùng AutoProcessor và qwen_vl_utils.process_vision_info
để xử lý image input.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from transformers import AutoProcessor

from src.inference.base import VLMAdapter

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

log = logging.getLogger("poc1")

# ── Prompt ───────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Answer the following question about the Vietnamese infographic "
    "concisely with a single term or short phrase. "
    "Do NOT explain. Just give the answer."
)


# ── Factory: load đúng model class tùy phiên bản Qwen ───────────────

def _load_qwen_model(hf_id: str):
    """
    Load Qwen2-VL, Qwen2.5-VL hoặc Qwen3-VL tùy theo tên model.

    Phân biệt model class dựa trên tên HF ID:
      - "Qwen3"   → Qwen3VLForConditionalGeneration
      - "Qwen2.5" → Qwen2_5_VLForConditionalGeneration
      - còn lại   → Qwen2VLForConditionalGeneration (Qwen2-VL gốc)
    """
    cls = None

    if "Qwen3" in hf_id:
        try:
            from transformers import Qwen3VLForConditionalGeneration
            cls = Qwen3VLForConditionalGeneration
            log.info(f"[QwenVLAdapter] Using Qwen3VLForConditionalGeneration")
        except ImportError:
            log.warning(
                "[QwenVLAdapter] Qwen3VL class not found in transformers, "
                "falling back to Qwen2.5-VL class"
            )

    if cls is None and "Qwen2.5" in hf_id:
        try:
            from transformers import Qwen2_5_VLForConditionalGeneration
            cls = Qwen2_5_VLForConditionalGeneration
            log.info(f"[QwenVLAdapter] Using Qwen2_5_VLForConditionalGeneration")
        except ImportError:
            log.warning(
                "[QwenVLAdapter] Qwen2_5_VL class not found in transformers, "
                "falling back to Qwen2VL class"
            )

    if cls is None:
        from transformers import Qwen2VLForConditionalGeneration
        cls = Qwen2VLForConditionalGeneration
        log.info(f"[QwenVLAdapter] Using Qwen2VLForConditionalGeneration")

    return cls.from_pretrained(
        hf_id,
        torch_dtype=torch.bfloat16,   # tương thích mọi phiên bản transformers
        device_map="auto",
        low_cpu_mem_usage=True,
    ).eval()


# ── Adapter class ────────────────────────────────────────────────────

class QwenVLAdapter(VLMAdapter):
    """Adapter cho Qwen2.5-VL-7B và Qwen3-VL-4B."""

    def load(self) -> None:
        log.info(f"[QwenVLAdapter] Loading {self.hf_id} …")
        self.model     = _load_qwen_model(self.hf_id)
        self.processor = AutoProcessor.from_pretrained(self.hf_id)
        self._device   = next(self.model.parameters()).device
        log.info(f"[QwenVLAdapter] ✔ Loaded on {self._device}")

    def infer(self, images: list["PILImage"], question: str) -> str:
        from qwen_vl_utils import process_vision_info

        max_new = self.cfg["inference"]["max_new_tokens"]

        # Qwen chat template: system + images (interleaved) + question
        # Mỗi ảnh là 1 entry {type: "image"} riêng biệt
        user_content: list[dict] = []
        for img in images:
            user_content.append({"type": "image", "image": img})
        user_content.append({"type": "text", "text": f"Question: {question}\nAnswer:"})

        messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": user_content,
            },
        ]

        # Tokenize
        text_input = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text_input],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._device)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=False,
            )

        # Decode chỉ phần model generate (bỏ phần prompt)
        generated = output_ids[:, inputs.input_ids.shape[1]:]
        response  = self.processor.batch_decode(
            generated,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        return response.strip()

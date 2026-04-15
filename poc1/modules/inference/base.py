"""
modules/inference/base.py
=========================
Trách nhiệm DUY NHẤT: Định nghĩa "hợp đồng" (interface) mà mọi model
adapter phải thực hiện. Runner chỉ biết về class này, không biết về
bất kỳ model cụ thể nào.

Lợi ích:
- Thêm model mới chỉ cần tạo file adapter mới, không sửa runner.
- Type hints rõ ràng giúp IDE hỗ trợ tốt hơn.
"""

from __future__ import annotations

import logging

import gc
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage


class VLMAdapter(ABC):
    """
    Abstract base class cho tất cả VLM adapters.

    Mỗi subclass đại diện cho một model/family cụ thể
    (Vintern, Qwen, InternVL, …).
    """

    def __init__(self, model_key: str, hf_id: str, cfg: dict) -> None:
        """
        Args:
            model_key: Key định danh model trong config (vd: "vintern-1b").
            hf_id:     HuggingFace model ID hoặc local path.
            cfg:       Full config dict từ poc1_config.yaml.
        """
        self.model_key = model_key
        self.hf_id     = hf_id
        self.cfg        = cfg
        self.model      = None  # set bởi load()
        self.processor  = None  # set bởi load() (nếu cần)
        self.tokenizer  = None  # set bởi load() (nếu cần)

    @abstractmethod
    def load(self) -> None:
        """
        Load model weights vào memory (GPU/CPU).
        Gọi 1 lần trước khi bắt đầu inference loop.
        """
        ...

    @abstractmethod
    def infer(self, image: "PILImage", question: str) -> str:
        """
        Chạy inference trả về answer string.

        Args:
            image:    PIL Image (đã ở mode RGB).
            question: Câu hỏi tiếng Việt.

        Returns:
            Câu trả lời dạng string, đã được strip.
        """
        ...

    def unload(self) -> None:
        """
        Giải phóng VRAM sau khi xong inference.
        Override nếu model cần cleanup đặc biệt.
        """
        try:
            import torch
            if self.model is not None:
                del self.model
                self.model = None
            if self.processor is not None:
                del self.processor
                self.processor = None
            if self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            logging.getLogger("poc1").info(f"[{self.model_key}] ✔ Model unloaded, VRAM freed")
        except Exception as e:
            logging.getLogger("poc1").info(f"[{self.model_key}] Warning: unload error: {e}")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(key={self.model_key!r}, hf_id={self.hf_id!r})"

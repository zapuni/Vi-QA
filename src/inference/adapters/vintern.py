"""
src/inference/adapters/vintern.py
=====================================
Trách nhiệm DUY NHẤT: Wrap Vintern-1B-v3.5 và Vintern-3B-R-beta
theo interface VLMAdapter.

Vintern dùng kiến trúc InternVL nên:
  - Load model: AutoModel + AutoTokenizer
  - Inference:  model.chat() với pixel_values tensor
  - Preprocessing: resize → normalize → stack thành tensor

Tương thích: Vintern-1B-v3_5, Vintern-3B-R-beta
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

from src.inference.base import VLMAdapter

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

log = logging.getLogger("poc1")

# ── Constants ────────────────────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)
IMAGE_SIZE    = 448        # kích thước chuẩn của InternVL family

SYSTEM_PROMPT = (
    "Answer the following question about the Vietnamese infographic "
    "concisely with a single term or short phrase. "
    "Do NOT explain. Just give the answer."
)


# ── Image preprocessing helpers ──────────────────────────────────────

def _build_transform(size: int = IMAGE_SIZE) -> T.Compose:
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((size, size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _find_closest_aspect_ratio(
    aspect_ratio: float,
    target_ratios: list[tuple],
    width: int,
    height: int,
    image_size: int,
) -> tuple[int, int]:
    """Chọn aspect ratio gần nhất để chia tile."""
    best_diff  = float("inf")
    best_ratio = (1, 1)
    area       = width * height
    for ratio in target_ratios:
        diff = abs(aspect_ratio - ratio[0] / ratio[1])
        if diff < best_diff:
            best_diff  = diff
            best_ratio = ratio
        elif diff == best_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio


def _dynamic_preprocess(
    image: "PILImage",
    min_num: int  = 1,
    max_num: int  = 12,
    image_size: int = IMAGE_SIZE,
    use_thumbnail: bool = True,
) -> list["PILImage"]:
    """
    Chia ảnh thành các tile dynamic (giống cách InternVL xử lý độ phân giải cao).
    Trả về list PIL Images để stack thành tensor.
    """
    w, h = image.size
    aspect_ratio = w / h

    target_ratios = sorted(
        {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if min_num <= i * j <= max_num
        },
        key=lambda x: x[0] * x[1],
    )
    ar = _find_closest_aspect_ratio(aspect_ratio, target_ratios, w, h, image_size)
    tw = image_size * ar[0]
    th = image_size * ar[1]
    blocks = ar[0] * ar[1]

    resized = image.resize((tw, th))
    cols = tw // image_size
    tiles = []
    for i in range(blocks):
        box = (
            (i % cols) * image_size,
            (i // cols) * image_size,
            ((i % cols) + 1) * image_size,
            ((i // cols) + 1) * image_size,
        )
        tiles.append(resized.crop(box))

    if use_thumbnail and len(tiles) != 1:
        tiles.append(image.resize((image_size, image_size)))

    return tiles


# ── Adapter class ────────────────────────────────────────────────────

class VinternAdapter(VLMAdapter):
    """Adapter cho Vintern-1B-v3.5 và Vintern-3B-R-beta."""

    def load(self) -> None:
        log.info(f"[VinternAdapter] Loading {self.hf_id} …")

        device = self.cfg["inference"].get("device", "cuda")

        # ── Load model KHÔNG dùng device_map="auto" ──────────────────
        # Lý do: InternVLChatModel (custom code) thiếu attribute
        # `all_tied_weights_keys` mà transformers >= 4.48 yêu cầu khi
        # dùng device_map. Vintern-1B/3B đủ nhỏ để fit trên 1 GPU,
        # nên load thẳng bằng .to(device) an toàn hơn.
        self.model = AutoModel.from_pretrained(
            self.hf_id,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            use_flash_attn=False,        # tắt flash-attn để tránh lỗi trên GPU không support
            low_cpu_mem_usage=True,      # giảm RAM khi load weights
        ).to(device).eval()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.hf_id,
            trust_remote_code=True,
            use_fast=False,
        )
        self._transform = _build_transform(IMAGE_SIZE)
        self._device = next(self.model.parameters()).device
        log.info(f"[VinternAdapter] ✔ Loaded on {self._device}")

    def _preprocess(self, image: "PILImage") -> torch.Tensor:
        """PIL Image → float bfloat16 tensor shape (N_tiles, C, H, W)."""
        tiles = _dynamic_preprocess(image, image_size=IMAGE_SIZE, use_thumbnail=True)
        tensor = torch.stack([self._transform(t) for t in tiles])
        return tensor.to(torch.bfloat16).to(self._device)

    def infer(self, images: list["PILImage"], question: str) -> str:
        # Preprocess từng ảnh rồi concat tất cả tiles
        all_tiles = []
        for img in images:
            all_tiles.append(self._preprocess(img))
        pixel_values = torch.cat(all_tiles, dim=0)

        # Prompt: mỗi ảnh cần 1 tag <image>
        image_tags = "\n".join(["<image>"] * len(images))
        prompt = f"{SYSTEM_PROMPT}\n{image_tags}\nQuestion: {question}\nAnswer:"
        max_new = self.cfg["inference"]["max_new_tokens"]

        gen_cfg = {
            "max_new_tokens": max_new,
            "pad_token_id":   self.tokenizer.eos_token_id,
            "do_sample":      False,
        }

        with torch.no_grad():
            response: str = self.model.chat(
                self.tokenizer,
                pixel_values,
                prompt,
                generation_config=gen_cfg,
            )

        # Cắt bỏ phần repeat system prompt nếu model lặp lại
        if "Answer:" in response:
            response = response.split("Answer:")[-1]
        return response.strip()

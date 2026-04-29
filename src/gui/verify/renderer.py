"""
src/gui/verify/renderer.py
============================
Render ảnh với bounding boxes cho verification GUI.

Trách nhiệm:
  - Load ảnh từ image_paths trong teacher output
  - Chuyển đổi bbox normalized [0-1000] → pixel coords
  - Vẽ bbox lên ảnh với label (img_idx hoặc evidence text)
  - Hỗ trợ cả single-image (4-element bbox) và multi-image (5-element bbox)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

log = logging.getLogger("verify_gui")

# Màu sắc cho bounding boxes (phân biệt ảnh khác nhau)
BBOX_COLORS = [
    "#FF4444",  # Red
    "#44FF44",  # Green
    "#4444FF",  # Blue
    "#FFAA00",  # Orange
    "#FF44FF",  # Magenta
    "#44FFFF",  # Cyan
    "#FFFF44",  # Yellow
    "#AA44FF",  # Purple
]


def load_images(record: dict, project_root: str | Path = ".") -> list[Image.Image]:
    """
    Load danh sách ảnh PIL từ teacher output record.

    Args:
        record: dict từ JSONL — có key "image_paths"
        project_root: Thư mục gốc của project (để resolve relative paths)

    Returns:
        List of PIL Images (mode RGB)
    """
    root = Path(project_root)
    images = []

    for img_path in record.get("image_paths", []):
        abs_path = root / img_path
        if not abs_path.exists():
            log.warning(f"[renderer] Image not found: {abs_path}")
            # Tạo placeholder
            placeholder = Image.new("RGB", (400, 300), "#333333")
            draw = ImageDraw.Draw(placeholder)
            draw.text((20, 140), f"Not found:\n{img_path}", fill="#FF4444")
            images.append(placeholder)
        else:
            img = Image.open(abs_path).convert("RGB")
            images.append(img)

    return images


def normalize_boxes(
    boxes: list[list[int]],
    num_images: int = 1,
) -> list[dict]:
    """
    Normalize bbox format thành danh sách dicts thống nhất.

    Input formats:
      - 4-element: [x1, y1, x2, y2]          → img_idx = 0
      - 5-element: [img_idx, x1, y1, x2, y2] → img_idx from bbox

    Returns:
        List of {"img_idx": int, "coords": [x1,y1,x2,y2]} normalized [0-1000]
    """
    result = []
    for box in boxes:
        if len(box) == 5:
            result.append({
                "img_idx": box[0],
                "coords": box[1:],
            })
        elif len(box) == 4:
            result.append({
                "img_idx": 0,
                "coords": box,
            })
        else:
            log.warning(f"[renderer] Invalid bbox length {len(box)}: {box}")
    return result


def draw_bboxes_on_image(
    image: Image.Image,
    boxes: list[dict],
    img_idx: int = 0,
    color: str | None = None,
    line_width: int = 3,
) -> Image.Image:
    """
    Vẽ bounding boxes lên ảnh.

    Args:
        image: PIL Image gốc
        boxes: List of normalized boxes (from normalize_boxes)
        img_idx: Chỉ vẽ boxes có img_idx này
        color: Màu vẽ, None = auto
        line_width: Độ dày đường vẽ

    Returns:
        PIL Image đã vẽ bbox
    """
    img_copy = image.copy()
    draw = ImageDraw.Draw(img_copy)

    w, h = img_copy.size

    # Lọc boxes thuộc ảnh này
    matching = [b for b in boxes if b["img_idx"] == img_idx]

    for i, box_info in enumerate(matching):
        coords = box_info["coords"]  # [x1, y1, x2, y2] normalized [0-1000]

        # Convert normalized [0-1000] → pixel coords
        x1 = int(coords[0] * w / 1000)
        y1 = int(coords[1] * h / 1000)
        x2 = int(coords[2] * w / 1000)
        y2 = int(coords[3] * h / 1000)

        # Clamp
        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w - 1))
        y2 = max(0, min(y2, h - 1))

        c = color or BBOX_COLORS[i % len(BBOX_COLORS)]
        draw.rectangle([x1, y1, x2, y2], outline=c, width=line_width)

        # Label
        label = f"Box {i}"
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
        except (IOError, OSError):
            font = ImageFont.load_default()

        # Text background
        text_bbox = draw.textbbox((x1, y1 - 18), label, font=font)
        draw.rectangle(text_bbox, fill=c)
        draw.text((x1, y1 - 18), label, fill="white", font=font)

    return img_copy


def render_record_images(
    record: dict,
    project_root: str | Path = ".",
) -> list[Image.Image]:
    """
    Render tất cả ảnh trong record với bounding boxes đã vẽ.

    Args:
        record: dict từ JSONL
        project_root: Thư mục gốc

    Returns:
        List of PIL Images đã vẽ bbox
    """
    images = load_images(record, project_root)
    boxes = normalize_boxes(
        record.get("grounding_boxes", []),
        num_images=len(images),
    )

    rendered = []
    for idx, img in enumerate(images):
        rendered_img = draw_bboxes_on_image(img, boxes, img_idx=idx)
        rendered.append(rendered_img)
        # Close original image to free memory (rendered_img is a copy)
        img.close()

    return rendered

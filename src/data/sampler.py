"""
src/data/sampler.py
=======================
Trách nhiệm DUY NHẤT: Trả về N mẫu ngẫu nhiên (reproducible) từ dataset.
Dùng cho dev_mode để test nhanh pipeline mà không cần chạy toàn bộ.
"""

from __future__ import annotations

import logging

import random


def sample(samples: list[dict], n: int, seed: int = 42) -> list[dict]:
    """
    Lấy N mẫu ngẫu nhiên. Nếu n >= len(samples) thì trả về tất cả.

    Args:
        samples: Danh sách samples đầy đủ.
        n:       Số lượng mẫu muốn lấy.
        seed:    Random seed để reproducible.

    Returns:
        Danh sách con.
    """
    if n >= len(samples):
        return samples

    rng = random.Random(seed)
    chosen = rng.sample(samples, n)
    logging.getLogger("poc1").info(f"[data/sampler] ✔ Sampled {len(chosen)}/{len(samples)} samples (seed={seed})")
    return chosen

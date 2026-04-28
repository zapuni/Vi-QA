"""
src/data/image_provider.py
============================
Lazy loading + prefetch buffer cho images.

Thiết kế:
  - Background ThreadPoolExecutor prefetch N ảnh tiếp theo
  - LRU-style cache: giữ tối đa max_cached ảnh trong RAM
  - Thread-safe: dùng Lock cho cache access
  - Auto-release: caller gọi release() sau khi xử lý xong

Giải quyết vấn đề:
  Trước đó, loader.py load TẤT CẢ ảnh PIL vào RAM cùng lúc
  (2600+ ảnh infographic → 7-39 GB RAM). Module này thay thế bằng
  lazy loading: chỉ giữ tối đa max_cached ảnh, prefetch trước
  prefetch_size ảnh tiếp theo trong background threads.

Sử dụng:
  provider = ImageProvider(prefetch_size=32, max_workers=2)

  # Trong inference loop:
  for i, sample in enumerate(pending):
      # Prefetch ảnh cho batch tiếp theo
      if i + 1 < len(pending):
          provider.schedule(pending[i+1]["image_paths"])

      # Load ảnh hiện tại (instant nếu đã prefetch)
      images = provider.get(sample["image_paths"])
      prediction = adapter.infer(images, sample["question"])

      # Giải phóng ngay
      provider.release(sample["image_paths"])

  provider.shutdown()
"""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor, Future
from pathlib import Path

from PIL import Image

log = logging.getLogger("vi_qa")


def _load_single_image(path: str) -> Image.Image:
    """Load 1 ảnh từ disk, convert sang RGB."""
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


class ImageProvider:
    """
    Lazy image loader với prefetch buffer.

    - Không load ảnh cho đến khi get() được gọi
    - Background threads prefetch ảnh tiếp theo để tránh I/O blocking
    - Auto-evict ảnh cũ khi vượt quá max_cached (LRU)
    - Thread-safe: mọi truy cập cache đều qua Lock

    Args:
        prefetch_size: Số ảnh schedule prefetch trước mỗi lần.
        max_workers:   Số background threads đọc disk song song.
        max_cached:    Giới hạn ảnh tối đa trong cache RAM.
    """

    def __init__(
        self,
        prefetch_size: int = 32,
        max_workers: int = 2,
        max_cached: int = 64,
    ) -> None:
        self.prefetch_size = prefetch_size
        self.max_cached = max_cached

        self._executor = ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="img_prefetch",
        )
        self._cache: dict[str, Image.Image] = {}
        self._futures: dict[str, Future] = {}
        self._lock = threading.Lock()
        self._access_order: list[str] = []  # LRU tracking

    def schedule(self, image_paths: list[str]) -> None:
        """
        Submit paths vào prefetch queue (non-blocking).

        Chỉ schedule những path chưa có trong cache
        và chưa có pending future.
        """
        with self._lock:
            for path in image_paths:
                if path not in self._cache and path not in self._futures:
                    future = self._executor.submit(_load_single_image, path)
                    self._futures[path] = future

    def schedule_batch(self, samples: list[dict], start_idx: int) -> None:
        """
        Schedule prefetch cho nhiều samples liên tiếp.

        Args:
            samples:   Danh sách samples (mỗi dict có key "image_paths").
            start_idx: Index bắt đầu schedule (thường là current + 1).
        """
        end_idx = min(start_idx + self.prefetch_size, len(samples))
        for i in range(start_idx, end_idx):
            self.schedule(samples[i]["image_paths"])

    def get(self, image_paths: list[str]) -> list[Image.Image]:
        """
        Lấy danh sách PIL Images.

        Fallback chain cho mỗi path:
          1. Cache hit → trả ngay (O(1))
          2. Pending future → wait rồi trả
          3. Cache miss → load synchronous từ disk
        """
        images = []
        for path in image_paths:
            img = self._get_single(path)
            images.append(img)
        return images

    def _get_single(self, path: str) -> Image.Image:
        """Lấy 1 ảnh, với fallback chain: cache → future → sync load."""
        with self._lock:
            # 1. Check cache
            if path in self._cache:
                self._touch_lru(path)
                return self._cache[path]

            # 2. Check pending future
            future = self._futures.pop(path, None)

        if future is not None:
            # Wait for prefetch to complete
            img = future.result()
        else:
            # 3. Sync load (cache miss, no prefetch)
            img = _load_single_image(path)

        # Store in cache
        with self._lock:
            self._cache[path] = img
            self._touch_lru(path)
            self._evict_if_needed()

        return img

    def release(self, image_paths: list[str]) -> None:
        """Giải phóng ảnh khỏi cache — close PIL handle + remove."""
        with self._lock:
            for path in image_paths:
                img = self._cache.pop(path, None)
                if img is not None:
                    try:
                        img.close()
                    except Exception:
                        pass
                if path in self._access_order:
                    self._access_order.remove(path)

    def _touch_lru(self, path: str) -> None:
        """Update LRU order (called under lock)."""
        if path in self._access_order:
            self._access_order.remove(path)
        self._access_order.append(path)

    def _evict_if_needed(self) -> None:
        """Evict oldest entries nếu cache vượt max_cached (called under lock)."""
        while len(self._cache) > self.max_cached and self._access_order:
            oldest = self._access_order.pop(0)
            img = self._cache.pop(oldest, None)
            if img is not None:
                try:
                    img.close()
                except Exception:
                    pass

    def clear(self) -> None:
        """Giải phóng toàn bộ cache."""
        with self._lock:
            for img in self._cache.values():
                try:
                    img.close()
                except Exception:
                    pass
            self._cache.clear()
            self._access_order.clear()
            # Cancel pending futures
            for f in self._futures.values():
                f.cancel()
            self._futures.clear()

    def shutdown(self) -> None:
        """Cleanup: clear cache + shutdown thread pool."""
        self.clear()
        self._executor.shutdown(wait=False)
        log.info(
            f"[ImageProvider] Shutdown — cleared cache, "
            f"thread pool stopped"
        )

    @property
    def cache_size(self) -> int:
        """Số ảnh hiện đang trong cache."""
        return len(self._cache)

    def __repr__(self) -> str:
        return (
            f"ImageProvider(prefetch_size={self.prefetch_size}, "
            f"max_cached={self.max_cached}, "
            f"cache_size={self.cache_size})"
        )

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            pass

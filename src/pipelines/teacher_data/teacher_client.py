"""
src/pipelines/teacher_data/teacher_client.py
=============================================
Async client gọi vLLM teacher API qua OpenAI-compatible endpoint.
Hỗ trợ:
  - AsyncOpenAI client
  - Semaphore-based concurrency control
  - Retry với exponential backoff
  - Image base64 encoding
  - Thinking mode (enable_thinking)
  - Streaming mode (stream=True) để tránh lỗi 524 timeout
    từ reverse proxy

Khi stream=True:
  - Dữ liệu được gửi dạng SSE chunks → connection stay alive
  - Tất cả chunks được gom lại thành dict cùng format với non-stream
  - Parser/Orchestrator downstream KHÔNG cần thay đổi
"""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path

from openai import AsyncOpenAI

from src.pipelines.teacher_data.config import TeacherConfig

log = logging.getLogger("teacher_data")


class TeacherClient:
    """Async client gọi vLLM teacher API."""

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config
        self.client = AsyncOpenAI(
            base_url=config.base_url,
            api_key=config.api_key,
        )
        self.sem = asyncio.Semaphore(config.max_concurrent)

    async def infer(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
    ) -> dict:
        """
        Gọi teacher API với retry.

        Args:
            system_prompt: System prompt.
            user_prompt:   User prompt.
            image_paths:   List đường dẫn ảnh tuyệt đối.

        Returns:
            Raw response dict từ OpenAI API (cùng format cho cả
            stream và non-stream mode).
        """
        last_error = None
        for attempt in range(self.config.retry_attempts):
            try:
                async with self.sem:
                    result = await asyncio.wait_for(
                        self._call_api(system_prompt, user_prompt, image_paths),
                        timeout=self.config.timeout_seconds,
                    )
                    return result
            except Exception as e:
                last_error = e
                if attempt < self.config.retry_attempts - 1:
                    wait = self.config.retry_backoff ** (attempt + 1)
                    log.warning(
                        f"[teacher_client] Attempt {attempt + 1} failed: {e}, "
                        f"retrying in {wait:.1f}s..."
                    )
                    await asyncio.sleep(wait)
                else:
                    log.error(f"[teacher_client] All {self.config.retry_attempts} attempts failed: {e}")

        raise last_error  # type: ignore[misc]

    async def _call_api(
        self,
        system_prompt: str,
        user_prompt: str,
        image_paths: list[str],
    ) -> dict:
        """Gọi thực API (stream hoặc non-stream tùy config)."""
        # Build user message content: text + images
        content: list[dict] = [{"type": "text", "text": user_prompt}]

        for img_path in image_paths:
            img_url = self._image_to_data_url(img_path)
            content.append({
                "type": "image_url",
                "image_url": {"url": img_url},
            })

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ]

        extra_body = {
            "chat_template_kwargs": {
                "enable_thinking": self.config.enable_thinking,
            }
        }

        if self.config.stream:
            return await self._call_api_stream(messages, extra_body)
        else:
            return await self._call_api_non_stream(messages, extra_body)

    async def _call_api_non_stream(
        self,
        messages: list[dict],
        extra_body: dict,
    ) -> dict:
        """Gọi API không stream (hành vi gốc)."""
        resp = await self.client.chat.completions.create(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_completion_tokens,
            messages=messages,
            extra_body=extra_body,
            stream=False,
        )
        return resp.model_dump()

    async def _call_api_stream(
        self,
        messages: list[dict],
        extra_body: dict,
    ) -> dict:
        """
        Gọi API với stream=True, gom chunks lại thành dict
        cùng format với non-stream response.

        Tránh lỗi HTTP 524 (Cloudflare/reverse proxy timeout)
        vì connection liên tục nhận SSE chunks thay vì chờ
        response hoàn chỉnh.

        Output format (giống hệt non-stream):
        {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "...",
                    "reasoning_content": "..."   # nếu enable_thinking
                },
                "finish_reason": "stop"
            }],
            "model": "...",
            "usage": {...}
        }
        """
        stream = await self.client.chat.completions.create(
            model=self.config.model_name,
            temperature=self.config.temperature,
            max_completion_tokens=self.config.max_completion_tokens,
            messages=messages,
            extra_body=extra_body,
            stream=True,
            stream_options={"include_usage": True},
        )

        # ── Gom tất cả chunks ───────────────────────────────────────
        content_parts: list[str] = []
        reasoning_parts: list[str] = []
        finish_reason: str | None = None
        model_name: str = ""
        usage: dict | None = None

        async for chunk in stream:
            # Metadata từ chunk đầu tiên hoặc chunk cuối
            if chunk.model:
                model_name = chunk.model
            if chunk.usage is not None:
                usage = chunk.usage.model_dump() if hasattr(chunk.usage, "model_dump") else dict(chunk.usage)

            if not chunk.choices:
                # Chunk cuối cùng thường chỉ chứa usage, không có choices
                continue

            delta = chunk.choices[0].delta

            # Gom content text
            if delta.content:
                content_parts.append(delta.content)

            # Gom reasoning_content (vLLM Thinking mode)
            # vLLM trả reasoning trong delta.reasoning_content
            reasoning_text = getattr(delta, "reasoning_content", None)
            if reasoning_text:
                reasoning_parts.append(reasoning_text)

            # Finish reason
            if chunk.choices[0].finish_reason:
                finish_reason = chunk.choices[0].finish_reason

        # ── Reassemble thành format giống non-stream ─────────────────
        full_content = "".join(content_parts)
        full_reasoning = "".join(reasoning_parts) if reasoning_parts else None

        message: dict = {
            "role": "assistant",
            "content": full_content,
        }
        if full_reasoning:
            # Parser tìm cả "reasoning" và "reasoning_content"
            message["reasoning_content"] = full_reasoning

        result: dict = {
            "choices": [{
                "message": message,
                "finish_reason": finish_reason or "stop",
                "index": 0,
            }],
            "model": model_name,
        }
        if usage:
            result["usage"] = usage

        log.debug(
            f"[teacher_client/stream] Assembled response: "
            f"content_len={len(full_content)}, "
            f"reasoning_len={len(full_reasoning) if full_reasoning else 0}, "
            f"finish_reason={finish_reason}"
        )

        return result

    @staticmethod
    def _image_to_data_url(img_path: str) -> str:
        """Convert image file → base64 data URL."""
        path = Path(img_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        suffix = path.suffix.lower()
        mime = "image/jpeg" if suffix in (".jpg", ".jpeg") else f"image/{suffix.lstrip('.')}"

        with open(path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        return f"data:{mime};base64,{img_b64}"

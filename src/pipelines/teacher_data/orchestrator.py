"""
src/pipelines/teacher_data/orchestrator.py
============================================
Orchestrator chính cho teacher data generation pipeline.

Pipeline flow:
  1. Load dataset → iter RawSamples
  2. Plan output profile cho từng sample
  3. Build prompt theo profile
  4. Gọi teacher API (async + semaphore)
  5. Parse response → structured dict
  6. Validate theo profile contract
  7. Self-correction retry nếu validation fail
  8. Write output → shard .jsonl
  9. Checkpoint mark done
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.pipelines.teacher_data.checkpoint import CheckpointManager
from src.pipelines.teacher_data.config import TeacherConfig
from src.pipelines.teacher_data.loader import DatasetLoader
from src.pipelines.teacher_data.models import TeacherOutput
from src.pipelines.teacher_data.parser import ResponseParser
from src.pipelines.teacher_data.planner import TaskPlanner
from src.pipelines.teacher_data.prompt_builder import PromptBuilder
from src.pipelines.teacher_data.teacher_client import TeacherClient
from src.pipelines.teacher_data.validator import OutputValidator
from src.pipelines.teacher_data.writer import ShardWriter

log = logging.getLogger("teacher_data")


class TeacherDataOrchestrator:
    """Orchestrator chính cho pipeline teacher data generation."""

    def __init__(self, config: TeacherConfig) -> None:
        self.config = config

        # Initialize components
        self.loader = DatasetLoader(config.data_root, config.json_files)
        self.planner = TaskPlanner()
        self.prompt_builder = PromptBuilder()
        self.client = TeacherClient(config)
        self.parser = ResponseParser()
        self.validator = OutputValidator()

        # Run output directory
        run_dir = Path(config.output_dir) / config.run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = run_dir
        self.writer = ShardWriter(run_dir, config.shard_size)
        self.checkpoint = CheckpointManager(run_dir, config.run_id)

    async def run(self) -> None:
        """Chạy pipeline chính."""
        log.info(f"Bắt đầu run: {self.config.run_id}")
        log.info(f"Teacher model: {self.config.model_name}")
        log.info(f"Max concurrent: {self.config.max_concurrent}")

        tasks: list[asyncio.Task] = []
        batch_count = 0
        total_pending = 0

        for sample in self.loader.iter_samples(subset=self.config.subset):
            # Skip nếu đã xử lý
            if self.checkpoint.is_done(sample.sample_id):
                continue

            total_pending += 1
            task = asyncio.create_task(self._process_sample(sample))
            tasks.append(task)

            # Batch execution tại checkpoint_interval
            if len(tasks) >= self.config.checkpoint_interval:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        log.error(f"[orchestrator] Batch exception: {r}")
                tasks = []
                batch_count += 1
                stats = self.checkpoint.progress["stats"]
                log.info(
                    f"[orchestrator] Checkpoint batch {batch_count}: "
                    f"success={stats['success']}, failed={stats['failed']}"
                )

        # Xử lý batch cuối
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    log.error(f"[orchestrator] Final batch exception: {r}")

        stats = self.checkpoint.progress["stats"]
        log.info("=" * 60)
        log.info(f"Hoàn thành run: {self.config.run_id}")
        log.info(f"Total pending: {total_pending}")
        log.info(f"Success: {stats['success']}")
        log.info(f"Failed: {stats['failed']}")
        log.info(f"Output: {self.run_dir}")
        log.info("=" * 60)

    async def _process_sample(self, sample) -> None:
        """Xử lý một sample với self-correction retry."""
        num_images = sample.num_images
        try:
            # 1. Plan output profile
            plan = self.planner.plan(sample)

            log.debug(
                f"[orchestrator] {sample.sample_id}: "
                f"num_images={num_images}, profile={plan['output_profile']}"
            )

            # 2. Build prompt
            req = self.prompt_builder.build(sample, plan)

            # 3. Gọi teacher API
            raw_resp = await self.client.infer(
                req.system_prompt,
                req.user_prompt,
                sample.image_paths,
            )

            # 4. Parse response
            parsed = self.parser.parse(
                sample.sample_id,
                raw_resp,
                plan["output_profile"],
            )

            # 5. Build TeacherOutput
            output = TeacherOutput(
                model_name=self.config.model_name,
                num_images=num_images,
                image_paths=sample.image_paths,
                run_meta={
                    "run_id": self.config.run_id,
                    "attempt": 1,
                    "num_images": num_images,
                },
                **parsed,
            )

            # 6. Validate
            is_valid, correction_prompt = self.validator.validate_with_retry_prompt(
                output
            )

            if not is_valid:
                # Self-correction retry
                log.info(
                    f"[orchestrator] Self-correction for {sample.sample_id} "
                    f"(num_images={num_images})..."
                )
                raw_resp = await self.client.infer(
                    req.system_prompt + "\n\n" + correction_prompt,
                    req.user_prompt,
                    sample.image_paths,
                )
                parsed = self.parser.parse(
                    sample.sample_id,
                    raw_resp,
                    plan["output_profile"],
                )
                output = TeacherOutput(
                    model_name=self.config.model_name,
                    num_images=num_images,
                    image_paths=sample.image_paths,
                    run_meta={
                        "run_id": self.config.run_id,
                        "attempt": 2,
                        "num_images": num_images,
                    },
                    **parsed,
                )
                # Validate lần cuối — nếu vẫn fail thì raise
                output.validation = self.validator.validate_or_raise(output)
            else:
                output.validation = self.validator.validate_or_raise(output)

            # 7. Write output
            self.writer.write(output)

            # 8. Checkpoint
            self.checkpoint.mark_done(sample.sample_id)

        except Exception as e:
            log.warning(
                f"[orchestrator] Failed sample {sample.sample_id} "
                f"(num_images={num_images}): {e}"
            )
            self.checkpoint.mark_failed(sample.sample_id, str(e))

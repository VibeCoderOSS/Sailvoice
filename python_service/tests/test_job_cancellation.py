from __future__ import annotations

import asyncio
from pathlib import Path

from app.config import Settings
from app.manager import JobCanceledError, JobManager
from app.schemas import JobRequestText


def _build_settings(tmp_path: Path) -> Settings:
    output_dir = tmp_path / 'outputs'
    model_cache_dir = tmp_path / 'models'
    temp_dir = tmp_path / 'tmp'
    whisper_cache_dir = model_cache_dir / 'whisper'
    alignment_model_dir = model_cache_dir / 'whisperx'

    for path in (output_dir, model_cache_dir, temp_dir, whisper_cache_dir, alignment_model_dir):
        path.mkdir(parents=True, exist_ok=True)

    return Settings(
        output_dir=output_dir,
        model_cache_dir=model_cache_dir,
        temp_dir=temp_dir,
        whisper_cache_dir=whisper_cache_dir,
        whisper_model_size='tiny',
        align_python_bin=tmp_path / 'runtime' / '.venv-align' / 'bin' / 'python3',
        alignment_model_dir=alignment_model_dir,
        alignment_languages=('de', 'en'),
        alignment_min_coverage=0.97,
        voice_secret_b64='',
        performance_profile='standard',
        quality_preset='balanced',
        allow_macos_fallback=False,
        tts_max_tokens=4096,
        tts_max_tokens_pdf=3072,
        pdf_subchunk_max_words=95,
        pdf_subchunk_max_chars=700,
        alignment_retry_threshold=0.94,
        alignment_retry_tail_gap_ms=2500,
        tts_voice_consistency_mode='strict',
        tts_chunk_seed_base=424242,
        tts_temperature_strict=0.18,
        tts_top_k_strict=20,
        tts_top_p_strict=0.85,
        tts_repetition_penalty_strict=1.12,
        standard_preset_speakers=('serena', 'vivian', 'ryan', 'aiden'),
        standard_default_speaker='serena',
    )


async def _wait_for(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError('Condition not reached before timeout.')


def test_cancel_queued_job_removes_it_from_pending_queue(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))
        release_first_job = asyncio.Event()
        started: list[str] = []

        async def fake_run_job(job_id: str) -> None:
            started.append(job_id)
            if len(started) == 1:
                await manager._update_job(job_id, state='running', progress=0.2, statusMessage='working')
                await release_first_job.wait()
                await manager._update_job(job_id, state='done', progress=1.0, statusMessage='done', phase='done', phaseProgress=1.0)
                return
            raise AssertionError('Canceled queued job should never start running.')

        manager._run_job = fake_run_job  # type: ignore[method-assign]

        await manager.start()
        try:
            first_job = await manager.create_text_job(
                JobRequestText(text='first', model='customvoice', speaker='serena', outputFormats=['mp3'])
            )
            await _wait_for(lambda: manager.running_job_id == first_job)

            second_job = await manager.create_text_job(
                JobRequestText(text='second', model='customvoice', speaker='serena', outputFormats=['mp3'])
            )

            await manager.cancel_job(second_job)
            canceled = await manager.get_job(second_job)
            assert canceled is not None
            assert canceled.state == 'canceled'
            assert canceled.cancelReason == 'Canceled by user.'

            release_first_job.set()
            await _wait_for(lambda: (awaitable := manager.store.get_job(first_job)) is not None and awaitable.state == 'done')
            await asyncio.sleep(0.05)
            assert started == [first_job]
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_cancel_waiting_language_job_clears_waiter(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))
        job_id = await manager.create_text_job(
            JobRequestText(text='language', model='customvoice', speaker='serena', outputFormats=['mp3'])
        )

        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        manager.language_waiters[job_id] = waiter
        await manager._update_job(
            job_id,
            state='waiting_language',
            statusMessage='waiting_language_confirmation',
            phase='queued',
            phaseProgress=0.05,
        )

        await manager.cancel_job(job_id)
        canceled = await manager.get_job(job_id)
        assert canceled is not None
        assert canceled.state == 'canceled'
        assert canceled.cancelReason == 'Canceled by user.'
        assert waiter.cancelled()

    asyncio.run(scenario())


def test_cancel_running_job_marks_it_canceled_at_safe_boundary(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))

        async def fake_run_job(job_id: str) -> None:
            await manager._update_job(job_id, state='running', progress=0.3, statusMessage='working')
            while job_id not in manager.cancel_requests:
                await asyncio.sleep(0.01)
            raise JobCanceledError(manager.cancel_requests[job_id])

        manager._run_job = fake_run_job  # type: ignore[method-assign]

        await manager.start()
        try:
            job_id = await manager.create_text_job(
                JobRequestText(text='running', model='customvoice', speaker='serena', outputFormats=['mp3'])
            )
            await _wait_for(lambda: manager.running_job_id == job_id)

            await manager.cancel_job(job_id)
            await _wait_for(lambda: (job := manager.store.get_job(job_id)) is not None and job.state == 'canceled')

            canceled = await manager.get_job(job_id)
            assert canceled is not None
            assert canceled.state == 'canceled'
            assert canceled.cancelReason == 'Canceled by user.'
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_delete_queued_job_removes_it_immediately(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))
        await manager.start()
        try:
            job_id = await manager.create_text_job(
                JobRequestText(text='queued delete', model='customvoice', speaker='serena', outputFormats=['mp3'])
            )

            await manager.delete_job(job_id)

            assert await manager.get_job(job_id) is None
            assert job_id not in manager.pending_jobs
            assert manager.cancel_requests[job_id] == 'Deleted by user.'
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_delete_running_job_removes_it_immediately(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))

        async def fake_run_job(job_id: str) -> None:
            await manager._update_job(job_id, state='running', progress=0.3, statusMessage='working')
            while job_id not in manager.cancel_requests:
                await asyncio.sleep(0.01)
            raise JobCanceledError(manager.cancel_requests[job_id])

        manager._run_job = fake_run_job  # type: ignore[method-assign]

        await manager.start()
        try:
            job_id = await manager.create_text_job(
                JobRequestText(text='running delete', model='customvoice', speaker='serena', outputFormats=['mp3'])
            )
            await _wait_for(lambda: manager.running_job_id == job_id)

            await manager.delete_job(job_id)

            assert await manager.get_job(job_id) is None
            assert manager.cancel_requests[job_id] == 'Deleted by user.'
            await _wait_for(lambda: manager.running_job_task is None or manager.running_job_task.done())
        finally:
            await manager.stop()

    asyncio.run(scenario())


def test_start_recovers_interrupted_active_jobs(tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))
        job_id = await manager.create_text_job(
            JobRequestText(text='recover me', model='customvoice', speaker='serena', outputFormats=['mp3', 'mp4'])
        )
        await manager._update_job(
            job_id,
            state='running',
            statusMessage='working',
            phase='synthesizing',
            phaseProgress=0.4,
            alignmentState='running',
            mp4State='running',
            mp4Progress=0.3,
        )

        restarted = JobManager(_build_settings(tmp_path))
        await restarted.start()
        try:
            recovered = await restarted.get_job(job_id)
            assert recovered is not None
            assert recovered.state == 'failed'
            assert recovered.error == 'Interrupted by app restart.'
            assert recovered.alignmentState == 'failed'
            assert recovered.alignmentError == 'Interrupted by app restart.'
            assert recovered.mp4State == 'failed'
            assert recovered.mp4Error == 'Interrupted by app restart.'
        finally:
            await restarted.stop()

    asyncio.run(scenario())

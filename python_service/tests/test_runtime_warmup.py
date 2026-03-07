from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from app.manager import JobManager


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


def test_warmup_runtime_is_idempotent_while_running(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))
        gate = asyncio.Event()
        probe_calls = 0

        monkeypatch.setattr(manager.engine, 'ensure_model', lambda *_args, **_kwargs: None)
        monkeypatch.setattr(manager.engine, 'load_model_into_ram', lambda *_args, **_kwargs: True)

        async def fake_probe(*, force: bool = False) -> tuple[bool, str | None]:
            if not force:
                return True, 'ok'
            nonlocal probe_calls
            probe_calls += 1
            await gate.wait()
            return True, 'ok'

        monkeypatch.setattr(manager, '_probe_alignment_runtime', fake_probe)

        first_status = await manager.warmup_runtime(models=['customvoice'], include_alignment_probe=True)
        assert first_status['warmupState'] == 'running'

        running_task = manager._warmup_task
        assert running_task is not None

        second_status = await manager.warmup_runtime(models=['customvoice'], include_alignment_probe=True)
        assert second_status['warmupState'] == 'running'
        assert manager._warmup_task is running_task

        gate.set()
        await running_task

        ready_status = await manager.get_runtime_status()
        assert ready_status['warmupState'] == 'ready'
        assert ready_status['warmupError'] is None
        assert ready_status['warmupAt'] is not None
        assert probe_calls == 1

    asyncio.run(scenario())


def test_warmup_runtime_failure_sets_failed_state(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))

        monkeypatch.setattr(manager.engine, 'ensure_model', lambda *_args, **_kwargs: 'download failed')
        monkeypatch.setattr(manager.engine, 'load_model_into_ram', lambda *_args, **_kwargs: True)

        status = await manager.warmup_runtime(models=['customvoice'], include_alignment_probe=False)
        assert status['warmupState'] == 'running'

        running_task = manager._warmup_task
        assert running_task is not None
        with pytest.raises(RuntimeError, match='download failed'):
            await running_task

        failed = await manager.get_runtime_status()
        assert failed['warmupState'] == 'failed'
        assert failed['warmupError'] is not None
        assert 'download failed' in failed['warmupError']

    asyncio.run(scenario())

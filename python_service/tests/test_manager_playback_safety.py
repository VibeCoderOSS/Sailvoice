from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import manager as manager_module
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


def test_job_profile_adaptation_does_not_mutate_global_settings(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))
        initial_profile = manager.settings.performance_profile

        monkeypatch.setattr(
            manager_module.psutil,
            'Process',
            lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=200 * 1024 * 1024)),
        )

        profile = await manager._resolve_job_performance_profile('job-low-memory')
        assert profile == 'standard'
        assert manager.settings.performance_profile == initial_profile
        assert manager._job_performance_profiles['job-low-memory'] == 'standard'

    asyncio.run(scenario())


def test_job_profile_can_switch_to_memory_for_single_job_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))

        monkeypatch.setattr(
            manager_module.psutil,
            'Process',
            lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=4_200 * 1024 * 1024)),
        )

        async def fake_update_job(*_args, **_kwargs):
            return None

        async def fake_publish(*_args, **_kwargs):
            return None

        monkeypatch.setattr(manager, '_update_job', fake_update_job)
        monkeypatch.setattr(manager, 'publish', fake_publish)

        profile = await manager._resolve_job_performance_profile('job-memory-pressure')
        assert profile == 'memory'
        assert manager.settings.performance_profile == 'standard'
        assert manager._job_performance_profiles['job-memory-pressure'] == 'memory'

    asyncio.run(scenario())


def test_alignment_parallelism_prefers_three_workers_for_32gb_hosts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manager = JobManager(_build_settings(tmp_path))

    monkeypatch.setattr(
        manager_module.psutil,
        'virtual_memory',
        lambda: SimpleNamespace(total=32 * 1024 * 1024 * 1024),
    )
    monkeypatch.setattr(
        manager_module.psutil,
        'Process',
        lambda: SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=8 * 1024 * 1024 * 1024)),
    )

    assert manager._alignment_parallelism('standard') == 3
    assert manager._alignment_parallelism('memory') == 1

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from app.manager import JobManager
from app.schemas import VoiceCreateRequest
from app.tts_engine import SynthesisResult


def _build_settings(tmp_path: Path) -> Settings:
    output_dir = tmp_path / 'outputs'
    model_cache_dir = tmp_path / 'models'
    temp_dir = tmp_path / 'tmp'
    whisper_cache_dir = model_cache_dir / 'whisper'
    alignment_model_dir = model_cache_dir / 'whisperx'

    for path in (output_dir, model_cache_dir, temp_dir, whisper_cache_dir, alignment_model_dir):
        path.mkdir(parents=True, exist_ok=True)

    settings = Settings(
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
    settings.assets_dir.mkdir(parents=True, exist_ok=True)
    settings.voices_dir.mkdir(parents=True, exist_ok=True)
    return settings


def test_preview_clone_voice_does_not_crash_and_cleans_temp_reference(tmp_path: Path, monkeypatch) -> None:
    async def scenario() -> None:
        manager = JobManager(_build_settings(tmp_path))

        sample_audio = tmp_path / 'sample.wav'
        sample_audio.write_bytes(b'RIFFfake-wave-data')

        voice = await manager.create_voice(
            VoiceCreateRequest(
                name='Test Clone',
                language='en',
                audioPath=str(sample_audio),
                referenceText='Reference sample',
            )
        )

        captured: dict[str, Path | None] = {'ref_path': None}
        output_wav = manager.settings.assets_dir / 'preview-test.wav'
        output_wav.write_bytes(b'RIFFpreview-audio')

        monkeypatch.setattr(
            'app.manager.detect_runtime_status',
            lambda: SimpleNamespace(compatible=True, reason='ok'),
        )
        monkeypatch.setattr(manager.engine, 'ensure_model', lambda model_id, prefer_official=False: None)

        def fake_synthesize_chunk(**kwargs):
            captured['ref_path'] = kwargs.get('voice_reference_path')
            return SynthesisResult(wav_path=output_wav, duration_ms=1000, warning=None)

        monkeypatch.setattr(manager.engine, 'synthesize_chunk', fake_synthesize_chunk)

        asset_id = await manager.preview_voice(
            text='Hello from clone preview.',
            model_id='base',
            voice_id=voice.id,
            language='en',
        )

        assert manager.get_asset_path(asset_id) == str(output_wav)
        assert captured['ref_path'] is not None
        assert not captured['ref_path'].exists()

    asyncio.run(scenario())

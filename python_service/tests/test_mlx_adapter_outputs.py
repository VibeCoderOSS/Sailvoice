from __future__ import annotations

import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from app import mlx_adapter
from app.mlx_adapter import (
    _collect_generated_audio_paths,
    _extract_paths_from_result,
    _normalize_mlx_audio_language,
    _resolve_custom_voice_speaker,
    _synthesize_with_mlx_audio,
)


def test_extract_paths_from_result_handles_nested_structures() -> None:
    payload = {
        'segments': [
            {'path': 'out_001.wav'},
            {'outputs': [{'audio_path': 'out_002.wav'}]},
        ],
        'other': {'files': ['out_003.wav']},
    }

    extracted = _extract_paths_from_result(payload)
    assert [str(item) for item in extracted] == ['out_001.wav', 'out_002.wav', 'out_003.wav']


def test_collect_generated_audio_paths_prefers_explicit_paths_and_sorts(tmp_path: Path) -> None:
    (tmp_path / 'out_010.wav').write_bytes(b'1')
    (tmp_path / 'out_002.wav').write_bytes(b'2')
    (tmp_path / 'out_001.wav').write_bytes(b'3')

    explicit = [Path('out_010.wav'), Path('out_002.wav'), Path('out_001.wav')]
    collected = _collect_generated_audio_paths(tmp_path, explicit)

    assert [item.name for item in collected] == ['out_001.wav', 'out_002.wav', 'out_010.wav']


def test_collect_generated_audio_paths_falls_back_to_glob(tmp_path: Path) -> None:
    (tmp_path / 'out_001.wav').write_bytes(b'1')
    (tmp_path / 'out_003.wav').write_bytes(b'3')
    (tmp_path / 'out_002.wav').write_bytes(b'2')

    collected = _collect_generated_audio_paths(tmp_path, explicit_paths=[])
    assert [item.name for item in collected] == ['out_001.wav', 'out_002.wav', 'out_003.wav']


def _write_silent_wav(path: Path, duration_ms: int = 120, sample_rate: int = 24000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_samples = max(1, int((duration_ms / 1000.0) * sample_rate))
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b'\x00\x00' * total_samples)


def test_synthesize_with_mlx_audio_forwards_sampling_controls_and_seed(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    mlx_seed_calls: list[int] = []
    numpy_seed_calls: list[int] = []

    def fake_generate_audio(**kwargs):
        captured.update(kwargs)
        output_path = Path(f"{kwargs['file_prefix']}_000.wav")
        _write_silent_wav(output_path, duration_ms=160)
        return str(output_path)

    def fake_import(name: str):
        if name == 'mlx_audio.tts.generate':
            return SimpleNamespace(generate_audio=fake_generate_audio)
        if name == 'mlx.core':
            return SimpleNamespace(random=SimpleNamespace(seed=lambda value: mlx_seed_calls.append(int(value))))
        if name == 'numpy':
            return SimpleNamespace(random=SimpleNamespace(seed=lambda value: numpy_seed_calls.append(int(value))))
        raise ImportError(name)

    monkeypatch.setattr(mlx_adapter.importlib, 'import_module', fake_import)
    monkeypatch.setattr(
        mlx_adapter,
        '_get_mlx_audio_model',
        lambda _model_dir: SimpleNamespace(config=SimpleNamespace(tts_model_type='base'), supported_speakers=[]),
    )
    monkeypatch.setattr(mlx_adapter, '_probe_duration_ms', lambda _path: 160)

    output_wav = tmp_path / 'final.wav'
    _synthesize_with_mlx_audio(
        text='hello world',
        model_dir=tmp_path / 'model',
        output_wav=output_wav,
        language='en',
        steps=None,
        max_tokens=3072,
        temperature=0.18,
        top_k=20,
        top_p=0.85,
        repetition_penalty=1.12,
        random_seed=1234567,
        ref_audio_path=None,
        ref_audio_text=None,
        design_instruction=None,
        voice_name=None,
        temp_dir=tmp_path,
    )

    assert output_wav.exists()
    assert captured.get('max_tokens') == 3072
    assert captured.get('temperature') == 0.18
    assert captured.get('top_k') == 20
    assert captured.get('top_p') == 0.85
    assert captured.get('repetition_penalty') == 1.12
    assert captured.get('lang_code') == 'english'
    assert mlx_seed_calls == [1234567]
    assert numpy_seed_calls == [1234567]


def test_resolve_custom_voice_speaker_rejects_unknown_explicit_speaker() -> None:
    with pytest.raises(RuntimeError, match='Unknown CustomVoice speaker'):
        _resolve_custom_voice_speaker(['serena', 'vivian'], 'not-a-speaker')


@pytest.mark.parametrize(
    ('raw_language', 'expected'),
    [
        ('de', 'german'),
        ('de-CH', 'german'),
        ('en', 'english'),
        ('fr', 'french'),
        ('it', 'italian'),
        ('es', 'spanish'),
        ('pt-BR', 'portuguese'),
        ('zh', 'chinese'),
        ('ja', 'japanese'),
        ('ko', 'korean'),
        ('ru', 'russian'),
        ('auto', 'auto'),
        ('german', 'german'),
        ('unknown-lang', 'auto'),
        (None, None),
        ('', None),
    ],
)
def test_normalize_mlx_audio_language_maps_to_qwen3_language_names(raw_language: str | None, expected: str | None) -> None:
    assert _normalize_mlx_audio_language(raw_language) == expected

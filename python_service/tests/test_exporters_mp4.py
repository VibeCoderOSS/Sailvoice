from __future__ import annotations

import math
import shutil
import struct
import subprocess
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.exporters import (
    _build_crawl_frame_plan,
    _normalize_words_to_audio,
    _select_crawl_profile,
    _validate_output_audio_duration,
    export_karaoke_mp4,
    probe_audio_duration_ms,
)
from app.schemas import WordTiming


def _build_settings(tmp_path: Path) -> SimpleNamespace:
    runtime_tmp_dir = tmp_path / 'tmp'
    runtime_tmp_dir.mkdir(parents=True, exist_ok=True)
    return SimpleNamespace(
        runtime_tmp_dir=runtime_tmp_dir,
        mp4_resolution='640x360',
        mp4_fps=30,
    )


def _write_tone_wav(path: Path, duration_ms: int, sample_rate: int = 24000) -> None:
    sample_count = int((duration_ms / 1000.0) * sample_rate)
    frequency = 440.0
    amplitude = 11000
    with wave.open(str(path), 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for idx in range(sample_count):
            value = int(amplitude * math.sin(2 * math.pi * frequency * idx / sample_rate))
            wav_file.writeframes(struct.pack('<h', value))


def _sample_words() -> list[WordTiming]:
    return [
        WordTiming(word='This', startMs=0, endMs=260),
        WordTiming(word='is', startMs=260, endMs=450),
        WordTiming(word='a', startMs=450, endMs=620),
        WordTiming(word='test', startMs=620, endMs=980),
    ]


def test_normalize_words_extends_timeline_to_audio_end() -> None:
    normalized = _normalize_words_to_audio(_sample_words(), 1300)
    assert normalized[-1].endMs == 1300
    assert all(normalized[idx].endMs <= normalized[idx + 1].endMs for idx in range(len(normalized) - 1))


def test_crawl_frame_plan_covers_target_audio_duration() -> None:
    plan = _build_crawl_frame_plan(_sample_words(), 1275, render_fps=30)
    assert plan[0] == 0
    assert plan[-1] == 1275
    assert all(plan[idx] <= plan[idx + 1] for idx in range(len(plan) - 1))
    max_step = max(plan[idx + 1] - plan[idx] for idx in range(len(plan) - 1))
    assert max_step <= 34


def test_select_crawl_profile_is_adaptive(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)

    short = _select_crawl_profile(settings, 30_000)
    medium = _select_crawl_profile(settings, 80_000)
    long = _select_crawl_profile(settings, 180_000)

    assert short.render_fps == 30
    assert medium.render_fps == 20
    assert long.render_fps == 12
    assert short.render_resolution[0] >= medium.render_resolution[0] >= long.render_resolution[0]


@pytest.mark.skipif(
    not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
    reason='ffmpeg/ffprobe not installed',
)
def test_export_karaoke_mp4_audio_tracks_target_duration(tmp_path: Path) -> None:
    settings = _build_settings(tmp_path)
    input_wav = tmp_path / 'input.wav'
    output_mp4 = tmp_path / 'output.mp4'
    _write_tone_wav(input_wav, duration_ms=1300)

    warning = export_karaoke_mp4(
        settings=settings,
        input_audio=input_wav,
        words=_sample_words(),
        output_mp4=output_mp4,
    )

    assert output_mp4.exists()
    assert warning is None or warning.startswith('Crawl renderer failed; plain fallback used')
    target_ms = probe_audio_duration_ms(input_wav)
    rendered_ms = _validate_output_audio_duration(output_mp4, target_ms, tolerance_ms=33)
    assert rendered_ms + 33 >= target_ms


@pytest.mark.skipif(
    not shutil.which('ffmpeg') or not shutil.which('ffprobe'),
    reason='ffmpeg/ffprobe not installed',
)
def test_validate_output_audio_duration_raises_on_underrun(tmp_path: Path) -> None:
    mp4_path = tmp_path / 'short.mp4'
    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel',
            'error',
            '-f',
            'lavfi',
            '-i',
            'color=c=#101612:s=320x240:r=30:d=0.30',
            '-f',
            'lavfi',
            '-i',
            'anullsrc=r=24000:cl=mono',
            '-map',
            '0:v:0',
            '-map',
            '1:a:0',
            '-t',
            '0.30',
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-c:a',
            'aac',
            str(mp4_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(RuntimeError, match='MP4 audio underrun detected'):
        _validate_output_audio_duration(mp4_path, target_audio_ms=900, tolerance_ms=33)

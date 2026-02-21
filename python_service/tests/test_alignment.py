from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.alignment import WhisperWordAligner


def build_settings():
    return SimpleNamespace(
        align_python_bin=Path('/tmp/non-existent-align-python'),
        alignment_model_dir=Path('/tmp/qwen3-align-models'),
        alignment_languages=('de', 'en'),
        alignment_min_coverage=0.97,
    )


def build_aligner() -> WhisperWordAligner:
    return WhisperWordAligner(build_settings())


def test_parse_timed_tokens_strict_success() -> None:
    aligner = build_aligner()
    parsed = aligner._parse_timed_tokens(
        [
            {'word': 'Hallo', 'startMs': 0, 'endMs': 100, 'confidence': 0.93},
            {'word': 'Welt', 'startMs': 100, 'endMs': 230, 'confidence': 0.91},
        ]
    )

    assert len(parsed) == 2
    assert parsed[0].word == 'Hallo'
    assert parsed[0].confidence == pytest.approx(0.93, abs=0.0001)
    assert parsed[1].start_ms == 100
    assert parsed[1].end_ms == 230


def test_parse_timed_tokens_rejects_non_monotonic_timeline() -> None:
    aligner = build_aligner()
    with pytest.raises(RuntimeError, match='Non-monotonic alignment detected'):
        aligner._parse_timed_tokens(
            [
                {'word': 'eins', 'startMs': 0, 'endMs': 100},
                {'word': 'zwei', 'startMs': 90, 'endMs': 180},
            ]
        )


def test_align_audio_returns_low_coverage_result(monkeypatch: pytest.MonkeyPatch) -> None:
    aligner = build_aligner()

    def fake_request(_: dict[str, object]) -> dict[str, object]:
        return {
            'ok': True,
            'coverage': 0.5,
            'matchedWords': 1,
            'sourceWords': 3,
            'alignedWords': 1,
            'tokens': [
                {'word': 'eins', 'startMs': 0, 'endMs': 120},
            ],
        }

    monkeypatch.setattr(aligner, '_request', fake_request)

    result = aligner.align_audio(Path('/tmp/does-not-matter.wav'), 'eins zwei drei', 'de')
    assert result.coverage == pytest.approx(0.5, abs=0.0001)
    assert result.matched_words == 1
    assert result.source_words == 3
    assert result.aligned_words == 1
    assert len(result.tokens) == 1

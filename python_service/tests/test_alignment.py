from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app import alignment as alignment_module
from app.alignment import TimedToken, WhisperWordAligner
from app.manager import _alignment_needs_retry, _build_alignment_retry_candidates


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

    def fake_request(_: dict[str, object], *, timeout_sec: float) -> dict[str, object]:
        assert timeout_sec >= 45.0
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


def test_request_times_out_and_restarts_worker(monkeypatch: pytest.MonkeyPatch) -> None:
    aligner = build_aligner()
    writes: list[str] = []
    closed = {'value': False}

    class FakeStdin:
        def write(self, data: str) -> None:
            writes.append(data)

        def flush(self) -> None:
            return None

    class FakeStdout:
        def readline(self) -> str:
            return ''

    fake_process = SimpleNamespace(
        stdin=FakeStdin(),
        stdout=FakeStdout(),
        poll=lambda: None,
    )

    monkeypatch.setattr(aligner, '_ensure_worker', lambda: fake_process)
    monkeypatch.setattr(alignment_module.select, 'select', lambda *_args, **_kwargs: ([], [], []))
    monkeypatch.setattr(aligner, 'close', lambda: closed.__setitem__('value', True))

    with pytest.raises(RuntimeError, match='timed out'):
        aligner._request({'op': 'probe'}, timeout_sec=0.01)

    assert writes
    assert closed['value'] is True


def test_alignment_retry_candidates_include_normalized_and_raw() -> None:
    candidates = _build_alignment_retry_candidates(
        'Albert Einstein — Kurzer Überblick',
        'Albert Einstein Kurzer Überblick Original',
    )
    labels = [label for label, _ in candidates]
    values = [value for _, value in candidates]
    assert labels[0] == 'primary'
    assert 'normalized' in labels
    assert 'raw_chunk' in labels
    assert len(values) == len(set(values))


def test_alignment_retry_candidates_skip_empty_values() -> None:
    candidates = _build_alignment_retry_candidates('   ', None)
    assert candidates == []


def test_alignment_retry_triggered_by_low_coverage() -> None:
    tokens = [TimedToken(word='eins', start_ms=0, end_ms=800, confidence=0.9)]
    assert _alignment_needs_retry(
        coverage=0.91,
        chunk_duration_ms=1_000,
        tokens=tokens,
        coverage_threshold=0.94,
        tail_gap_threshold_ms=2_500,
    )


def test_alignment_retry_triggered_by_tail_gap() -> None:
    tokens = [TimedToken(word='eins', start_ms=0, end_ms=500, confidence=0.9)]
    assert _alignment_needs_retry(
        coverage=0.99,
        chunk_duration_ms=3_400,
        tokens=tokens,
        coverage_threshold=0.94,
        tail_gap_threshold_ms=2_500,
    )

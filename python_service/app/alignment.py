from __future__ import annotations

import json
import os
import select
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from typing import Any
from uuid import uuid4

from .config import Settings


_ALIGN_INSTALL_HINT = (
    'Alignment runtime missing. Create runtime/.venv-align and install: '
    'source runtime/.venv-align/bin/activate && pip install -r python_service/requirements-align.txt'
)


@dataclass(slots=True)
class TimedToken:
    word: str
    start_ms: int
    end_ms: int
    confidence: float | None = None


@dataclass(slots=True)
class AlignmentResult:
    tokens: list[TimedToken]
    coverage: float
    matched_words: int
    source_words: int
    aligned_words: int


class WhisperWordAligner:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._process: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def close(self) -> None:
        if self._process is None:
            return
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None

    def warmup(self, languages: tuple[str, ...] | None = None) -> dict[str, Any]:
        selected_languages = tuple(
            item.strip().lower()
            for item in (languages or self.settings.alignment_languages)
            if item.strip()
        ) or ('de', 'en')
        payload = {
            'op': 'warmup',
            'languages': list(selected_languages),
            'minCoverage': self.settings.alignment_min_coverage,
            'modelDir': str(self.settings.alignment_model_dir),
        }
        return self._request(payload, timeout_sec=120.0)

    def probe(self) -> dict[str, Any]:
        payload = {
            'op': 'probe',
            'modelDir': str(self.settings.alignment_model_dir),
        }
        return self._request(payload, timeout_sec=15.0)

    def align_audio(self, audio_path: Path, text: str, language: str | None) -> AlignmentResult:
        clean_text = text.strip()
        if not clean_text:
            raise RuntimeError('Alignment source text is empty.')

        request_payload = {
            'op': 'align',
            'id': uuid4().hex,
            'audioPath': str(audio_path),
            'text': clean_text,
            'language': (language or '').strip().lower() or None,
            'minCoverage': self.settings.alignment_min_coverage,
        }
        audio_duration_seconds = self._probe_audio_duration_seconds(audio_path)
        timeout_sec = min(180.0, max(45.0, audio_duration_seconds * 2.5))
        response = self._request(request_payload, timeout_sec=timeout_sec)
        coverage = float(response.get('coverage') or 0.0)
        matched_words = int(response.get('matchedWords') or 0)
        source_words = int(response.get('sourceWords') or 0)
        aligned_words = int(response.get('alignedWords') or 0)
        tokens = self._parse_timed_tokens(response.get('tokens') or [])
        if not tokens:
            raise RuntimeError('WhisperX returned no aligned words.')
        return AlignmentResult(
            tokens=tokens,
            coverage=coverage,
            matched_words=matched_words,
            source_words=source_words,
            aligned_words=aligned_words,
        )

    def _request(self, payload: dict[str, Any], *, timeout_sec: float) -> dict[str, Any]:
        with self._lock:
            process = self._ensure_worker()
            serialized = json.dumps(payload, ensure_ascii=True)
            assert process.stdin is not None
            try:
                process.stdin.write(f'{serialized}\n')
                process.stdin.flush()
            except OSError as exc:
                code = process.poll()
                self._process = None
                raise RuntimeError(
                    f'Alignment worker is not available (code={code}). {_ALIGN_INSTALL_HINT}'
                ) from exc

            assert process.stdout is not None
            response: dict[str, Any] | None = None
            deadline = monotonic() + max(1.0, timeout_sec)
            operation = str(payload.get('op') or 'request')
            for _ in range(64):
                remaining = deadline - monotonic()
                if remaining <= 0:
                    self.close()
                    raise RuntimeError(
                        f'Alignment {operation} request timed out after {timeout_sec:.0f}s. Worker restarted.'
                    )
                ready, _, _ = select.select([process.stdout], [], [], remaining)
                if not ready:
                    self.close()
                    raise RuntimeError(
                        f'Alignment {operation} request timed out after {timeout_sec:.0f}s. Worker restarted.'
                    )
                line = process.stdout.readline()
                if not line:
                    break
                try:
                    parsed = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if isinstance(parsed, dict) and 'ok' in parsed:
                    response = parsed
                    break

            if response is None:
                code = process.poll()
                self._process = None
                raise RuntimeError(f'Alignment worker exited unexpectedly (code={code}).')

            if not response.get('ok', False):
                message = str(response.get('error') or 'Unknown alignment failure')
                raise RuntimeError(message)
            return response

    @staticmethod
    def _probe_audio_duration_seconds(audio_path: Path) -> float:
        try:
            completed = subprocess.run(
                [
                    'ffprobe',
                    '-v',
                    'error',
                    '-show_entries',
                    'format=duration',
                    '-of',
                    'default=noprint_wrappers=1:nokey=1',
                    str(audio_path),
                ],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            )
            return max(0.0, float((completed.stdout or '0').strip() or 0.0))
        except Exception:
            return 60.0

    def _ensure_worker(self) -> subprocess.Popen[str]:
        if self._process is not None and self._process.poll() is None:
            return self._process

        python_bin = self.settings.align_python_bin
        if not python_bin.exists():
            raise RuntimeError(f'{_ALIGN_INSTALL_HINT} (missing: {python_bin})')

        worker_script = Path(__file__).with_name('alignment_worker.py')
        env = os.environ.copy()
        env.setdefault('TTS_ALIGN_MODEL_DIR', str(self.settings.alignment_model_dir))
        env.setdefault('TTS_ALIGN_MIN_COVERAGE', str(self.settings.alignment_min_coverage))

        process = subprocess.Popen(
            [str(python_bin), str(worker_script)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            cwd=str(Path(__file__).resolve().parents[2]),
            env=env,
            bufsize=1,
        )
        self._process = process
        return process

    @staticmethod
    def _parse_timed_tokens(raw_tokens: list[dict[str, Any]]) -> list[TimedToken]:
        parsed: list[TimedToken] = []
        previous_end = 0
        for item in raw_tokens:
            word = str(item.get('word') or '').strip()
            if not word:
                continue
            start_ms = int(item.get('startMs', 0))
            end_ms = int(item.get('endMs', 0))
            confidence_raw = item.get('confidence')
            confidence = float(confidence_raw) if confidence_raw is not None else None
            if start_ms < 0:
                raise RuntimeError(f'Invalid alignment start time: {start_ms}')
            if end_ms <= start_ms:
                raise RuntimeError(f'Invalid alignment time range: {start_ms}..{end_ms}')
            if start_ms < previous_end:
                raise RuntimeError(
                    f'Non-monotonic alignment detected ({start_ms} < previous end {previous_end}).'
                )
            previous_end = end_ms
            parsed.append(TimedToken(word=word, start_ms=start_ms, end_ms=end_ms, confidence=confidence))

        return parsed

from __future__ import annotations

import importlib
import inspect
import json
import os
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _json_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, ensure_ascii=True) + '\n')
    sys.stdout.flush()


def _normalize_token(value: str) -> str:
    return re.sub(r'\W+', '', value, flags=re.UNICODE).strip().lower()


def _source_tokens(text: str) -> list[str]:
    tokens: list[str] = []
    for raw in re.findall(r'\S+', text):
        normalized = _normalize_token(raw)
        if normalized:
            tokens.append(normalized)
    return tokens


def _normalize_language(language: str | None) -> str:
    candidate = (language or '').strip().lower()
    if not candidate:
        return 'en'
    if '-' in candidate:
        candidate = candidate.split('-', maxsplit=1)[0]
    if '_' in candidate:
        candidate = candidate.split('_', maxsplit=1)[0]
    return candidate or 'en'


def _duration_seconds(audio_samples: Any) -> float:
    if hasattr(audio_samples, 'shape') and len(audio_samples.shape) > 0:
        sample_count = int(audio_samples.shape[0])
        return max(0.001, sample_count / 16000.0)
    try:
        return max(0.001, len(audio_samples) / 16000.0)
    except Exception:
        return 0.001


@dataclass(slots=True)
class AlignmentWorker:
    model_dir: Path
    package_dir: Path
    _alignment_module: Any
    _audio_module: Any
    _align_models: dict[str, tuple[Any, Any]]
    _loaded_modules: tuple[str, ...]

    @classmethod
    def create(cls) -> 'AlignmentWorker':
        try:
            package_dir, alignment_module, audio_module, loaded_modules = cls._load_whisperx_alignment_modules()
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                'whisperx alignment module import failed for strict forced alignment '
                f'({exc.__class__.__name__}: {exc}). '
                'Install runtime/.venv-align from python_service/requirements-align.txt'
            ) from exc

        model_dir = Path(os.getenv('TTS_ALIGN_MODEL_DIR', 'runtime/models/whisperx')).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)
        return cls(
            model_dir=model_dir,
            package_dir=package_dir,
            _alignment_module=alignment_module,
            _audio_module=audio_module,
            _align_models={},
            _loaded_modules=loaded_modules,
        )

    @staticmethod
    def _discover_whisperx_package_dir() -> Path:
        for entry in sys.path:
            base = Path(entry or '.').resolve()
            candidate = base / 'whisperx'
            if not candidate.is_dir():
                continue
            if (candidate / 'alignment.py').exists() and (candidate / 'audio.py').exists():
                return candidate
        raise RuntimeError('whisperx package path not found in alignment runtime.')

    @staticmethod
    def _reset_whisperx_modules() -> None:
        for module_name in list(sys.modules.keys()):
            if module_name == 'whisperx' or module_name.startswith('whisperx.'):
                del sys.modules[module_name]

    @classmethod
    def _load_whisperx_alignment_modules(cls) -> tuple[Path, Any, Any, tuple[str, ...]]:
        package_dir = cls._discover_whisperx_package_dir()
        cls._reset_whisperx_modules()

        package_stub = types.ModuleType('whisperx')
        package_stub.__path__ = [str(package_dir)]
        package_stub.__package__ = 'whisperx'
        sys.modules['whisperx'] = package_stub

        loaded_modules: list[str] = []
        try:
            alignment_module = importlib.import_module('whisperx.alignment')
            loaded_modules.append('whisperx.alignment')
            audio_module = importlib.import_module('whisperx.audio')
            loaded_modules.append('whisperx.audio')
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f'packageDir={package_dir}; loaded={loaded_modules}; '
                f'cause={exc.__class__.__name__}: {exc}'
            ) from exc

        return package_dir, alignment_module, audio_module, tuple(loaded_modules)

    def probe(self) -> dict[str, Any]:
        return {
            'ok': True,
            'op': 'probe',
            'modelDir': str(self.model_dir),
            'packageDir': str(self.package_dir),
            'loadedModules': list(self._loaded_modules),
        }

    def warmup(self, languages: list[str]) -> dict[str, Any]:
        loaded: list[str] = []
        for language in languages:
            normalized = _normalize_language(language)
            self._get_align_model(normalized)
            loaded.append(normalized)
        return {
            'ok': True,
            'op': 'warmup',
            'languages': loaded,
            'modelDir': str(self.model_dir),
        }

    def align(
        self,
        audio_path: str,
        text: str,
        language: str | None,
        min_coverage: float,
    ) -> dict[str, Any]:
        clean_text = text.strip()
        if not clean_text:
            raise RuntimeError('Alignment text is empty.')
        path = Path(audio_path)
        if not path.exists():
            raise RuntimeError(f'Audio path does not exist: {audio_path}')

        normalized_language = _normalize_language(language)
        align_model, metadata = self._get_align_model(normalized_language)
        audio = self._audio_module.load_audio(str(path))
        duration_sec = _duration_seconds(audio)
        segments = [{'text': clean_text, 'start': 0.0, 'end': duration_sec}]

        aligned = self._call_align(segments=segments, align_model=align_model, metadata=metadata, audio=audio)
        words = self._extract_words(aligned)

        source_tokens = _source_tokens(clean_text)
        aligned_tokens = [_normalize_token(item.get('word', '')) for item in words]
        aligned_tokens = [token for token in aligned_tokens if token]

        matched_words = 0
        cursor = 0
        for source_token in source_tokens:
            search_end = min(len(aligned_tokens), cursor + 12)
            found_at = None
            for idx in range(cursor, search_end):
                if aligned_tokens[idx] == source_token:
                    found_at = idx
                    break
            if found_at is not None:
                matched_words += 1
                cursor = found_at + 1

        source_words = len(source_tokens)
        aligned_words = len(aligned_tokens)
        coverage = matched_words / max(1, source_words)

        return {
            'ok': True,
            'op': 'align',
            'language': normalized_language,
            'coverage': coverage,
            'matchedWords': matched_words,
            'sourceWords': source_words,
            'alignedWords': aligned_words,
            'minCoverage': min_coverage,
            'tokens': words,
        }

    def _extract_words(self, aligned_payload: dict[str, Any]) -> list[dict[str, Any]]:
        words_raw: list[dict[str, Any]] = []
        word_segments = aligned_payload.get('word_segments')
        if isinstance(word_segments, list):
            words_raw.extend(item for item in word_segments if isinstance(item, dict))

        if not words_raw:
            for segment in aligned_payload.get('segments', []):
                if not isinstance(segment, dict):
                    continue
                for item in segment.get('words', []) or []:
                    if isinstance(item, dict):
                        words_raw.append(item)

        parsed: list[dict[str, Any]] = []
        for item in words_raw:
            word = str(item.get('word') or '').strip()
            start = item.get('start')
            end = item.get('end')
            if not word or start is None or end is None:
                continue
            start_ms = max(0, int(round(float(start) * 1000)))
            end_ms = max(start_ms + 1, int(round(float(end) * 1000)))
            confidence_raw = item.get('score')
            confidence = float(confidence_raw) if confidence_raw is not None else None
            parsed.append(
                {
                    'word': word,
                    'startMs': start_ms,
                    'endMs': end_ms,
                    'confidence': confidence,
                }
            )

        parsed.sort(key=lambda item: (item['startMs'], item['endMs']))
        return parsed

    def _get_align_model(self, language: str) -> tuple[Any, Any]:
        cached = self._align_models.get(language)
        if cached is not None:
            return cached

        kwargs = {'language_code': language, 'device': 'cpu', 'model_dir': str(self.model_dir)}
        try:
            signature = inspect.signature(self._alignment_module.load_align_model)
            supported_kwargs = {key: value for key, value in kwargs.items() if key in signature.parameters}
        except Exception:
            supported_kwargs = kwargs

        model, metadata = self._alignment_module.load_align_model(**supported_kwargs)
        self._align_models[language] = (model, metadata)
        return model, metadata

    def _call_align(
        self,
        segments: list[dict[str, Any]],
        align_model: Any,
        metadata: Any,
        audio: Any,
    ) -> dict[str, Any]:
        try:
            return self._alignment_module.align(
                segments,
                align_model,
                metadata,
                audio,
                'cpu',
                return_char_alignments=False,
                print_progress=False,
            )
        except TypeError:
            try:
                return self._alignment_module.align(
                    segments,
                    align_model,
                    metadata,
                    audio,
                    'cpu',
                    return_char_alignments=False,
                )
            except TypeError:
                return self._alignment_module.align(
                    segments,
                    align_model,
                    metadata,
                    audio,
                    'cpu',
                )


def main() -> int:
    try:
        worker = AlignmentWorker.create()
    except Exception as exc:  # noqa: BLE001
        _json_response({'ok': False, 'error': str(exc)})
        return 1

    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
            operation = payload.get('op')
            if operation == 'probe':
                _json_response(worker.probe())
                continue
            if operation == 'warmup':
                languages = payload.get('languages') or ['de', 'en']
                result = worker.warmup([str(item) for item in languages])
                _json_response(result)
                continue
            if operation == 'align':
                result = worker.align(
                    audio_path=str(payload.get('audioPath') or ''),
                    text=str(payload.get('text') or ''),
                    language=payload.get('language'),
                    min_coverage=float(payload.get('minCoverage') or 0.97),
                )
                if payload.get('id'):
                    result['id'] = payload['id']
                _json_response(result)
                continue

            _json_response({'ok': False, 'error': f'Unknown operation: {operation}'})
        except Exception as exc:  # noqa: BLE001
            _json_response({'ok': False, 'error': str(exc)})
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import queue
import re
import subprocess
import sys
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import chain
from pathlib import Path
from typing import Any
from uuid import uuid4

import psutil

from .alignment import AlignmentResult, TimedToken, WhisperWordAligner
from .config import MODEL_REPOS, WHISPERX_MODEL_REPO, Settings
from .exporters import (
    concat_audio_chunks,
    export_karaoke_mp4,
    export_mp3,
    probe_audio_duration_ms,
    trim_chunk_silence_inplace,
    write_alignment,
    write_transcript,
)
from .language import DetectionResult, detect_language
from .pdf_utils import ChunkBreak, ExtractedWord, PdfChunkItem, TextChunk, iter_pdf_chunks
from .runtime_guard import detect_runtime_status
from .schemas import (
    Artifact,
    JobRequestPdf,
    JobRequestText,
    JobStatus,
    VoiceCreateRequest,
    VoiceDesignCreateRequest,
    VoiceItem,
    VoiceUpdateRequest,
    WordTiming,
)
from .storage import PersistentStore
from .tts_engine import MlxQwenEngine, chunk_text


def utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


def _clean_token(value: str) -> str:
    return re.sub(r'\W+', '', value).lower()


def _build_pdf_synthesis_text(chunk: TextChunk) -> str:
    if not chunk.words:
        return chunk.text.strip()

    break_map: dict[int, tuple[str, int]] = {}
    for item in chunk.breaks:
        index = min(max(0, int(item.afterWordIndex)), len(chunk.words) - 1)
        pause_ms = max(0, int(item.durationMs))
        if pause_ms <= 0:
            continue
        reason = str(item.reason or '').strip().lower() or 'paragraph'
        if reason not in {'heading', 'paragraph'}:
            reason = 'paragraph'
        existing = break_map.get(index)
        if existing is None:
            break_map[index] = (reason, pause_ms)
            continue
        existing_reason, existing_ms = existing
        if reason == 'heading' and existing_reason != 'heading':
            break_map[index] = (reason, pause_ms)
            continue
        if pause_ms > existing_ms:
            break_map[index] = (reason, pause_ms)

    rendered_tokens: list[str] = []
    sentence_end_re = re.compile(r'[.!?…:;]$')
    clause_end_re = re.compile(r'[,.!?;:]$')
    for index, word in enumerate(chunk.words):
        token = word.word.strip()
        if not token:
            continue
        marker = break_map.get(index)
        if marker is None:
            rendered_tokens.append(token)
            continue
        reason, _pause_ms = marker
        if reason == 'heading':
            if token and not sentence_end_re.search(token):
                token = f'{token}.'
        else:
            if token and not clause_end_re.search(token):
                token = f'{token},'
        rendered_tokens.append(token)

    return ' '.join(rendered_tokens).strip()


def _normalize_alignment_retry_text(text: str) -> str:
    collapsed = re.sub(r'\s+', ' ', text).strip()
    if not collapsed:
        return ''
    cleaned = re.sub(r'[“”"‘’`´~^*_#|/\\<>\[\]{}()–—•·…]', ' ', collapsed)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _build_alignment_retry_candidates(primary_text: str, raw_chunk_text: str | None) -> list[tuple[str, str]]:
    candidates: list[tuple[str, str]] = []

    def add(label: str, value: str) -> None:
        normalized = value.strip()
        if not normalized:
            return
        if any(existing == normalized for _, existing in candidates):
            return
        candidates.append((label, normalized))

    add('primary', primary_text)
    add('normalized', _normalize_alignment_retry_text(primary_text))
    if raw_chunk_text:
        add('raw_chunk', raw_chunk_text)
    return candidates


def _is_sentence_boundary_token(token: str) -> bool:
    value = token.strip()
    if not value:
        return False
    return bool(re.search(r'[.!?…:;]["\'\)\]]*$', value))


def _dedupe_breaks(items: list[ChunkBreak]) -> list[ChunkBreak]:
    keep: dict[int, ChunkBreak] = {}
    for item in items:
        idx = int(item.afterWordIndex)
        if idx < 0:
            continue
        existing = keep.get(idx)
        if existing is None or int(item.durationMs) > int(existing.durationMs):
            keep[idx] = ChunkBreak(
                afterWordIndex=idx,
                durationMs=max(0, int(item.durationMs)),
                reason='heading' if item.reason == 'heading' else 'paragraph',
            )
    return [keep[idx] for idx in sorted(keep)]


def _split_pdf_chunk_for_synthesis(
    chunk: TextChunk,
    *,
    max_words: int,
    max_chars: int,
) -> list[TextChunk]:
    safe_max_words = max(10, int(max_words))
    safe_max_chars = max(180, int(max_chars))

    if not chunk.words:
        cleaned = chunk.text.strip()
        if not cleaned:
            return []
        return [TextChunk(text=item, words=[], breaks=[]) for item in chunk_text(cleaned, max_chars=safe_max_chars)]

    words = chunk.words
    total = len(words)

    break_indices = {int(item.afterWordIndex) for item in chunk.breaks if 0 <= int(item.afterWordIndex) < total}
    sentence_indices = {
        idx for idx, item in enumerate(words) if _is_sentence_boundary_token(item.word)
    }

    subchunks: list[TextChunk] = []
    start = 0
    while start < total:
        hard_word_end = min(total, start + safe_max_words)

        char_end = start
        char_budget = 0
        for idx in range(start, hard_word_end):
            token = words[idx].word.strip()
            if not token:
                continue
            extra = len(token) + (1 if char_budget > 0 else 0)
            if char_end > start and (char_budget + extra) > safe_max_chars:
                break
            char_budget += extra
            char_end = idx + 1

        if char_end <= start:
            char_end = min(total, start + 1)

        candidate_end = min(hard_word_end, char_end)
        if candidate_end < total:
            chosen_boundary = None
            for idx in range(candidate_end - 1, start, -1):
                if idx in break_indices:
                    chosen_boundary = idx + 1
                    break
            if chosen_boundary is None:
                for idx in range(candidate_end - 1, start, -1):
                    if idx in sentence_indices:
                        chosen_boundary = idx + 1
                        break
            end = chosen_boundary if chosen_boundary is not None else candidate_end
        else:
            end = total

        if end <= start:
            end = min(total, start + 1)

        segment_words = words[start:end]
        segment_text = ' '.join(item.word.strip() for item in segment_words if item.word.strip()).strip()
        if not segment_text:
            start = end
            continue

        segment_breaks = [
            ChunkBreak(
                afterWordIndex=(int(item.afterWordIndex) - start),
                durationMs=int(item.durationMs),
                reason='heading' if item.reason == 'heading' else 'paragraph',
            )
            for item in chunk.breaks
            if start <= int(item.afterWordIndex) < end
        ]

        subchunks.append(
            TextChunk(
                text=segment_text,
                words=segment_words.copy(),
                breaks=_dedupe_breaks(segment_breaks),
            )
        )
        start = end

    return subchunks


def _alignment_tail_gap_ms(tokens: list[TimedToken], chunk_duration_ms: int) -> int:
    if not tokens:
        return max(0, int(chunk_duration_ms))
    last_end = max(token.end_ms for token in tokens)
    return max(0, int(chunk_duration_ms) - max(0, int(last_end)))


def _alignment_needs_retry(
    *,
    coverage: float,
    chunk_duration_ms: int,
    tokens: list[TimedToken],
    coverage_threshold: float,
    tail_gap_threshold_ms: int,
) -> bool:
    if coverage < coverage_threshold:
        return True
    tail_gap_ms = _alignment_tail_gap_ms(tokens, chunk_duration_ms)
    return tail_gap_ms > tail_gap_threshold_ms


@dataclass(frozen=True, slots=True)
class ChunkGenerationProfile:
    mode: str
    temperature: float | None
    top_k: int | None
    top_p: float | None
    repetition_penalty: float | None
    random_seed: int | None


@dataclass(frozen=True, slots=True)
class ResolvedVoiceRoute:
    runtime_model_id: str
    selected_voice: VoiceItem | None
    speaker: str | None


def _normalize_optional_speaker(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    return normalized or None


def _resolve_voice_route(
    *,
    requested_model_id: str,
    selected_voice: VoiceItem | None,
    requested_speaker: str | None,
    default_speaker: str,
) -> ResolvedVoiceRoute:
    normalized_speaker = _normalize_optional_speaker(requested_speaker)
    if selected_voice is not None:
        if selected_voice.type == 'clone':
            return ResolvedVoiceRoute(runtime_model_id='base', selected_voice=selected_voice, speaker=None)
        if selected_voice.type == 'design':
            return ResolvedVoiceRoute(runtime_model_id='voicedesign', selected_voice=selected_voice, speaker=None)

    if normalized_speaker:
        return ResolvedVoiceRoute(runtime_model_id='customvoice', selected_voice=None, speaker=normalized_speaker)

    if requested_model_id == 'customvoice':
        return ResolvedVoiceRoute(runtime_model_id='customvoice', selected_voice=None, speaker=default_speaker)

    return ResolvedVoiceRoute(runtime_model_id='customvoice', selected_voice=None, speaker=default_speaker)


def _compute_job_seed(
    *,
    job_id: str,
    voice_id: str | None,
    speaker: str | None,
    model_id: str,
    language: str,
    seed_base: int,
) -> int:
    digest = hashlib.sha256(
        f'{seed_base}|{job_id}|{voice_id or "-"}|{speaker or "-"}|{model_id}|{language}'.encode('utf-8')
    ).digest()
    return max(1, int.from_bytes(digest[:8], byteorder='big') % 2_147_483_647)


def _build_chunk_generation_profile(
    *,
    job_id: str,
    voice_id: str | None,
    speaker: str | None,
    model_id: str,
    language: str,
    chunk_count: int,
    consistency_mode: str,
    seed_base: int,
    strict_temperature: float,
    strict_top_k: int,
    strict_top_p: float,
    strict_repetition_penalty: float,
) -> ChunkGenerationProfile:
    mode = (consistency_mode or 'strict').strip().lower()
    if mode not in {'strict', 'balanced', 'natural'}:
        mode = 'strict'

    # Keep strict consistency as default, and force strict for multi-chunk jobs.
    effective_mode = 'strict' if mode == 'strict' or chunk_count > 1 else mode
    seed = _compute_job_seed(
        job_id=job_id,
        voice_id=voice_id,
        speaker=speaker,
        model_id=model_id,
        language=language,
        seed_base=seed_base,
    )

    if effective_mode == 'strict':
        return ChunkGenerationProfile(
            mode='strict',
            temperature=strict_temperature,
            top_k=strict_top_k,
            top_p=strict_top_p,
            repetition_penalty=strict_repetition_penalty,
            random_seed=seed,
        )

    if effective_mode == 'balanced':
        return ChunkGenerationProfile(
            mode='balanced',
            temperature=0.45,
            top_k=40,
            top_p=0.92,
            repetition_penalty=1.08,
            random_seed=seed,
        )

    return ChunkGenerationProfile(
        mode='natural',
        temperature=None,
        top_k=None,
        top_p=None,
        repetition_penalty=None,
        random_seed=None,
    )


class JobCanceledError(RuntimeError):
    pass


class JobManager:
    @staticmethod
    def _consume_background_exception(task: asyncio.Task[Any]) -> None:
        with suppress(asyncio.CancelledError):
            task.exception()

    def _on_warmup_task_done(self, task: asyncio.Task[Any]) -> None:
        self._consume_background_exception(task)
        if self._warmup_task is task:
            self._warmup_task = None

    def __init__(self, settings: Settings):
        self.settings = settings
        self.store = PersistentStore(settings)
        self.engine = MlxQwenEngine(settings)
        self.aligner = WhisperWordAligner(settings)
        self.job_inputs: dict[str, dict[str, Any]] = {}
        self.pending_jobs: deque[str] = deque()
        self.queue_event = asyncio.Event()
        self.queue_lock = asyncio.Lock()
        self.subscribers: dict[str, set[asyncio.Queue[dict[str, Any]]]] = {}
        self.language_waiters: dict[str, asyncio.Future[str]] = {}
        self.mp4_tasks: dict[str, asyncio.Task[None]] = {}
        self.worker_task: asyncio.Task[None] | None = None
        self.running_job_id: str | None = None
        self.running_job_task: asyncio.Task[None] | None = None
        self.cancel_requests: dict[str, str] = {}
        self.alignment_runtime_verified = False
        self.alignment_probe_error: str | None = None
        self.alignment_probe_at: datetime | None = None
        self._warmup_state: str = 'idle'
        self._warmup_error: str | None = None
        self._warmup_at: datetime | None = None
        self._warmup_task: asyncio.Task[None] | None = None
        self._warmup_lock = asyncio.Lock()
        self._job_performance_profiles: dict[str, str] = {}
        self._reference_stt_model: Any | None = None
        self.lock = asyncio.Lock()

    async def start(self) -> None:
        if self.worker_task is None:
            await self._recover_interrupted_jobs()
            self.worker_task = asyncio.create_task(self._worker_loop(), name='qwen3-tts-worker')

    async def stop(self) -> None:
        if self.worker_task:
            self.worker_task.cancel()
            with suppress(asyncio.CancelledError):
                await self.worker_task
            self.worker_task = None
        if self._warmup_task and not self._warmup_task.done():
            self._warmup_task.cancel()
            with suppress(asyncio.CancelledError):
                await self._warmup_task
        self._warmup_task = None
        for task in list(self.mp4_tasks.values()):
            task.cancel()
        for task in list(self.mp4_tasks.values()):
            with suppress(asyncio.CancelledError):
                await task
        self.mp4_tasks.clear()
        self.cancel_requests.clear()
        async with self.queue_lock:
            self.pending_jobs.clear()
            self.queue_event.clear()
            self.running_job_id = None
            self.running_job_task = None
        self._job_performance_profiles.clear()
        self.aligner.close()
        self.alignment_runtime_verified = False
        self.alignment_probe_error = None
        self.alignment_probe_at = None

    async def _recover_interrupted_jobs(self) -> None:
        interrupted_message = 'Interrupted by app restart.'
        async with self.lock:
            jobs = self.store.list_jobs()
            for job in jobs:
                job_active = job.state in {'queued', 'running', 'waiting_language', 'canceling'}
                mp4_active = job.mp4State in {'queued', 'running'}
                alignment_active = job.alignmentState in {'queued', 'running'}
                if not job_active and not mp4_active and not alignment_active:
                    continue

                updated = job.model_copy(
                    update={
                        'state': 'failed' if job_active else job.state,
                        'error': interrupted_message if job_active else job.error,
                        'statusMessage': interrupted_message,
                        'phase': 'failed' if job_active else job.phase,
                        'phaseProgress': 1.0 if job_active else job.phaseProgress,
                        'alignmentState': 'failed' if alignment_active else job.alignmentState,
                        'alignmentError': interrupted_message if alignment_active else job.alignmentError,
                        'mp4State': 'failed' if mp4_active else job.mp4State,
                        'mp4Error': interrupted_message if mp4_active else job.mp4Error,
                        'mp4Progress': None if mp4_active else job.mp4Progress,
                        'cancelReason': None,
                        'updatedAt': utc_now(),
                    }
                )
                self.store.upsert_job(updated)

    async def _enqueue_job(self, job_id: str) -> None:
        async with self.queue_lock:
            if job_id not in self.pending_jobs:
                self.pending_jobs.append(job_id)
            self.queue_event.set()

    async def _remove_pending_job(self, job_id: str) -> bool:
        async with self.queue_lock:
            try:
                self.pending_jobs.remove(job_id)
            except ValueError:
                removed = False
            else:
                removed = True
            if not self.pending_jobs:
                self.queue_event.clear()
            return removed

    async def _get_next_pending_job(self) -> str:
        while True:
            await self.queue_event.wait()
            async with self.queue_lock:
                if self.pending_jobs:
                    job_id = self.pending_jobs.popleft()
                    if not self.pending_jobs:
                        self.queue_event.clear()
                    return job_id
                self.queue_event.clear()

    def _pages_per_chunk(self, performance_profile: str | None = None) -> int:
        profile = performance_profile or self.settings.performance_profile
        return 2 if profile == 'memory' else 4

    def _text_chunk_size(self, performance_profile: str | None = None) -> int:
        profile = performance_profile or self.settings.performance_profile
        return 280 if profile == 'memory' else 420

    async def _resolve_job_performance_profile(self, job_id: str) -> str:
        configured_profile = self.settings.performance_profile if self.settings.performance_profile in {'standard', 'memory'} else 'standard'
        if configured_profile == 'memory':
            self._job_performance_profiles[job_id] = 'memory'
            return 'memory'

        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        if memory_mb > 3600:
            self._job_performance_profiles[job_id] = 'memory'
            warning = 'Memory pressure detected; using memory saver tuning for this job.'
            await self._update_job(job_id, warning=warning)
            await self.publish(job_id, {'type': 'warning', 'message': warning})
            return 'memory'

        self._job_performance_profiles[job_id] = 'standard'
        return 'standard'

    def _alignment_parallelism(self, performance_profile: str) -> int:
        if performance_profile == 'memory':
            return 1

        total_gb = psutil.virtual_memory().total / (1024 ** 3)
        rss_gb = psutil.Process().memory_info().rss / (1024 ** 3)
        if total_gb >= 28 and rss_gb < 14:
            return 3
        return 2

    async def warmup_runtime(
        self,
        models: list[str] | tuple[str, ...] | None = None,
        *,
        include_alignment_probe: bool = True,
    ) -> dict[str, Any]:
        normalized_models = tuple(model for model in (models or ('customvoice',)) if model in MODEL_REPOS)
        if not normalized_models:
            normalized_models = ('customvoice',)

        async with self._warmup_lock:
            if self._warmup_task and not self._warmup_task.done():
                return await self.get_runtime_status()
            if self._warmup_state != 'ready':
                self._warmup_state = 'running'
                self._warmup_error = None
                self._warmup_task = asyncio.create_task(
                    self._run_runtime_warmup(
                        models=normalized_models,
                        include_alignment_probe=include_alignment_probe,
                    ),
                    name='qwen3-tts-runtime-warmup',
                )
                self._warmup_task.add_done_callback(self._on_warmup_task_done)

        return await self.get_runtime_status()

    async def _run_runtime_warmup(
        self,
        *,
        models: tuple[str, ...],
        include_alignment_probe: bool,
    ) -> None:
        try:
            for model_id in models:
                warning = await asyncio.to_thread(self.engine.ensure_model, model_id, False)
                if warning:
                    raise RuntimeError(warning)
                await asyncio.to_thread(self.engine.load_model_into_ram, model_id)

            if include_alignment_probe:
                alignment_ready, alignment_reason = await self._probe_alignment_runtime(force=True)
                if not alignment_ready:
                    raise RuntimeError(alignment_reason or 'Alignment runtime probe failed.')

            self._warmup_state = 'ready'
            self._warmup_error = None
            self._warmup_at = utc_now()
        except Exception as exc:  # noqa: BLE001
            self._warmup_state = 'failed'
            self._warmup_error = str(exc)
            self._warmup_at = utc_now()
            raise

    def _get_reference_stt_model(self):
        if self._reference_stt_model is not None:
            return self._reference_stt_model

        from faster_whisper import WhisperModel

        compute_types = ('int8', 'int8_float32', 'float32')
        last_error: Exception | None = None
        for compute_type in compute_types:
            try:
                self._reference_stt_model = WhisperModel(
                    self.settings.whisper_model_size,
                    device='cpu',
                    compute_type=compute_type,
                    download_root=str(self.settings.whisper_cache_dir),
                )
                return self._reference_stt_model
            except Exception as exc:  # noqa: BLE001
                last_error = exc

        raise RuntimeError(f'Failed to load faster-whisper model: {last_error}')

    def _transcribe_reference_audio(
        self,
        audio_path: Path,
        language_hint: str | None = None,
    ) -> tuple[str | None, str | None]:
        model = self._get_reference_stt_model()

        kwargs: dict[str, Any] = {
            'beam_size': 1,
            'vad_filter': False,
            'word_timestamps': False,
            'condition_on_previous_text': False,
        }
        if language_hint:
            kwargs['language'] = language_hint

        try:
            segments, info = model.transcribe(str(audio_path), **kwargs)
        except TypeError:
            kwargs.pop('condition_on_previous_text', None)
            segments, info = model.transcribe(str(audio_path), **kwargs)

        text_parts = [segment.text.strip() for segment in segments if getattr(segment, 'text', '').strip()]
        transcript = ' '.join(text_parts).strip() or None
        language = (getattr(info, 'language', None) or '').strip().lower() or None
        return transcript, language

    def _resolve_clone_reference_text(
        self,
        voice_ref_path: Path,
        stored_reference_text: str | None,
        fallback_text: str,
        language_hint: str | None,
    ) -> tuple[str, str | None, str | None, bool]:
        if stored_reference_text and stored_reference_text.strip():
            return stored_reference_text.strip(), None, None, False

        try:
            transcript, transcript_language = self._transcribe_reference_audio(
                voice_ref_path,
                language_hint=language_hint,
            )
        except Exception as exc:  # noqa: BLE001
            fallback = fallback_text.strip() or 'reference voice sample'
            warning = (
                f'Voice reference text missing; transcription failed ({exc}). '
                'Using fallback text. Add reference text in Voices for best quality.'
            )
            return fallback, None, warning, False

        if transcript:
            warning = 'Voice reference text missing; auto-transcribed from reference audio for stable clone quality.'
            return transcript, transcript_language, warning, True

        fallback = fallback_text.strip() or 'reference voice sample'
        warning = 'Voice reference text missing and transcription returned empty text. Using fallback text.'
        return fallback, transcript_language, warning, False

    def _validate_voice_selection(self, voice_id: str | None, model_id: str, speaker: str | None = None) -> None:
        normalized_speaker = _normalize_optional_speaker(speaker)
        if voice_id and normalized_speaker:
            raise ValueError('Provide either voiceId or speaker, not both.')
        if normalized_speaker and model_id != 'customvoice':
            raise ValueError('Speaker presets require model "customvoice".')
        if not voice_id:
            return
        voice = self.store.get_voice(voice_id)
        if voice.type == 'design' and model_id != 'voicedesign':
            raise ValueError('Voice Design presets require model "voicedesign".')
        if voice.type == 'clone' and model_id == 'voicedesign':
            raise ValueError('Voice Clone presets are not compatible with model "voicedesign".')

    async def create_text_job(self, request: JobRequestText) -> str:
        self._validate_voice_selection(request.voiceId, request.model, request.speaker)
        job_id = uuid4().hex
        now = utc_now()
        mp4_requested = 'mp4' in request.outputFormats
        job = JobStatus(
            id=job_id,
            state='queued',
            progress=0.0,
            statusMessage='queued',
            createdAt=now,
            updatedAt=now,
            sourceType='text',
            sourceLabel='Text input',
            model=request.model,
            detectedLanguage='auto',
            mp4State='queued' if mp4_requested else 'not_requested',
            mp4Error=None,
            mp4Progress=0.0 if mp4_requested else None,
            alignmentState='queued',
            alignmentError=None,
            alignmentCoverage=None,
            alignmentRetryCount=0,
            audioDurationMs=None,
            artifacts=[],
            phase='queued',
            phaseProgress=0.0,
            alignmentWarning=None,
            words=[],
        )

        async with self.lock:
            self.store.upsert_job(job)
            self.job_inputs[job_id] = {
                'type': 'text',
                'request': request.model_dump(mode='json'),
            }
        await self._enqueue_job(job_id)
        return job_id

    async def create_pdf_job(self, request: JobRequestPdf) -> str:
        self._validate_voice_selection(request.voiceId, request.model, request.speaker)
        job_id = uuid4().hex
        now = utc_now()
        source_label = Path(request.pdfPath).name
        mp4_requested = 'mp4' in request.outputFormats
        job = JobStatus(
            id=job_id,
            state='queued',
            progress=0.0,
            statusMessage='queued',
            createdAt=now,
            updatedAt=now,
            sourceType='pdf',
            sourceLabel=source_label,
            model=request.model,
            detectedLanguage='auto',
            mp4State='queued' if mp4_requested else 'not_requested',
            mp4Error=None,
            mp4Progress=0.0 if mp4_requested else None,
            alignmentState='queued',
            alignmentError=None,
            alignmentCoverage=None,
            alignmentRetryCount=0,
            audioDurationMs=None,
            artifacts=[],
            phase='queued',
            phaseProgress=0.0,
            alignmentWarning=None,
            words=[],
        )

        async with self.lock:
            self.store.upsert_job(job)
            self.job_inputs[job_id] = {
                'type': 'pdf',
                'request': request.model_dump(mode='json'),
            }
        await self._enqueue_job(job_id)
        return job_id

    async def get_job(self, job_id: str) -> JobStatus | None:
        return self.store.get_job(job_id)

    async def list_jobs(self) -> list[JobStatus]:
        return self.store.list_jobs()

    async def delete_job(self, job_id: str) -> None:
        delete_reason = 'Deleted by user.'
        artifact_paths: list[str] = []
        await self._remove_pending_job(job_id)
        async with self.lock:
            job = self.store.get_job(job_id)
            if not job:
                raise KeyError('Job not found')

            self.cancel_requests[job_id] = delete_reason
            artifact_paths = self.store.delete_job(job_id)
            self.job_inputs.pop(job_id, None)
            self._job_performance_profiles.pop(job_id, None)
            waiter = self.language_waiters.pop(job_id, None)
            if waiter and not waiter.done():
                waiter.cancel()
            mp4_task = self.mp4_tasks.pop(job_id, None)
            if mp4_task and not mp4_task.done():
                mp4_task.cancel()

        extra_paths = [
            self.settings.assets_dir / f'{job_id}-final.wav',
            self.settings.assets_dir / f'{job_id}.mp3',
            self.settings.assets_dir / f'{job_id}.mp4',
            self.settings.assets_dir / f'{job_id}-alignment.json',
            self.settings.assets_dir / f'{job_id}-transcript.txt',
        ]

        all_paths = {Path(path) for path in artifact_paths}
        all_paths.update(extra_paths)
        for path in all_paths:
            if path.exists() and path.is_file():
                path.unlink(missing_ok=True)

        await self.publish(job_id, {'type': 'canceled', 'message': delete_reason})

    async def _finalize_job_canceled(
        self,
        job_id: str,
        reason: str,
        *,
        publish_event: bool = True,
    ) -> None:
        task = self.mp4_tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()
        self._job_performance_profiles.pop(job_id, None)
        self.cancel_requests.pop(job_id, None)
        waiter = self.language_waiters.pop(job_id, None)
        if waiter and not waiter.done():
            waiter.cancel()
        async with self.lock:
            job = self.store.get_job(job_id)
            if not job:
                return
            if job.state == 'canceled':
                return
            next_job = job.model_copy(
                update={
                    'state': 'canceled',
                    'cancelReason': reason,
                    'statusMessage': reason,
                    'phase': 'canceled',
                    'phaseProgress': 1.0,
                    'updatedAt': utc_now(),
                    'mp4State': 'failed' if job.mp4State in {'queued', 'running'} else job.mp4State,
                    'mp4Error': reason if job.mp4State in {'queued', 'running'} else job.mp4Error,
                    'mp4Progress': None if job.mp4State in {'queued', 'running'} else job.mp4Progress,
                }
            )
            self.store.upsert_job(next_job)
        if publish_event:
            await self.publish(
                job_id,
                {
                    'type': 'progress',
                    'progress': 1.0,
                    'message': reason,
                    'phase': 'canceled',
                    'phaseProgress': 1.0,
                },
            )
            await self.publish(job_id, {'type': 'canceled', 'message': reason})

    async def _cancel_background_mp4(self, job_id: str, reason: str) -> None:
        task = self.mp4_tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task
        await self._update_job(
            job_id,
            mp4State='failed',
            mp4Error=reason,
            mp4Progress=None,
            warning=reason,
            statusMessage='mp4_background_failed',
            phase='done',
            phaseProgress=1.0,
        )
        await self.publish(job_id, {'type': 'warning', 'message': reason})

    async def cancel_job(self, job_id: str, reason: str = 'Canceled by user.') -> None:
        removed_from_pending = await self._remove_pending_job(job_id)
        waiter_to_cancel: asyncio.Future[str] | None = None
        finalize_immediately = False
        cancel_background_mp4_only = False
        progress_value = 0.0

        async with self.lock:
            job = self.store.get_job(job_id)
            if not job:
                raise KeyError('Job not found')
            progress_value = job.progress

            if job.state in {'done', 'failed', 'canceled'}:
                if job.mp4State in {'queued', 'running'}:
                    cancel_background_mp4_only = True
                else:
                    raise RuntimeError('Cannot cancel a terminal job.')
            elif job.state == 'queued':
                self.cancel_requests[job_id] = reason
                finalize_immediately = removed_from_pending and self.running_job_id != job_id
                if not finalize_immediately:
                    updated = job.model_copy(
                        update={
                            'state': 'canceling',
                            'cancelReason': reason,
                            'statusMessage': 'canceling',
                            'phase': 'canceling',
                            'phaseProgress': job.phaseProgress,
                            'updatedAt': utc_now(),
                        }
                    )
                    self.store.upsert_job(updated)
            elif job.state == 'waiting_language':
                self.cancel_requests[job_id] = reason
                waiter_to_cancel = self.language_waiters.get(job_id)
                finalize_immediately = True
            elif job.state in {'running', 'canceling'}:
                self.cancel_requests[job_id] = reason
                updated = job.model_copy(
                    update={
                        'state': 'canceling',
                        'cancelReason': reason,
                        'statusMessage': 'canceling',
                        'phase': 'canceling',
                        'updatedAt': utc_now(),
                    }
                )
                self.store.upsert_job(updated)
            else:
                raise RuntimeError(f'Cannot cancel job in state {job.state}.')

        if cancel_background_mp4_only:
            await self._cancel_background_mp4(job_id, reason)
            return

        if waiter_to_cancel and not waiter_to_cancel.done():
            waiter_to_cancel.cancel()

        if finalize_immediately:
            await self._finalize_job_canceled(job_id, reason)
            return

        await self.publish(
            job_id,
            {
                'type': 'progress',
                'progress': max(0.0, min(1.0, progress_value)),
                'message': 'canceling',
                'phase': 'canceling',
                'phaseProgress': 1.0,
            },
        )

    async def list_voices(self) -> list[VoiceItem]:
        return self.store.list_voices()

    async def create_voice(self, request: VoiceCreateRequest) -> VoiceItem:
        return self.store.create_voice(
            name=request.name,
            language=request.language,
            audio_path=request.audioPath,
            reference_text=request.referenceText,
        )

    async def create_design_voice(self, request: VoiceDesignCreateRequest) -> VoiceItem:
        return self.store.create_design_voice(
            name=request.name,
            language=request.language,
            description=request.description,
        )

    async def rename_voice(self, voice_id: str, request: VoiceUpdateRequest) -> VoiceItem:
        return self.store.rename_voice(voice_id=voice_id, name=request.name)

    async def delete_voice(self, voice_id: str) -> None:
        self.store.delete_voice(voice_id)

    async def preview_voice(
        self,
        text: str,
        model_id: str,
        voice_id: str | None = None,
        language: str = 'en',
        speaker: str | None = None,
    ) -> str:
        self._validate_voice_selection(voice_id, model_id, speaker)
        runtime_status = detect_runtime_status()
        if not runtime_status.compatible and not self.settings.allow_macos_fallback:
            raise RuntimeError(f'MLX runtime required. {runtime_status.reason}')

        effective_model_id = model_id
        voice_ref_path: Path | None = None
        voice_ref_text: str | None = None
        design_instruction: str | None = None
        design_voice_name: str | None = None
        standard_speaker_name: str | None = None
        effective_language = language

        if voice_id:
            voice = self.store.get_voice(voice_id)
            if voice.language:
                effective_language = voice.language
            if voice.type == 'clone':
                effective_model_id = 'base'
                voice_ref_path = self.store.materialize_voice_reference(voice_id)
                voice_ref_text, transcribed_language, _, persistable = await asyncio.to_thread(
                    self._resolve_clone_reference_text,
                    voice_ref_path,
                    self.store.get_voice_reference_text(voice_id),
                    text[:200],
                    effective_language,
                )
                if not voice.language and transcribed_language:
                    effective_language = transcribed_language
                if persistable:
                    self.store.update_clone_voice_metadata(
                        voice_id=voice_id,
                        reference_text=voice_ref_text,
                        language=transcribed_language if not voice.language else None,
                    )
            elif voice.type == 'design':
                effective_model_id = 'voicedesign'
                design_instruction = (voice.description or '').strip() or None
                design_voice_name = voice.name
        else:
            normalized_speaker = _normalize_optional_speaker(speaker)
            if normalized_speaker:
                effective_model_id = 'customvoice'
                standard_speaker_name = normalized_speaker
            elif model_id == 'customvoice':
                effective_model_id = 'customvoice'
                standard_speaker_name = self.settings.standard_default_speaker

        try:
            model_warning = self.engine.ensure_model(effective_model_id, prefer_official=False)
            if model_warning and not self.settings.allow_macos_fallback:
                raise RuntimeError(model_warning)

            result = self.engine.synthesize_chunk(
                text=text,
                model_id=effective_model_id,
                language=effective_language,
                chunk_index=0,
                voice_reference_path=voice_ref_path,
                voice_reference_text=voice_ref_text,
                voice_design_instruction=design_instruction,
                voice_design_name=standard_speaker_name if effective_model_id == 'customvoice' else design_voice_name,
            )
        finally:
            if voice_ref_path is not None:
                voice_ref_path.unlink(missing_ok=True)

        asset_id = self.store.register_asset(result.wav_path)
        return asset_id

    async def list_models(self) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        for model_id in MODEL_REPOS:
            for source, repo in MODEL_REPOS[model_id].items():
                local_dir = self.settings.model_cache_dir / repo.replace('/', '--')
                downloaded = local_dir.exists() and any(local_dir.iterdir())
                items.append(
                    {
                        'model': model_id,
                        'source': source,
                        'repo': repo,
                        'localPath': str(local_dir),
                        'downloaded': downloaded,
                    }
                )
        whisperx_marker = self._whisperx_ready_marker()
        items.append(
            {
                'model': 'whisperx',
                'source': 'mlx',
                'repo': WHISPERX_MODEL_REPO,
                'localPath': str(self.settings.alignment_model_dir),
                'downloaded': whisperx_marker.exists(),
            }
        )
        return items

    async def download_model(self, model: str, source: str) -> dict[str, Any]:
        if model == 'whisperx':
            try:
                runtime_note = await self._ensure_alignment_models()
            except Exception as exc:  # noqa: BLE001
                return {
                    'model': 'whisperx',
                    'source': 'mlx',
                    'repo': WHISPERX_MODEL_REPO,
                    'localPath': str(self.settings.alignment_model_dir),
                    'downloaded': False,
                    'warning': str(exc),
                }
            return {
                'model': 'whisperx',
                'source': 'mlx',
                'repo': WHISPERX_MODEL_REPO,
                'localPath': str(self.settings.alignment_model_dir),
                'downloaded': True,
                'warning': runtime_note,
            }

        if model not in MODEL_REPOS:
            raise KeyError(f'Unknown model: {model}')
        if source not in MODEL_REPOS[model]:
            raise KeyError(f'Unknown source: {source}')

        prefer_official = source == 'official'
        warning = await asyncio.to_thread(self.engine.ensure_model, model, prefer_official)
        repo = MODEL_REPOS[model][source]
        local_dir = self.settings.model_cache_dir / repo.replace('/', '--')
        downloaded = local_dir.exists() and any(local_dir.iterdir())

        return {
            'model': model,
            'source': source,
            'repo': repo,
            'localPath': str(local_dir),
            'downloaded': downloaded,
            'warning': warning,
        }

    def _alignment_runtime_install_hint(self) -> str:
        return (
            'Run in this project folder: '
            'python3 -m venv runtime/.venv-align && '
            'source runtime/.venv-align/bin/activate && '
            'pip install -U pip && '
            'pip install --upgrade --force-reinstall -r python_service/requirements-align.txt'
        )

    @staticmethod
    def _is_alignment_runtime_missing(message: str) -> bool:
        lowered = message.lower()
        return (
            'alignment runtime missing' in lowered
            or 'whisperx package path not found' in lowered
            or 'no module named' in lowered
            or 'runtime/.venv-align' in lowered
            or 'missing:' in lowered
        )

    def _run_command(self, args: list[str], cwd: Path, context: str) -> None:
        try:
            subprocess.run(
                args,
                cwd=str(cwd),
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as exc:
            output = (exc.stderr or exc.stdout or '').strip()
            tail = '\n'.join(output.splitlines()[-12:]) if output else 'No additional error output.'
            raise RuntimeError(f'{context}: {tail}') from exc

    def _bootstrap_alignment_runtime(self) -> str:
        project_root = Path(__file__).resolve().parents[2]
        python_bin = self.settings.align_python_bin
        venv_root = python_bin.parent.parent

        if not python_bin.exists():
            self._run_command(
                [sys.executable, '-m', 'venv', str(venv_root)],
                cwd=project_root,
                context='Failed to create alignment runtime virtualenv',
            )

        resolved_python = python_bin
        if not resolved_python.exists():
            fallback_python = venv_root / 'bin' / 'python'
            if fallback_python.exists():
                resolved_python = fallback_python
            else:
                raise RuntimeError(
                    f'Alignment runtime binary not found at {python_bin}. '
                    f'{self._alignment_runtime_install_hint()}'
                )

        self._run_command(
            [str(resolved_python), '-m', 'pip', 'install', '-U', 'pip'],
            cwd=project_root,
            context='Failed to upgrade pip in alignment runtime',
        )
        self._run_command(
            [
                str(resolved_python),
                '-m',
                'pip',
                'install',
                '--upgrade',
                '--force-reinstall',
                '-r',
                'python_service/requirements-align.txt',
            ],
            cwd=project_root,
            context='Failed to install WhisperX alignment dependencies',
        )
        self.aligner.close()
        self.alignment_runtime_verified = False
        self.alignment_probe_error = None
        self.alignment_probe_at = None
        return 'Alignment runtime was prepared automatically in runtime/.venv-align.'

    async def _probe_alignment_runtime(self, force: bool = False) -> tuple[bool, str | None]:
        if self.alignment_runtime_verified and not force:
            return True, self.alignment_probe_error

        try:
            probe_result = await asyncio.to_thread(self.aligner.probe)
        except Exception as exc:  # noqa: BLE001
            self.alignment_runtime_verified = False
            self.alignment_probe_error = str(exc)
            self.alignment_probe_at = utc_now()
            return False, self.alignment_probe_error

        self.alignment_runtime_verified = True
        loaded_modules = probe_result.get('loadedModules') or []
        loaded_note = ', '.join(str(item) for item in loaded_modules) if loaded_modules else 'worker ready'
        self.alignment_probe_error = None
        self.alignment_probe_at = utc_now()
        return True, f'Alignment runtime ready ({loaded_note}).'

    async def _ensure_alignment_runtime_available(self) -> None:
        ok, reason = await self._probe_alignment_runtime(force=False)
        if ok:
            return
        raise RuntimeError(f'Alignment runtime probe failed: {reason}')

    def _whisperx_ready_marker(self) -> Path:
        return self.settings.alignment_model_dir / '.ready.json'

    async def _ensure_alignment_models(self) -> str | None:
        runtime_note: str | None = None
        marker_path = self._whisperx_ready_marker()
        try:
            await self._ensure_alignment_runtime_available()
        except Exception as exc:  # noqa: BLE001
            if not self._is_alignment_runtime_missing(str(exc)):
                raise
            try:
                runtime_note = await asyncio.to_thread(self._bootstrap_alignment_runtime)
                await self._ensure_alignment_runtime_available()
            except Exception as bootstrap_exc:  # noqa: BLE001
                message = str(bootstrap_exc)
                if self._alignment_runtime_install_hint() not in message:
                    message = f'{message}. {self._alignment_runtime_install_hint()}'
                raise RuntimeError(message) from bootstrap_exc
        if marker_path.exists():
            return runtime_note
        warmup_result = await asyncio.to_thread(self.aligner.warmup, self.settings.alignment_languages)
        marker_payload = {
            'languages': warmup_result.get('languages') or list(self.settings.alignment_languages),
            'modelDir': str(self.settings.alignment_model_dir),
            'repo': WHISPERX_MODEL_REPO,
        }
        marker_path.write_text(json.dumps(marker_payload, indent=2), 'utf-8')
        return runtime_note

    async def get_runtime_status(self) -> dict[str, Any]:
        runtime = detect_runtime_status()
        alignment_runtime_ready, probe_reason = await self._probe_alignment_runtime(force=False)
        alignment_models_ready = self._whisperx_ready_marker().exists()
        alignment_reason = (
            probe_reason
            if probe_reason
            else f'Alignment runtime missing: {self.settings.align_python_bin}'
        )
        alignment_probe_at = self.alignment_probe_at.isoformat() if self.alignment_probe_at else None
        warmup_at = self._warmup_at.isoformat() if self._warmup_at else None
        return {
            'mlxCompatible': runtime.compatible,
            'fallbackEnabled': self.settings.allow_macos_fallback,
            'runtimeVersion': runtime.runtimeVersion,
            'supportsQwen3Tts': runtime.supportsQwen3Tts,
            'minRequiredVersion': runtime.minRequiredVersion,
            'reason': runtime.reason,
            'alignmentRuntimeReady': alignment_runtime_ready,
            'alignmentModelsReady': alignment_models_ready,
            'alignmentReason': alignment_reason,
            'alignmentProbeAt': alignment_probe_at,
            'alignmentProbeError': self.alignment_probe_error,
            'warmupState': self._warmup_state,
            'warmupError': self._warmup_error,
            'warmupAt': warmup_at,
        }


    async def confirm_job_language(self, job_id: str, language: str) -> None:
        selected = (language or '').strip().lower()
        if not selected:
            raise ValueError('Language cannot be empty.')

        waiter = self.language_waiters.get(job_id)
        if waiter is None or waiter.done():
            raise KeyError('Job is not waiting for language confirmation.')

        waiter.set_result(selected)

    def get_asset_path(self, asset_id: str) -> str | None:
        return self.store.get_asset_path(asset_id)

    async def subscribe(self, job_id: str) -> asyncio.Queue[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        if job_id not in self.subscribers:
            self.subscribers[job_id] = set()
        self.subscribers[job_id].add(queue)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue[dict[str, Any]]) -> None:
        if job_id in self.subscribers:
            self.subscribers[job_id].discard(queue)
            if not self.subscribers[job_id]:
                self.subscribers.pop(job_id, None)

    async def publish(self, job_id: str, payload: dict[str, Any]) -> None:
        queues = self.subscribers.get(job_id, set())
        for queue in queues:
            await queue.put(payload)

    async def _worker_loop(self) -> None:
        while True:
            job_id = await self._get_next_pending_job()
            try:
                self.running_job_id = job_id
                self.running_job_task = asyncio.create_task(self._run_job(job_id), name=f'run-job-{job_id}')
                await self.running_job_task
            except JobCanceledError as exc:
                await self._finalize_job_canceled(job_id, str(exc) or 'Canceled by user.')
            except Exception as exc:  # noqa: BLE001
                await self._set_job_error(job_id, str(exc))
            finally:
                self.running_job_id = None
                self.running_job_task = None

    async def _set_job_error(self, job_id: str, message: str) -> None:
        task = self.mp4_tasks.pop(job_id, None)
        if task and not task.done():
            task.cancel()
        self._job_performance_profiles.pop(job_id, None)
        self.cancel_requests.pop(job_id, None)
        async with self.lock:
            job = self.store.get_job(job_id)
            if not job:
                return
            if job.state == 'canceled':
                return
            next_mp4_state = job.mp4State
            next_mp4_error = job.mp4Error
            next_mp4_progress = job.mp4Progress
            if job.mp4State in {'queued', 'running'}:
                next_mp4_state = 'failed'
                next_mp4_error = message
                next_mp4_progress = None
            failed = job.model_copy(
                update={
                    'state': 'failed',
                    'error': message,
                    'updatedAt': utc_now(),
                    'progress': job.progress,
                    'statusMessage': message,
                    'mp4State': next_mp4_state,
                    'mp4Error': next_mp4_error,
                    'mp4Progress': next_mp4_progress,
                    'alignmentState': 'failed' if job.alignmentState in {'queued', 'running'} else job.alignmentState,
                    'alignmentError': message,
                    'phase': 'failed',
                    'phaseProgress': job.phaseProgress,
                }
            )
            self.store.upsert_job(failed)
        await self.publish(job_id, {'type': 'error', 'message': message})

    async def _update_job(self, job_id: str, **changes: Any) -> JobStatus:
        async with self.lock:
            job = self.store.get_job(job_id)
            if not job:
                raise KeyError(job_id)
            next_job = job.model_copy(update={'updatedAt': utc_now(), **changes})
            self.store.upsert_job(next_job)
        return next_job

    async def _append_artifact(self, job_id: str, artifact: Artifact) -> None:
        async with self.lock:
            job = self.store.get_job(job_id)
            if not job:
                raise KeyError(job_id)
            next_job = job.model_copy(update={'updatedAt': utc_now(), 'artifacts': [*job.artifacts, artifact]})
            self.store.upsert_job(next_job)

    async def _append_word(self, job_id: str, word: WordTiming) -> None:
        async with self.lock:
            job = self.store.get_job(job_id)
            if not job:
                raise KeyError(job_id)
            words = [*job.words, word]
            next_job = job.model_copy(update={'updatedAt': utc_now(), 'words': words})
            self.store.upsert_job(next_job)

    async def _wait_for_language_confirmation(self, job_id: str, detection: DetectionResult) -> str:
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        self.language_waiters[job_id] = waiter

        raw_candidates = detection.candidates if detection.candidates else [('de', 0.0), ('en', 0.0)]
        candidates_payload = [{'code': code, 'confidence': confidence} for code, confidence in raw_candidates]
        suggested_language = detection.code if detection.code != 'auto' else (detection.candidates[0][0] if detection.candidates else None)
        if suggested_language is None:
            suggested_language = raw_candidates[0][0]

        await self._update_job(
            job_id,
            state='waiting_language',
            progress=0.05,
            detectedLanguage=suggested_language or 'auto',
            statusMessage='waiting_language_confirmation',
            phase='queued',
            phaseProgress=0.05,
        )
        await self.publish(
            job_id,
            {
                'type': 'language_required',
                'candidates': candidates_payload,
                'suggestedLanguage': suggested_language,
                'confidence': detection.confidence,
                'reason': detection.reason,
            },
        )

        try:
            selected = await asyncio.wait_for(waiter, timeout=600.0)
        except asyncio.CancelledError as exc:
            reason = self.cancel_requests.get(job_id, 'Canceled by user.')
            raise JobCanceledError(reason) from exc
        except asyncio.TimeoutError as exc:
            raise RuntimeError('Language confirmation timed out. Please start the job again.') from exc
        finally:
            self.language_waiters.pop(job_id, None)

        await self._update_job(
            job_id,
            state='running',
            detectedLanguage=selected,
            progress=0.08,
            statusMessage='language_confirmed',
            phase='model_check',
            phaseProgress=0.0,
        )
        return selected

    async def _run_job(self, job_id: str) -> None:
        payload = self.job_inputs.get(job_id)
        if not payload:
            raise RuntimeError('Job payload missing')

        job = self.store.get_job(job_id)
        if not job:
            raise RuntimeError('Job not found')
        if job.state == 'canceled':
            raise JobCanceledError(job.cancelReason or 'Canceled by user.')
        if job_id in self.cancel_requests:
            raise JobCanceledError(self.cancel_requests[job_id])

        progress_value = 0.02
        progress_lock = asyncio.Lock()
        phase_value = 'queued'
        phase_progress_value = 0.0

        async def ensure_not_canceled() -> None:
            reason = self.cancel_requests.get(job_id)
            if reason:
                raise JobCanceledError(reason)

        async def publish_progress(
            target: float,
            message: str,
            *,
            phase: str | None = None,
            phase_progress: float | None = None,
        ) -> None:
            nonlocal progress_value, phase_value, phase_progress_value
            async with progress_lock:
                progress_value = max(progress_value, target)
                if phase is not None:
                    phase_value = phase
                if phase_progress is not None:
                    phase_progress_value = min(1.0, max(0.0, phase_progress))
                effective = progress_value
            current_job = self.store.get_job(job_id)
            next_state = 'running'
            if current_job and current_job.state in {'waiting_language', 'canceling', 'canceled', 'done'}:
                next_state = current_job.state
            await self._update_job(
                job_id,
                state=next_state,
                progress=effective,
                statusMessage=message,
                phase=phase_value,
                phaseProgress=phase_progress_value,
            )
            await self.publish(
                job_id,
                {
                    'type': 'progress',
                    'progress': effective,
                    'message': message,
                    'phase': phase_value,
                    'phaseProgress': phase_progress_value,
                },
            )

        await ensure_not_canceled()
        await self._update_job(
            job_id,
            state='running',
            progress=0.02,
            statusMessage='queued',
            cancelReason=None,
            alignmentState='queued',
            alignmentError=None,
            alignmentCoverage=None,
            alignmentWarning=None,
            alignmentRetryCount=0,
            phase='queued',
            phaseProgress=0.02,
        )
        await self.publish(
            job_id,
            {
                'type': 'progress',
                'progress': 0.02,
                'message': 'queued',
                'phase': 'queued',
                'phaseProgress': 0.02,
            },
        )

        output_formats: list[str]
        preferred_language: str | None
        selected_voice_id: str | None
        selected_speaker: str | None
        job_performance_profile = await self._resolve_job_performance_profile(job_id)
        source_text = ''
        text_chunks: list[TextChunk] = []
        pdf_items_stream: Any | None = None
        estimated_total_chunks = 0

        if payload['type'] == 'text':
            request = JobRequestText.model_validate(payload['request'])
            source_text = request.text
            text_chunks = [
                TextChunk(text=item, words=[])
                for item in chunk_text(source_text, max_chars=self._text_chunk_size(job_performance_profile))
            ]
            output_formats = request.outputFormats
            preferred_language = request.language
            selected_voice_id = request.voiceId
            selected_speaker = _normalize_optional_speaker(request.speaker)
            estimated_total_chunks = len(text_chunks)
        else:
            request = JobRequestPdf.model_validate(payload['request'])
            page_range = None
            if request.pageRange:
                page_range = (request.pageRange.start - 1, request.pageRange.end - 1)

            prebuffer, tail_iter, detection_sample, selected_pages = self._prepare_pdf_stream(
                request.pdfPath,
                page_range=page_range,
                pages_per_chunk=self._pages_per_chunk(job_performance_profile),
            )
            pdf_items_stream = chain(prebuffer, tail_iter)
            output_formats = request.outputFormats
            preferred_language = request.language
            selected_voice_id = request.voiceId
            selected_speaker = _normalize_optional_speaker(request.speaker)
            source_text = detection_sample
            sample_subchunks = 0
            sample_pages = 0
            for item in prebuffer:
                sample_subchunks += max(
                    1,
                    len(
                        _split_pdf_chunk_for_synthesis(
                            item.chunk,
                            max_words=self.settings.pdf_subchunk_max_words,
                            max_chars=self.settings.pdf_subchunk_max_chars,
                        )
                    ),
                )
                sample_pages += max(1, item.pageEnd - item.pageStart + 1)
            if sample_pages > 0:
                chunks_per_page = sample_subchunks / sample_pages
                estimated_total_chunks = max(1, int(math.ceil(max(1, selected_pages) * chunks_per_page)))
            else:
                estimated_total_chunks = max(
                    1,
                    math.ceil(max(1, selected_pages) / max(1, self._pages_per_chunk(job_performance_profile))),
                )

            if not prebuffer:
                raise RuntimeError('PDF has no readable text.')

            if selected_pages > 100:
                large_pdf_warning = 'Large PDF detected. Processing page by page in chunked mode.'
                await self._update_job(job_id, warning=large_pdf_warning)
                await self.publish(job_id, {'type': 'warning', 'message': large_pdf_warning})

        if payload['type'] == 'text' and not text_chunks:
            raise RuntimeError('No text chunks generated for this job.')

        await ensure_not_canceled()
        await publish_progress(0.05, 'model_check', phase='model_check', phase_progress=0.0)
        selected_voice = self.store.get_voice(selected_voice_id) if selected_voice_id else None
        if preferred_language:
            detected_language = preferred_language
            await self._update_job(job_id, detectedLanguage=detected_language, statusMessage='language_selected')
        elif selected_voice and selected_voice.language:
            detected_language = selected_voice.language
            await self._update_job(job_id, detectedLanguage=detected_language, statusMessage='voice_language_selected')
        else:
            detection = detect_language(source_text)
            if detection.needs_confirmation:
                detected_language = await self._wait_for_language_confirmation(job_id, detection)
                await ensure_not_canceled()
            else:
                detected_language = detection.code
                await self._update_job(job_id, detectedLanguage=detected_language, statusMessage='language_detected')

        voice_route = _resolve_voice_route(
            requested_model_id=job.model,
            selected_voice=selected_voice,
            requested_speaker=selected_speaker,
            default_speaker=self.settings.standard_default_speaker,
        )
        runtime_model_id = voice_route.runtime_model_id
        locked_standard_speaker = voice_route.speaker
        if selected_voice and selected_voice.type in {'clone', 'design'}:
            locked_standard_speaker = None

        await ensure_not_canceled()
        await publish_progress(0.07, 'loading_model', phase='model_check', phase_progress=0.5)
        model_warning = self.engine.ensure_model(runtime_model_id, prefer_official=False)
        if model_warning:
            await self._update_job(job_id, warning=model_warning)
            await self.publish(job_id, {'type': 'warning', 'message': model_warning})

        runtime_status = detect_runtime_status()
        if not runtime_status.compatible:
            if not self.settings.allow_macos_fallback:
                raise RuntimeError(f'MLX runtime required. {runtime_status.reason}')

            fallback_warning = (
                f'{runtime_status.reason} Falling back to macOS say because fallback is enabled in settings.'
            )
            await self._update_job(job_id, warning=fallback_warning)
            await self.publish(job_id, {'type': 'warning', 'message': fallback_warning})

        await publish_progress(0.09, 'model_load_ram:loading', phase='model_load_ram', phase_progress=0.15)
        model_already_loaded = await asyncio.to_thread(self.engine.is_model_loaded_in_ram, runtime_model_id)
        if model_already_loaded:
            await publish_progress(0.10, 'model_load_ram:already_loaded', phase='model_load_ram', phase_progress=1.0)
        else:
            await asyncio.to_thread(self.engine.load_model_into_ram, runtime_model_id)
            await ensure_not_canceled()
            await publish_progress(0.10, 'model_load_ram:loading_now', phase='model_load_ram', phase_progress=1.0)

        await ensure_not_canceled()
        await publish_progress(0.11, 'loading_alignment_model', phase='model_check', phase_progress=0.9)
        await self._ensure_alignment_models()
        await ensure_not_canceled()

        chunk_audio_paths: list[Path] = []
        full_text_parts: list[str] = []
        current_offset_ms = 0
        aligned_by_chunk: dict[int, list[WordTiming]] = {}
        alignment_coverages: list[float] = []
        alignment_tasks: list[asyncio.Task[None]] = []
        alignment_lock = asyncio.Lock()
        alignment_semaphore = asyncio.Semaphore(self._alignment_parallelism(job_performance_profile))
        alignment_completed = 0
        alignment_warning_latest: str | None = None
        alignment_retry_count = 0
        voice_ref_path: Path | None = None
        voice_ref_text: str | None = None
        design_instruction: str | None = None
        design_voice_name: str | None = None
        chunk_sequence = 0

        try:
            if selected_voice:
                if selected_voice.type == 'clone':
                    if job.model != 'base':
                        await self._update_job(
                            job_id,
                            warning='Voice clone uses the Base model (ICL) on this MLX runtime.',
                        )
                    voice_ref_path = self.store.materialize_voice_reference(selected_voice_id)
                    stored_reference_text = self.store.get_voice_reference_text(selected_voice_id)
                    voice_ref_text, transcribed_language, reference_warning, persistable_reference = await asyncio.to_thread(
                        self._resolve_clone_reference_text,
                        voice_ref_path,
                        stored_reference_text,
                        source_text[:300],
                        detected_language,
                    )
                    if not selected_voice.language and transcribed_language:
                        detected_language = transcribed_language
                        await self._update_job(
                            job_id,
                            detectedLanguage=detected_language,
                            statusMessage='voice_language_from_reference',
                        )
                    if persistable_reference:
                        selected_voice = self.store.update_clone_voice_metadata(
                            voice_id=selected_voice_id,
                            reference_text=voice_ref_text,
                            language=transcribed_language if not selected_voice.language else None,
                        )
                    if reference_warning:
                        await self._update_job(
                            job_id,
                            warning=reference_warning,
                        )
                        await self.publish(job_id, {'type': 'warning', 'message': reference_warning})
                elif selected_voice.type == 'design':
                    design_instruction = (selected_voice.description or '').strip() or None
                    design_voice_name = selected_voice.name
                    if not design_instruction:
                        await self._update_job(
                            job_id,
                            warning='Voice design preset has empty description. Add a description for stronger style control.',
                        )
            elif locked_standard_speaker:
                design_voice_name = locked_standard_speaker

            effective_voice_id = selected_voice.id if selected_voice else selected_voice_id
            chunk_generation_profile = _build_chunk_generation_profile(
                job_id=job_id,
                voice_id=effective_voice_id,
                speaker=locked_standard_speaker,
                model_id=runtime_model_id,
                language=detected_language,
                chunk_count=max(1, estimated_total_chunks),
                consistency_mode=self.settings.tts_voice_consistency_mode,
                seed_base=self.settings.tts_chunk_seed_base,
                strict_temperature=self.settings.tts_temperature_strict,
                strict_top_k=self.settings.tts_top_k_strict,
                strict_top_p=self.settings.tts_top_p_strict,
                strict_repetition_penalty=self.settings.tts_repetition_penalty_strict,
            )
            consistency_message = (
                f'voice_consistency:{chunk_generation_profile.mode} (seed={chunk_generation_profile.random_seed})'
                if chunk_generation_profile.random_seed is not None
                else f'voice_consistency:{chunk_generation_profile.mode}'
            )
            voice_lock_message = (
                f'voice_lock:{runtime_model_id}/{locked_standard_speaker}'
                if locked_standard_speaker
                else f'voice_lock:{runtime_model_id}/{selected_voice.id if selected_voice else "none"}'
            )
            await self._update_job(job_id, statusMessage=f'{consistency_message}; {voice_lock_message}')
            if not runtime_status.compatible and chunk_generation_profile.mode != 'natural':
                fallback_consistency_warning = (
                    'voice_consistency_limited_on_fallback: macOS fallback cannot fully enforce strict chunk '
                    'voice consistency.'
                )
                await self._update_job(job_id, warning=fallback_consistency_warning)
                await self.publish(job_id, {'type': 'warning', 'message': fallback_consistency_warning})

            async def synthesize_text_segment(
                text: str,
                *,
                chunk_number: int,
                segment_number: int,
                segment_count: int,
                is_pdf_chunk: bool,
            ) -> tuple[Path, int]:
                total_hint = max(estimated_total_chunks, chunk_number)
                synth_message = f'synth chunk {chunk_number}/{total_hint}'
                chunk_base_ratio = (chunk_number - 1) / total_hint
                chunk_span_ratio = 1.0 / max(1, total_hint)
                segment_base_ratio = (segment_number - 1) / max(1, segment_count)
                segment_span_ratio = 1.0 / max(1, segment_count)

                await ensure_not_canceled()
                synth_task = asyncio.create_task(
                    asyncio.to_thread(
                        self.engine.synthesize_chunk,
                        text=text,
                        model_id=runtime_model_id,
                        language=detected_language,
                        chunk_index=((chunk_number - 1) * 1000) + (segment_number - 1),
                        max_tokens=(
                            self.settings.tts_max_tokens_pdf
                            if is_pdf_chunk
                            else self.settings.tts_max_tokens
                        ),
                        temperature=chunk_generation_profile.temperature,
                        top_k=chunk_generation_profile.top_k,
                        top_p=chunk_generation_profile.top_p,
                        repetition_penalty=chunk_generation_profile.repetition_penalty,
                        random_seed=chunk_generation_profile.random_seed,
                        voice_reference_path=voice_ref_path,
                        voice_reference_text=voice_ref_text,
                        voice_design_instruction=design_instruction,
                        voice_design_name=design_voice_name,
                    )
                )

                loop = asyncio.get_running_loop()
                started = loop.time()
                expected_seconds = max(1.1, len(re.findall(r'\S+', text)) * 0.12)
                while not synth_task.done():
                    elapsed = loop.time() - started
                    local_progress = min(0.94, elapsed / expected_seconds)
                    local_chunk_ratio = segment_base_ratio + (segment_span_ratio * local_progress)
                    synth_ratio = chunk_base_ratio + (chunk_span_ratio * local_chunk_ratio)
                    await publish_progress(
                        0.14 + 0.34 * synth_ratio,
                        synth_message,
                        phase='synthesizing',
                        phase_progress=synth_ratio,
                    )
                    await asyncio.sleep(0.18)

                result = await synth_task
                await ensure_not_canceled()
                finished_chunk_ratio = segment_base_ratio + segment_span_ratio
                finished_synth_ratio = chunk_base_ratio + (chunk_span_ratio * finished_chunk_ratio)
                await publish_progress(
                    0.14 + 0.34 * finished_synth_ratio,
                    synth_message,
                    phase='synthesizing',
                    phase_progress=finished_synth_ratio,
                )

                trim_before, trim_after = await asyncio.to_thread(
                    trim_chunk_silence_inplace,
                    result.wav_path,
                    self.settings.runtime_tmp_dir,
                    0.20 if is_pdf_chunk else 0.35,
                    220 if is_pdf_chunk else 1400,
                )
                trimmed_ms = trim_before - trim_after
                expected_trim_warn_ms = 220 if is_pdf_chunk else 350
                if trimmed_ms > expected_trim_warn_ms:
                    trim_warning = (
                        f'Silence trim exceeded expected bound at chunk {chunk_number}/{total_hint}: '
                        f'{trim_before}ms -> {trim_after}ms (trimmed {trimmed_ms}ms).'
                    )
                    await self._update_job(job_id, warning=trim_warning)
                    await self.publish(job_id, {'type': 'warning', 'message': trim_warning})

                if result.warning:
                    await self._update_job(job_id, warning=result.warning)
                    await self.publish(job_id, {'type': 'warning', 'message': result.warning})

                return result.wav_path, max(1, trim_after)

            async def run_alignment(
                chunk: TextChunk,
                chunk_number: int,
                chunk_wav_path: Path,
                offset_ms: int,
                chunk_duration_ms: int,
                alignment_text: str | None = None,
            ) -> None:
                nonlocal alignment_completed, alignment_warning_latest, alignment_retry_count
                total_hint = max(estimated_total_chunks, chunk_number)
                align_message = f'align chunk {chunk_number}/{total_hint}'
                async with alignment_semaphore:
                    await ensure_not_canceled()
                    await self._update_job(
                        job_id,
                        alignmentState='running',
                        alignmentError=None,
                        statusMessage=align_message,
                        phase='aligning',
                    )
                    await self.publish(
                        job_id,
                        {
                            'type': 'alignment_progress',
                            'progress': alignment_completed / max(1, total_hint),
                            'message': align_message,
                        },
                    )

                    source_for_alignment = (alignment_text or chunk.text).strip() or chunk.text
                    retry_candidates = _build_alignment_retry_candidates(
                        source_for_alignment,
                        chunk.text.strip() if alignment_text else None,
                    )

                    alignment_result: AlignmentResult | None = None
                    timed_tokens: list[TimedToken] = []
                    chosen_alignment_text = source_for_alignment
                    last_alignment_exception: Exception | None = None

                    for attempt_index, (attempt_label, attempt_text) in enumerate(retry_candidates):
                        try:
                            alignment_result = await asyncio.to_thread(
                                self.aligner.align_audio,
                                chunk_wav_path,
                                attempt_text,
                                detected_language,
                            )
                            timed_tokens = alignment_result.tokens
                            chosen_alignment_text = attempt_text
                            if attempt_index > 0:
                                retry_success_warning = (
                                    f'alignment_retry_success: recovered at chunk {chunk_number}/{total_hint} '
                                    f'using {attempt_label}.'
                                )
                                alignment_warning_latest = retry_success_warning
                                await self._update_job(
                                    job_id,
                                    alignmentWarning=retry_success_warning,
                                    statusMessage='alignment_retry_success',
                                )
                                await self.publish(job_id, {'type': 'warning', 'message': retry_success_warning})
                            break
                        except Exception as exc:  # noqa: BLE001
                            last_alignment_exception = exc
                            has_more = attempt_index + 1 < len(retry_candidates)
                            no_words_error = 'no aligned words' in str(exc).lower()
                            if no_words_error and has_more:
                                continue
                            break

                    if alignment_result is None:
                        fallback_warning = (
                            f'alignment_retry_failed_fallback: Alignment fallback at chunk '
                            f'{chunk_number}/{total_hint}: {last_alignment_exception}'
                        )
                        alignment_retry_count += 1
                        alignment_warning_latest = fallback_warning
                        await self._update_job(
                            job_id,
                            alignmentState='running',
                            alignmentError=None,
                            alignmentWarning=fallback_warning,
                            statusMessage='alignment_fallback',
                        )
                        await self.publish(job_id, {'type': 'warning', 'message': fallback_warning})
                        timed_tokens = self._estimate_timed_tokens(chunk.text, chunk_duration_ms)
                    else:
                        retry_needed = _alignment_needs_retry(
                            coverage=alignment_result.coverage,
                            chunk_duration_ms=chunk_duration_ms,
                            tokens=timed_tokens,
                            coverage_threshold=self.settings.alignment_retry_threshold,
                            tail_gap_threshold_ms=self.settings.alignment_retry_tail_gap_ms,
                        )
                        if retry_needed:
                            best_result = alignment_result
                            best_tokens = list(timed_tokens)
                            best_tail_gap_ms = _alignment_tail_gap_ms(best_tokens, chunk_duration_ms)
                            improved = False

                            for attempt_label, attempt_text in retry_candidates:
                                if attempt_text == chosen_alignment_text:
                                    continue
                                try:
                                    attempt_result = await asyncio.to_thread(
                                        self.aligner.align_audio,
                                        chunk_wav_path,
                                        attempt_text,
                                        detected_language,
                                    )
                                except Exception:
                                    continue

                                attempt_tokens = attempt_result.tokens
                                attempt_tail_gap_ms = _alignment_tail_gap_ms(attempt_tokens, chunk_duration_ms)
                                better_coverage = attempt_result.coverage > (best_result.coverage + 0.015)
                                better_tail = attempt_tail_gap_ms + 180 < best_tail_gap_ms
                                if not (better_coverage or better_tail):
                                    continue

                                best_result = attempt_result
                                best_tokens = list(attempt_tokens)
                                chosen_alignment_text = attempt_text
                                best_tail_gap_ms = attempt_tail_gap_ms
                                improved = True

                                retry_success_warning = (
                                    f'alignment_retry_success: recovered at chunk {chunk_number}/{total_hint} '
                                    f'using {attempt_label}.'
                                )
                                alignment_warning_latest = retry_success_warning
                                await self._update_job(
                                    job_id,
                                    alignmentWarning=retry_success_warning,
                                    statusMessage='alignment_retry_success',
                                )
                                await self.publish(job_id, {'type': 'warning', 'message': retry_success_warning})

                            if improved:
                                alignment_retry_count += 1
                                alignment_result = best_result
                                timed_tokens = best_tokens
                            else:
                                retry_failed_warning = (
                                    'alignment_retry_failed_fallback: '
                                    f'Low coverage or long tail gap at chunk {chunk_number}/{total_hint}; '
                                    'keeping best-effort markers.'
                                )
                                alignment_retry_count += 1
                                alignment_warning_latest = retry_failed_warning
                                await self._update_job(
                                    job_id,
                                    alignmentWarning=retry_failed_warning,
                                    statusMessage='alignment_retry_failed_fallback',
                                )
                                await self.publish(job_id, {'type': 'warning', 'message': retry_failed_warning})

                        if alignment_result.coverage < self.settings.alignment_min_coverage:
                            low_coverage_warning = (
                                f'Low alignment coverage at chunk {chunk_number}/{total_hint}: '
                                f'{alignment_result.matched_words}/{max(1, alignment_result.source_words)} '
                                f'({alignment_result.coverage:.2%})'
                            )
                            alignment_warning_latest = low_coverage_warning
                            await self._update_job(
                                job_id,
                                alignmentWarning=low_coverage_warning,
                                statusMessage='alignment_low_coverage',
                            )
                            await self.publish(job_id, {'type': 'warning', 'message': low_coverage_warning})

                        if not timed_tokens:
                            timed_tokens = self._estimate_timed_tokens(chosen_alignment_text, chunk_duration_ms)

                    aligned_words = self._align_words(timed_tokens, chunk.words, offset_ms=offset_ms)
                    if not aligned_words:
                        fallback_tokens = self._estimate_timed_tokens(chunk.text, chunk_duration_ms)
                        aligned_words = self._align_words(fallback_tokens, chunk.words, offset_ms=offset_ms)
                    if not aligned_words:
                        aligned_words = [
                            WordTiming(
                                word='...',
                                startMs=offset_ms,
                                endMs=max(offset_ms + 1, offset_ms + chunk_duration_ms),
                                confidence=0.0,
                                page=None,
                                bbox=None,
                            )
                        ]

                    coverage = alignment_result.coverage if alignment_result else 0.0
                    async with alignment_lock:
                        aligned_by_chunk[chunk_number] = aligned_words
                        alignment_coverages.append(coverage)
                        alignment_completed += 1
                        coverage_mean = sum(alignment_coverages) / len(alignment_coverages)

                    for word in aligned_words:
                        await self._append_word(job_id, word)
                        await self.publish(job_id, {'type': 'word', 'payload': word.model_dump(mode='json')})

                    align_ratio = alignment_completed / max(1, total_hint)
                    align_progress = 0.56 + 0.20 * align_ratio
                    await publish_progress(
                        align_progress,
                        f'align chunk done {chunk_number}/{total_hint}',
                        phase='aligning',
                        phase_progress=align_ratio,
                    )
                    await self._update_job(
                        job_id,
                        alignmentState='running',
                        alignmentCoverage=coverage_mean,
                        alignmentError=None,
                        alignmentWarning=alignment_warning_latest,
                    )
                    await self.publish(
                        job_id,
                        {
                            'type': 'alignment_done',
                            'coverage': coverage,
                        },
                    )

            async def synthesize_and_queue_alignment(chunk: TextChunk, chunk_number: int) -> None:
                nonlocal current_offset_ms
                await ensure_not_canceled()
                total_hint = max(estimated_total_chunks, chunk_number)
                chunk_base_ratio = (chunk_number - 1) / total_hint

                synth_message = f'synth chunk {chunk_number}/{total_hint}'
                await publish_progress(
                    0.14 + 0.34 * chunk_base_ratio,
                    synth_message,
                    phase='synthesizing',
                    phase_progress=chunk_base_ratio,
                )

                synthesis_text = chunk.text
                if payload['type'] == 'pdf':
                    mapped_pdf_text = _build_pdf_synthesis_text(chunk)
                    if mapped_pdf_text:
                        synthesis_text = mapped_pdf_text

                chunk_wav_path, chunk_duration_ms = await synthesize_text_segment(
                    synthesis_text,
                    chunk_number=chunk_number,
                    segment_number=1,
                    segment_count=1,
                    is_pdf_chunk=(payload['type'] == 'pdf'),
                )
                await ensure_not_canceled()

                chunk_audio_paths.append(chunk_wav_path)
                full_text_parts.append(chunk.text)
                chunk_start_ms = current_offset_ms
                chunk_end_ms = chunk_start_ms + max(1, chunk_duration_ms)
                current_offset_ms = chunk_end_ms

                chunk_asset_id = self.store.register_asset(chunk_wav_path)
                await self._append_artifact(
                    job_id,
                    Artifact(
                        id=chunk_asset_id,
                        type='audio_chunk',
                        path=str(chunk_wav_path),
                        sizeBytes=chunk_wav_path.stat().st_size,
                    ),
                )

                await self.publish(
                    job_id,
                    {
                        'type': 'audio_chunk',
                        'assetId': chunk_asset_id,
                        'startMs': chunk_start_ms,
                        'endMs': chunk_end_ms,
                    },
                )

                alignment_tasks.append(
                    asyncio.create_task(
                        run_alignment(
                            chunk=chunk,
                            chunk_number=chunk_number,
                            chunk_wav_path=chunk_wav_path,
                            offset_ms=chunk_start_ms,
                            chunk_duration_ms=max(1, chunk_duration_ms),
                            alignment_text=synthesis_text,
                        ),
                        name=f'align-{job_id}-{chunk_number}',
                    )
                )

            if payload['type'] == 'text':
                for chunk in text_chunks:
                    await ensure_not_canceled()
                    chunk_sequence += 1
                    await synthesize_and_queue_alignment(chunk, chunk_sequence)
            else:
                assert pdf_items_stream is not None
                for item in pdf_items_stream:
                    await ensure_not_canceled()
                    extract_message = f'extract_page {min(item.pageEnd + 1, item.totalPages)}/{item.totalPages}'
                    extract_progress = 0.06 + 0.05 * (min(item.pageEnd + 1, item.totalPages) / max(1, item.totalPages))
                    await publish_progress(extract_progress, extract_message, phase='model_check', phase_progress=1.0)
                    subchunks = _split_pdf_chunk_for_synthesis(
                        item.chunk,
                        max_words=self.settings.pdf_subchunk_max_words,
                        max_chars=self.settings.pdf_subchunk_max_chars,
                    )
                    for subchunk in subchunks:
                        await ensure_not_canceled()
                        chunk_sequence += 1
                        await synthesize_and_queue_alignment(subchunk, chunk_sequence)

            if alignment_tasks:
                await ensure_not_canceled()
                await publish_progress(0.74, 'waiting_alignment', phase='aligning', phase_progress=0.95)
                await asyncio.gather(*alignment_tasks)
        finally:
            if voice_ref_path is not None:
                voice_ref_path.unlink(missing_ok=True)

        await ensure_not_canceled()
        if not chunk_audio_paths:
            raise RuntimeError('No audio was generated for this job.')

        all_words: list[WordTiming] = []
        for index in sorted(aligned_by_chunk):
            all_words.extend(aligned_by_chunk[index])

        coverage_mean = sum(alignment_coverages) / max(1, len(alignment_coverages))

        final_wav = self.settings.assets_dir / f'{job_id}-final.wav'
        await ensure_not_canceled()
        await publish_progress(0.80, 'merge_audio', phase='merge_export', phase_progress=0.2)
        await asyncio.to_thread(concat_audio_chunks, chunk_audio_paths, final_wav, self.settings.runtime_tmp_dir)
        final_audio_duration_ms = await asyncio.to_thread(probe_audio_duration_ms, final_wav)

        if not all_words:
            fallback_warning = 'Alignment produced no words; using estimated timing markers.'
            alignment_warning_latest = fallback_warning
            await self._update_job(job_id, alignmentWarning=fallback_warning, warning=fallback_warning)
            await self.publish(job_id, {'type': 'warning', 'message': fallback_warning})
            fallback_tokens = self._estimate_timed_tokens('\n'.join(full_text_parts), final_audio_duration_ms)
            all_words = self._align_words(fallback_tokens, [], offset_ms=0)

        if all_words and all_words[-1].endMs > final_audio_duration_ms + 33:
            clip_warning = (
                f'Alignment timeline exceeded audio duration; clipping to {final_audio_duration_ms}ms.'
            )
            alignment_warning_latest = clip_warning
            await self._update_job(job_id, alignmentWarning=clip_warning, warning=clip_warning)
            await self.publish(job_id, {'type': 'warning', 'message': clip_warning})
            clipped_words: list[WordTiming] = []
            previous_end = 0
            for word in all_words:
                start_ms = min(max(previous_end, word.startMs), max(0, final_audio_duration_ms - 1))
                end_ms = min(max(start_ms + 1, word.endMs), final_audio_duration_ms)
                clipped_words.append(word.model_copy(update={'startMs': start_ms, 'endMs': end_ms}))
                previous_end = end_ms
            all_words = clipped_words

        # Persist alignment + transcript for diagnostics and reader synchronization.
        alignment_path = self.settings.assets_dir / f'{job_id}-alignment.json'
        transcript_path = self.settings.assets_dir / f'{job_id}-transcript.txt'
        write_alignment(all_words, alignment_path)
        write_transcript('\n\n'.join(full_text_parts), transcript_path)

        alignment_asset = self.store.register_asset(alignment_path)
        transcript_asset = self.store.register_asset(transcript_path)

        await self._append_artifact(
            job_id,
            Artifact(
                id=alignment_asset,
                type='alignment',
                path=str(alignment_path),
                sizeBytes=alignment_path.stat().st_size,
            ),
        )
        await self._append_artifact(
            job_id,
            Artifact(
                id=transcript_asset,
                type='transcript',
                path=str(transcript_path),
                sizeBytes=transcript_path.stat().st_size,
            ),
        )

        mp3_path = self.settings.assets_dir / f'{job_id}.mp3'
        if 'mp3' in output_formats or 'mp4' in output_formats:
            await ensure_not_canceled()
            await publish_progress(0.90, 'export_mp3', phase='merge_export', phase_progress=0.8)
            await asyncio.to_thread(export_mp3, final_wav, mp3_path, self.settings.mp3_bitrate)

        if 'mp3' in output_formats:
            mp3_asset = self.store.register_asset(mp3_path)
            await self._append_artifact(
                job_id,
                Artifact(id=mp3_asset, type='mp3', path=str(mp3_path), sizeBytes=mp3_path.stat().st_size),
            )

        mp4_requested = 'mp4' in output_formats
        await ensure_not_canceled()
        await self._update_job(
            job_id,
            state='done',
            progress=1.0,
            words=all_words,
            statusMessage='mp4_background_queued' if mp4_requested else 'done',
            audioDurationMs=final_audio_duration_ms,
            mp4State='queued' if mp4_requested else 'not_requested',
            mp4Error=None,
            mp4Progress=0.0 if mp4_requested else None,
            alignmentState='done',
            alignmentError=None,
            alignmentCoverage=coverage_mean,
            alignmentRetryCount=alignment_retry_count,
            phase='done',
            phaseProgress=1.0,
            alignmentWarning=alignment_warning_latest,
        )
        await self.publish(
            job_id,
            {'type': 'progress', 'progress': 1.0, 'message': 'done', 'phase': 'done', 'phaseProgress': 1.0},
        )
        await self.publish(job_id, {'type': 'done'})

        if mp4_requested:
            task = asyncio.create_task(
                self._run_mp4_background(job_id=job_id, input_audio=final_wav, words=list(all_words)),
                name=f'mp4-background-{job_id}',
            )
            self.mp4_tasks[job_id] = task

        self._job_performance_profiles.pop(job_id, None)

    def _prepare_pdf_stream(
        self,
        pdf_path: str,
        page_range: tuple[int, int] | None,
        pages_per_chunk: int,
    ) -> tuple[list[PdfChunkItem], Any, str, int]:
        items = iter_pdf_chunks(
            pdf_path=pdf_path,
            page_range=page_range,
            pages_per_chunk=pages_per_chunk,
        )

        prebuffer: list[PdfChunkItem] = []
        detection_parts: list[str] = []
        selected_pages = 0

        for item in items:
            prebuffer.append(item)
            selected_pages = item.totalPages
            if item.chunk.text.strip():
                detection_parts.append(item.chunk.text.strip())
            joined = '\n\n'.join(detection_parts)
            if len(joined) >= 900 or len(prebuffer) >= 2:
                break

        detection_sample = '\n\n'.join(detection_parts).strip()
        if not detection_sample and prebuffer:
            detection_sample = prebuffer[0].chunk.text

        return prebuffer, items, detection_sample, selected_pages

    def _estimate_timed_tokens(self, text: str, duration_ms: int) -> list[TimedToken]:
        tokens = [item.strip() for item in re.findall(r'\S+', text) if item.strip()]
        if not tokens:
            return []
        total_ms = max(1, duration_ms)
        span_ms = max(45, int(total_ms / max(1, len(tokens))))
        cursor = 0
        estimated: list[TimedToken] = []
        for index, token in enumerate(tokens):
            start_ms = min(total_ms - 1, cursor)
            if index == len(tokens) - 1:
                end_ms = total_ms
            else:
                end_ms = min(total_ms, max(start_ms + 1, start_ms + span_ms))
            estimated.append(
                TimedToken(
                    word=token,
                    start_ms=start_ms,
                    end_ms=end_ms,
                    confidence=0.0,
                )
            )
            cursor = end_ms
        return estimated

    def _align_words(self, timed_tokens: list[TimedToken], extracted_words: list[ExtractedWord], offset_ms: int) -> list[WordTiming]:
        aligned: list[WordTiming] = []
        cursor = 0

        for timed in timed_tokens:
            page = None
            bbox = None

            if extracted_words:
                token = _clean_token(timed.word)
                found_idx = None
                search_limit = min(len(extracted_words), cursor + 64)
                for idx in range(cursor, search_limit):
                    if _clean_token(extracted_words[idx].word) == token:
                        found_idx = idx
                        break

                if found_idx is None and cursor < len(extracted_words):
                    found_idx = cursor

                if found_idx is not None:
                    match = extracted_words[found_idx]
                    page = match.page
                    bbox = match.bbox
                    cursor = found_idx + 1

            aligned.append(
                WordTiming(
                    word=timed.word,
                    startMs=offset_ms + timed.start_ms,
                    endMs=offset_ms + timed.end_ms,
                    confidence=timed.confidence,
                    page=page,
                    bbox=bbox,
                )
            )

        return aligned

    async def _run_mp4_background(self, job_id: str, input_audio: Path, words: list[WordTiming]) -> None:
        try:
            await self._update_job(
                job_id,
                mp4State='running',
                mp4Error=None,
                mp4Progress=0.0,
                statusMessage='mp4_background_running',
                phase='done',
                phaseProgress=1.0,
            )
            await self.publish(
                job_id,
                {'type': 'progress', 'progress': 1.0, 'message': 'mp4_background_running', 'phase': 'done', 'phaseProgress': 1.0},
            )

            mp4_path = self.settings.assets_dir / f'{job_id}.mp4'
            progress_queue: queue.SimpleQueue[tuple[float, str]] = queue.SimpleQueue()

            def progress_callback(progress: float, message: str) -> None:
                progress_queue.put((progress, message))

            export_task = asyncio.create_task(
                asyncio.to_thread(
                    export_karaoke_mp4,
                    self.settings,
                    input_audio,
                    words,
                    mp4_path,
                    progress_callback,
                )
            )

            last_progress = 0.0
            last_message = 'mp4_background_running'
            loop = asyncio.get_running_loop()
            last_progress_at = loop.time()
            while not export_task.done():
                drained = False
                while True:
                    try:
                        next_progress, next_message = progress_queue.get_nowait()
                    except queue.Empty:
                        break
                    drained = True
                    bounded_progress = min(1.0, max(last_progress, float(next_progress)))
                    message = next_message or last_message
                    if abs(bounded_progress - last_progress) < 0.01 and message == last_message:
                        continue
                    last_progress = bounded_progress
                    last_message = message
                    last_progress_at = loop.time()
                    await self._update_job(
                        job_id,
                        mp4State='running',
                        mp4Error=None,
                        mp4Progress=last_progress,
                        statusMessage=message,
                        phase='done',
                        phaseProgress=1.0,
                    )
                    await self.publish(
                        job_id,
                        {
                            'type': 'mp4_background_progress',
                            'progress': last_progress,
                            'message': message,
                        },
                    )
                if not drained:
                    if loop.time() - last_progress_at > 60.0:
                        export_task.cancel()
                        with suppress(asyncio.CancelledError):
                            await export_task
                        raise RuntimeError('MP4 background export stalled for more than 60s without progress.')
                    await asyncio.sleep(0.12)

            export_warning = await export_task
            while True:
                try:
                    next_progress, next_message = progress_queue.get_nowait()
                except queue.Empty:
                    break
                bounded_progress = min(1.0, max(last_progress, float(next_progress)))
                message = next_message or last_message
                last_progress = bounded_progress
                last_message = message

            mp4_asset = self.store.register_asset(mp4_path)
            await self._append_artifact(
                job_id,
                Artifact(id=mp4_asset, type='mp4', path=str(mp4_path), sizeBytes=mp4_path.stat().st_size),
            )
            if export_warning:
                await self._update_job(job_id, warning=export_warning)
                await self.publish(job_id, {'type': 'warning', 'message': export_warning})
            await self._update_job(
                job_id,
                mp4State='done',
                mp4Error=None,
                mp4Progress=1.0,
                statusMessage='done',
                phase='done',
                phaseProgress=1.0,
            )
            await self.publish(
                job_id,
                {'type': 'progress', 'progress': 1.0, 'message': 'mp4_background_done', 'phase': 'done', 'phaseProgress': 1.0},
            )
            await self.publish(
                job_id,
                {'type': 'mp4_background_progress', 'progress': 1.0, 'message': 'mp4_background_done'},
            )
        except asyncio.CancelledError:
            raise
        except KeyError:
            return
        except Exception as exc:  # noqa: BLE001
            message = f'MP4 background export failed: {exc}'
            try:
                await self._update_job(
                    job_id,
                    mp4State='failed',
                    mp4Error=str(exc),
                    mp4Progress=None,
                    warning=message,
                    statusMessage='mp4_background_failed',
                    phase='done',
                    phaseProgress=1.0,
                )
            except KeyError:
                return
            await self.publish(job_id, {'type': 'warning', 'message': message})
        finally:
            self.mp4_tasks.pop(job_id, None)

    def _adapt_performance_profile_if_needed(self, job_id: str) -> None:
        process = psutil.Process()
        memory_mb = process.memory_info().rss / (1024 * 1024)
        if memory_mb > 3600 and self._job_performance_profiles.get(job_id) != 'memory':
            self._job_performance_profiles[job_id] = 'memory'

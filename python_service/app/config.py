from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = PROJECT_ROOT / 'runtime'

MODEL_REPOS = {
    'base': {
        'mlx': 'mlx-community/Qwen3-TTS-12Hz-1.7B-Base-8bit',
        'official': 'Qwen/Qwen3-TTS-12Hz-1.7B-Base',
    },
    'customvoice': {
        'mlx': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit',
        'official': 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
    },
    'voicedesign': {
        'mlx': 'mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-8bit',
        'official': 'Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign',
    },
}

WHISPERX_MODEL_REPO = 'whisperx/forced-alignment'

QUALITY_PRESET_TO_STEPS = {
    'speed': 16,
    'balanced': 28,
    'quality': 40,
}

VOICE_CONSISTENCY_MODES = {'strict', 'balanced', 'natural'}
VoiceConsistencyMode = Literal['strict', 'balanced', 'natural']


@dataclass(slots=True)
class Settings:
    output_dir: Path
    model_cache_dir: Path
    temp_dir: Path
    whisper_cache_dir: Path
    whisper_model_size: str
    align_python_bin: Path
    alignment_model_dir: Path
    alignment_languages: tuple[str, ...]
    alignment_min_coverage: float
    voice_secret_b64: str
    performance_profile: str
    quality_preset: str
    allow_macos_fallback: bool
    tts_max_tokens: int
    tts_max_tokens_pdf: int
    pdf_subchunk_max_words: int
    pdf_subchunk_max_chars: int
    alignment_retry_threshold: float
    alignment_retry_tail_gap_ms: int
    tts_voice_consistency_mode: VoiceConsistencyMode
    tts_chunk_seed_base: int
    tts_temperature_strict: float
    tts_top_k_strict: int
    tts_top_p_strict: float
    tts_repetition_penalty_strict: float
    standard_preset_speakers: tuple[str, ...]
    standard_default_speaker: str
    default_sample_rate: int = 24000
    mp3_bitrate: str = '192k'
    mp4_resolution: str = '1920x1080'
    mp4_fps: int = 30

    @property
    def assets_dir(self) -> Path:
        return self.output_dir / 'assets'

    @property
    def voices_dir(self) -> Path:
        return self.output_dir / 'voices'

    @property
    def runtime_tmp_dir(self) -> Path:
        return self.temp_dir

    @property
    def jobs_index_path(self) -> Path:
        return self.output_dir / 'jobs.json'

    @property
    def voices_index_path(self) -> Path:
        return self.voices_dir / 'index.json'

    @property
    def inference_steps(self) -> int:
        return QUALITY_PRESET_TO_STEPS.get(self.quality_preset, QUALITY_PRESET_TO_STEPS['balanced'])

    @property
    def say_rate(self) -> int:
        if self.quality_preset == 'quality':
            return 165
        if self.quality_preset == 'speed':
            return 205
        return 182

    @classmethod
    def from_env(cls) -> 'Settings':
        output_dir = Path(os.getenv('TTS_OUTPUT_DIR', str(RUNTIME_ROOT / 'outputs')))
        model_cache_dir = Path(os.getenv('TTS_MODEL_CACHE_DIR', str(RUNTIME_ROOT / 'models')))
        temp_dir = Path(os.getenv('TTS_TMP_DIR', str(RUNTIME_ROOT / 'tmp')))
        whisper_cache_dir = Path(os.getenv('TTS_WHISPER_CACHE_DIR', str(model_cache_dir / 'whisper')))
        whisper_model_size = os.getenv('TTS_WHISPER_MODEL', 'tiny').strip() or 'tiny'
        align_python_default = RUNTIME_ROOT / '.venv-align' / 'bin' / 'python3'
        align_python_bin = Path(os.getenv('TTS_ALIGN_PYTHON', str(align_python_default)))
        alignment_model_dir = Path(os.getenv('TTS_ALIGN_MODEL_DIR', str(model_cache_dir / 'whisperx')))
        alignment_languages_raw = os.getenv('TTS_ALIGN_LANGUAGES', 'de,en').strip()
        alignment_languages = tuple(
            item.strip().lower() for item in alignment_languages_raw.split(',') if item.strip()
        ) or ('de', 'en')
        alignment_min_coverage_raw = os.getenv('TTS_ALIGN_MIN_COVERAGE', '0.97').strip()
        try:
            alignment_min_coverage = float(alignment_min_coverage_raw)
        except ValueError:
            alignment_min_coverage = 0.97
        alignment_min_coverage = min(1.0, max(0.01, alignment_min_coverage))
        voice_secret_b64 = os.getenv('TTS_VOICE_SECRET', '')
        performance_profile = os.getenv('TTS_PERFORMANCE_PROFILE', 'standard')
        quality_preset = os.getenv('TTS_QUALITY_PRESET', 'balanced')
        allow_macos_fallback_raw = os.getenv('TTS_ALLOW_MACOS_FALLBACK', '0').strip().lower()
        allow_macos_fallback = allow_macos_fallback_raw in {'1', 'true', 'yes', 'on'}
        tts_max_tokens_raw = os.getenv('TTS_MAX_TOKENS', '4096').strip()
        tts_max_tokens_pdf_raw = os.getenv('TTS_MAX_TOKENS_PDF', '3072').strip()
        pdf_subchunk_max_words_raw = os.getenv('TTS_PDF_SUBCHUNK_MAX_WORDS', '95').strip()
        pdf_subchunk_max_chars_raw = os.getenv('TTS_PDF_SUBCHUNK_MAX_CHARS', '700').strip()
        alignment_retry_threshold_raw = os.getenv('TTS_ALIGN_RETRY_THRESHOLD', '0.94').strip()
        alignment_retry_tail_gap_ms_raw = os.getenv('TTS_ALIGN_RETRY_TAIL_GAP_MS', '2500').strip()
        tts_voice_consistency_mode_raw = os.getenv('TTS_VOICE_CONSISTENCY_MODE', 'strict').strip().lower()
        tts_chunk_seed_base_raw = os.getenv('TTS_CHUNK_SEED_BASE', '424242').strip()
        tts_temperature_strict_raw = os.getenv('TTS_TEMPERATURE_STRICT', '0.18').strip()
        tts_top_k_strict_raw = os.getenv('TTS_TOP_K_STRICT', '20').strip()
        tts_top_p_strict_raw = os.getenv('TTS_TOP_P_STRICT', '0.85').strip()
        tts_repetition_penalty_strict_raw = os.getenv('TTS_REPETITION_PENALTY_STRICT', '1.12').strip()
        standard_preset_speakers_raw = os.getenv(
            'TTS_STANDARD_PRESET_SPEAKERS',
            'serena,vivian,ryan,aiden',
        ).strip()
        standard_default_speaker_raw = os.getenv('TTS_STANDARD_DEFAULT_SPEAKER', 'serena').strip()

        try:
            tts_max_tokens = int(tts_max_tokens_raw)
        except ValueError:
            tts_max_tokens = 4096
        tts_max_tokens = min(8192, max(256, tts_max_tokens))

        try:
            tts_max_tokens_pdf = int(tts_max_tokens_pdf_raw)
        except ValueError:
            tts_max_tokens_pdf = 3072
        tts_max_tokens_pdf = min(8192, max(256, tts_max_tokens_pdf))

        try:
            pdf_subchunk_max_words = int(pdf_subchunk_max_words_raw)
        except ValueError:
            pdf_subchunk_max_words = 95
        pdf_subchunk_max_words = min(260, max(25, pdf_subchunk_max_words))

        try:
            pdf_subchunk_max_chars = int(pdf_subchunk_max_chars_raw)
        except ValueError:
            pdf_subchunk_max_chars = 700
        pdf_subchunk_max_chars = min(2400, max(220, pdf_subchunk_max_chars))

        try:
            alignment_retry_threshold = float(alignment_retry_threshold_raw)
        except ValueError:
            alignment_retry_threshold = 0.94
        alignment_retry_threshold = min(1.0, max(0.01, alignment_retry_threshold))

        try:
            alignment_retry_tail_gap_ms = int(alignment_retry_tail_gap_ms_raw)
        except ValueError:
            alignment_retry_tail_gap_ms = 2500
        alignment_retry_tail_gap_ms = min(10000, max(200, alignment_retry_tail_gap_ms))

        tts_voice_consistency_mode: VoiceConsistencyMode
        if tts_voice_consistency_mode_raw in VOICE_CONSISTENCY_MODES:
            tts_voice_consistency_mode = tts_voice_consistency_mode_raw  # type: ignore[assignment]
        else:
            tts_voice_consistency_mode = 'strict'

        try:
            tts_chunk_seed_base = int(tts_chunk_seed_base_raw)
        except ValueError:
            tts_chunk_seed_base = 424242
        tts_chunk_seed_base = max(1, tts_chunk_seed_base)

        try:
            tts_temperature_strict = float(tts_temperature_strict_raw)
        except ValueError:
            tts_temperature_strict = 0.18
        tts_temperature_strict = min(1.0, max(0.0, tts_temperature_strict))

        try:
            tts_top_k_strict = int(tts_top_k_strict_raw)
        except ValueError:
            tts_top_k_strict = 20
        tts_top_k_strict = max(0, tts_top_k_strict)

        try:
            tts_top_p_strict = float(tts_top_p_strict_raw)
        except ValueError:
            tts_top_p_strict = 0.85
        tts_top_p_strict = min(1.0, max(0.1, tts_top_p_strict))

        try:
            tts_repetition_penalty_strict = float(tts_repetition_penalty_strict_raw)
        except ValueError:
            tts_repetition_penalty_strict = 1.12
        tts_repetition_penalty_strict = max(1.0, tts_repetition_penalty_strict)

        default_speaker = (standard_default_speaker_raw or 'serena').strip()
        configured_speakers = [
            item.strip()
            for item in standard_preset_speakers_raw.split(',')
            if item.strip()
        ]
        if not configured_speakers:
            configured_speakers = [default_speaker]
        if default_speaker.lower() not in {item.lower() for item in configured_speakers}:
            configured_speakers.insert(0, default_speaker)

        if quality_preset not in QUALITY_PRESET_TO_STEPS:
            quality_preset = 'balanced'
        settings = cls(
            output_dir=output_dir,
            model_cache_dir=model_cache_dir,
            temp_dir=temp_dir,
            whisper_cache_dir=whisper_cache_dir,
            whisper_model_size=whisper_model_size,
            align_python_bin=align_python_bin,
            alignment_model_dir=alignment_model_dir,
            alignment_languages=alignment_languages,
            alignment_min_coverage=alignment_min_coverage,
            voice_secret_b64=voice_secret_b64,
            performance_profile=performance_profile,
            quality_preset=quality_preset,
            allow_macos_fallback=allow_macos_fallback,
            tts_max_tokens=tts_max_tokens,
            tts_max_tokens_pdf=tts_max_tokens_pdf,
            pdf_subchunk_max_words=pdf_subchunk_max_words,
            pdf_subchunk_max_chars=pdf_subchunk_max_chars,
            alignment_retry_threshold=alignment_retry_threshold,
            alignment_retry_tail_gap_ms=alignment_retry_tail_gap_ms,
            tts_voice_consistency_mode=tts_voice_consistency_mode,
            tts_chunk_seed_base=tts_chunk_seed_base,
            tts_temperature_strict=tts_temperature_strict,
            tts_top_k_strict=tts_top_k_strict,
            tts_top_p_strict=tts_top_p_strict,
            tts_repetition_penalty_strict=tts_repetition_penalty_strict,
            standard_preset_speakers=tuple(configured_speakers),
            standard_default_speaker=default_speaker,
        )
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        settings.assets_dir.mkdir(parents=True, exist_ok=True)
        settings.voices_dir.mkdir(parents=True, exist_ok=True)
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        settings.whisper_cache_dir.mkdir(parents=True, exist_ok=True)
        settings.alignment_model_dir.mkdir(parents=True, exist_ok=True)
        return settings

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


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
        )
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        settings.assets_dir.mkdir(parents=True, exist_ok=True)
        settings.voices_dir.mkdir(parents=True, exist_ok=True)
        settings.model_cache_dir.mkdir(parents=True, exist_ok=True)
        settings.temp_dir.mkdir(parents=True, exist_ok=True)
        settings.whisper_cache_dir.mkdir(parents=True, exist_ok=True)
        settings.alignment_model_dir.mkdir(parents=True, exist_ok=True)
        return settings

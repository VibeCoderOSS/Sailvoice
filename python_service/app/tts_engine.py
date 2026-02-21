from __future__ import annotations

import math
import re
import subprocess
import tempfile
import wave
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from huggingface_hub import snapshot_download

from .config import MODEL_REPOS, Settings
from .mlx_adapter import (
    preload_runtime_model,
    runtime_available,
    runtime_model_loaded,
    synthesize_with_runtime,
)


@dataclass(slots=True)
class SynthesisResult:
    wav_path: Path
    duration_ms: int
    warning: str | None


def chunk_text(text: str, max_chars: int = 380) -> list[str]:
    cleaned = re.sub(r'\s+', ' ', text).strip()
    if not cleaned:
        return []

    sentences = re.split(r'(?<=[.!?])\s+', cleaned)
    chunks: list[str] = []
    buffer = ''

    for sentence in sentences:
        if len(buffer) + len(sentence) + 1 <= max_chars:
            buffer = f'{buffer} {sentence}'.strip()
        else:
            if buffer:
                chunks.append(buffer)
            if len(sentence) <= max_chars:
                buffer = sentence
            else:
                cursor = 0
                while cursor < len(sentence):
                    part = sentence[cursor : cursor + max_chars]
                    chunks.append(part)
                    cursor += max_chars
                buffer = ''

    if buffer:
        chunks.append(buffer)
    return chunks


class MlxQwenEngine:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._downloaded_models: set[str] = set()
        self._local_model_dirs: dict[str, Path] = {}
        self._runtime_available = runtime_available()

    @property
    def backend_label(self) -> str:
        if self._runtime_available:
            return 'mlx-qwen-runtime'
        return 'macos-say-fallback'

    def ensure_model(self, model_id: str, prefer_official: bool = False) -> str | None:
        mode = 'official' if prefer_official else 'mlx'
        repo = MODEL_REPOS[model_id][mode]
        if repo in self._downloaded_models:
            return None

        local_dir = self.settings.model_cache_dir / repo.replace('/', '--')
        local_dir.mkdir(parents=True, exist_ok=True)
        self._local_model_dirs[model_id] = local_dir

        if any(local_dir.iterdir()):
            self._downloaded_models.add(repo)
            return None

        try:
            snapshot_download(
                repo_id=repo,
                local_dir=local_dir,
                local_dir_use_symlinks=False,
                resume_download=True,
                max_workers=4,
            )
            self._downloaded_models.add(repo)
            return None
        except Exception as exc:  # noqa: BLE001
            return f'Model download skipped ({repo}): {exc}'

    def is_model_loaded_in_ram(self, model_id: str) -> bool:
        model_dir = self._local_model_dirs.get(model_id)
        if not self._runtime_available or model_dir is None:
            return False
        return runtime_model_loaded(model_dir)

    def load_model_into_ram(self, model_id: str) -> bool:
        model_dir = self._local_model_dirs.get(model_id)
        if not self._runtime_available or model_dir is None:
            return False
        if runtime_model_loaded(model_dir):
            return False
        preload_runtime_model(model_dir)
        return True

    def synthesize_chunk(
        self,
        text: str,
        model_id: str,
        language: str | None,
        chunk_index: int,
        voice_reference_path: Path | None = None,
        voice_reference_text: str | None = None,
        voice_design_instruction: str | None = None,
        voice_design_name: str | None = None,
    ) -> SynthesisResult:
        chunk_id = f'{chunk_index:04d}-{uuid4().hex[:8]}'
        wav_path = self.settings.assets_dir / f'chunk-{chunk_id}.wav'

        warning_parts: list[str] = []
        runtime_used = False

        if self._runtime_available and model_id in self._local_model_dirs:
            try:
                synthesize_with_runtime(
                    text=text,
                    model_dir=self._local_model_dirs[model_id],
                    output_wav=wav_path,
                    language=language,
                    steps=self.settings.inference_steps,
                    ref_audio_path=voice_reference_path,
                    ref_audio_text=voice_reference_text,
                    design_instruction=voice_design_instruction,
                    voice_name=voice_design_name,
                    temp_dir=self.settings.runtime_tmp_dir,
                )
                runtime_used = True
            except Exception as exc:  # noqa: BLE001
                if not self.settings.allow_macos_fallback:
                    raise RuntimeError(f'MLX runtime failed ({exc}). macOS fallback is disabled.') from exc
                warning_parts.append(f'MLX runtime failed ({exc}). Falling back to macOS say.')

        if not runtime_used:
            if not self.settings.allow_macos_fallback:
                raise RuntimeError('MLX runtime unavailable and macOS fallback is disabled.')
            if voice_reference_path:
                warning_parts.append('Voice clone is not available in macOS fallback mode; using default voice.')
            if voice_design_instruction:
                warning_parts.append('Voice design instruction is not available in macOS fallback mode; using default voice.')
            try:
                self._synthesize_with_macos_say(text=text, language=language, output_wav=wav_path)
            except Exception as exc:  # noqa: BLE001
                warning_parts.append(f'say fallback failed ({exc}), generated tone placeholder instead.')
                self._synthesize_tone(text=text, output_wav=wav_path)

        duration_ms = self._probe_duration_ms(wav_path)
        warning = '; '.join(warning_parts) if warning_parts else None
        return SynthesisResult(wav_path=wav_path, duration_ms=duration_ms, warning=warning)

    def _voice_for_language(self, language: str | None) -> str:
        if not language:
            return 'Eddy'
        code = language.lower()
        if code.startswith('de'):
            return 'Flo'
        if code.startswith('it'):
            return 'Alice'
        if code.startswith('fr'):
            return 'Thomas'
        if code.startswith('en'):
            return 'Eddy'
        return 'Eddy'

    def _synthesize_with_macos_say(self, text: str, language: str | None, output_wav: Path) -> None:
        voice = self._voice_for_language(language)
        with tempfile.TemporaryDirectory(prefix='qwen3-tts-', dir=self.settings.runtime_tmp_dir) as tmp_dir:
            tmp_aiff = Path(tmp_dir) / 'chunk.aiff'
            subprocess.run(['say', '-v', voice, '-r', str(self.settings.say_rate), '-o', str(tmp_aiff), text], check=True)
            subprocess.run(
                [
                    'ffmpeg',
                    '-y',
                    '-hide_banner',
                    '-loglevel',
                    'error',
                    '-i',
                    str(tmp_aiff),
                    '-ar',
                    str(self.settings.default_sample_rate),
                    '-ac',
                    '1',
                    str(output_wav),
                ],
                check=True,
            )

    def _synthesize_tone(self, text: str, output_wav: Path) -> None:
        word_count = max(1, len(re.findall(r"\S+", text)))
        duration_sec = max(0.7, word_count * 0.28)
        sample_rate = self.settings.default_sample_rate
        total_samples = int(sample_rate * duration_sec)
        frequency = 220.0
        amplitude = 12000

        with wave.open(str(output_wav), 'w') as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)

            frames = bytearray()
            for i in range(total_samples):
                value = int(amplitude * math.sin(2 * math.pi * frequency * i / sample_rate))
                frames.extend(value.to_bytes(2, byteorder='little', signed=True))
            wav_file.writeframes(frames)

    def _probe_duration_ms(self, audio_path: Path) -> int:
        probe = subprocess.run(
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
        )
        duration_sec = float((probe.stdout or '0').strip() or 0.0)
        return max(1, int(round(duration_sec * 1000)))

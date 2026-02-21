from __future__ import annotations

import importlib
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

RUNTIME_MODULES = (
    'mlx_audio',
    'qwen_tts_mlx',
    'qwen3_tts_mlx',
    'qwen_tts',
)

_MLX_AUDIO_MODEL_CACHE: dict[str, Any] = {}


def _find_runtime_module():
    for name in RUNTIME_MODULES:
        try:
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def runtime_available() -> bool:
    return _find_runtime_module() is not None


def _extract_path_from_result(result: Any) -> Path | None:
    if isinstance(result, (str, Path)):
        return Path(result)
    if isinstance(result, (list, tuple)):
        for item in result:
            path = _extract_path_from_result(item)
            if path is not None:
                return path
    if isinstance(result, dict):
        for key in ('path', 'audio_path', 'output_path', 'file'):
            if key in result and isinstance(result[key], (str, Path)):
                return Path(result[key])
    return None


def _call_with_step_variants(func: Any, kwargs: dict[str, Any], steps: int | None):
    if steps is None:
        return func(**kwargs)

    for key in ('steps', 'num_steps', 'inference_steps', 'n_steps', 'diffusion_steps'):
        try:
            return func(**{**kwargs, key: steps})
        except TypeError:
            continue
    return func(**kwargs)


def _call_with_optional_voice_clone(
    func: Any,
    kwargs: dict[str, Any],
    steps: int | None,
    ref_audio_path: Path | None,
) -> Any:
    if ref_audio_path is None:
        return _call_with_step_variants(func, kwargs, steps)

    clone_keys = ('ref_audio', 'reference_audio', 'voice_audio', 'speaker_wav')

    for key in clone_keys:
        try:
            return _call_with_step_variants(func, {**kwargs, key: str(ref_audio_path)}, steps)
        except TypeError:
            continue

    raise RuntimeError(
        f'Runtime does not accept voice-clone input arguments ({clone_keys}). '
        'Use a qwen3_tts-capable runtime.'
    )


def _normalize_output_to_wav(path_like: str | Path, output_wav: Path) -> None:
    source = Path(path_like)
    if not source.exists():
        raise RuntimeError(f'Runtime produced missing file: {source}')

    if source.suffix.lower() == '.wav':
        shutil.copy2(source, output_wav)
        return

    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            str(source),
            str(output_wav),
        ],
        check=True,
    )


def _get_mlx_audio_model(model_dir: Path) -> Any:
    key = str(model_dir.resolve())
    if key in _MLX_AUDIO_MODEL_CACHE:
        return _MLX_AUDIO_MODEL_CACHE[key]

    utils = importlib.import_module('mlx_audio.tts.utils')
    load_model = getattr(utils, 'load_model', None)
    if not callable(load_model):
        raise RuntimeError('mlx_audio.tts.utils.load_model not found.')

    model = load_model(key)
    _MLX_AUDIO_MODEL_CACHE[key] = model
    return model


def runtime_model_loaded(model_dir: Path) -> bool:
    key = str(model_dir.resolve())
    return key in _MLX_AUDIO_MODEL_CACHE


def preload_runtime_model(model_dir: Path) -> None:
    _get_mlx_audio_model(model_dir)


def _resolve_custom_voice_speaker(
    supported_speakers: list[str],
    requested_voice: str | None,
) -> str:
    if not supported_speakers:
        raise RuntimeError('CustomVoice model reports no supported speakers.')

    if requested_voice:
        requested_norm = requested_voice.strip().lower()
        for speaker in supported_speakers:
            if speaker.lower() == requested_norm:
                return speaker

    preferred = (
        'serena',
        'vivian',
        'aiden',
        'ryan',
        'eric',
        'dylan',
        'ono_anna',
        'sohee',
        'uncle_fu',
    )
    for candidate in preferred:
        for speaker in supported_speakers:
            if speaker.lower() == candidate:
                return speaker

    return supported_speakers[0]


def _synthesize_with_mlx_audio(
    text: str,
    model_dir: Path,
    output_wav: Path,
    language: str | None,
    steps: int | None,
    ref_audio_path: Path | None,
    ref_audio_text: str | None,
    design_instruction: str | None,
    voice_name: str | None,
    temp_dir: Path | None,
) -> None:
    generate_mod = importlib.import_module('mlx_audio.tts.generate')
    generate_audio = getattr(generate_mod, 'generate_audio', None)
    if not callable(generate_audio):
        raise RuntimeError('mlx_audio.tts.generate.generate_audio not found.')

    model = _get_mlx_audio_model(model_dir)
    tts_model_type = str(getattr(getattr(model, 'config', None), 'tts_model_type', '')).strip().lower()
    supported_speakers = [str(item) for item in (getattr(model, 'supported_speakers', None) or []) if str(item).strip()]

    with tempfile.TemporaryDirectory(prefix='mlx-audio-gen-', dir=temp_dir) as tmp_dir:
        prefix = Path(tmp_dir) / 'out'
        kwargs: dict[str, Any] = {
            'model': model,
            'text': text,
            'file_prefix': str(prefix),
        }
        if language:
            # mlx-audio expects lang_code on several runtimes; language is kept as fallback.
            kwargs['lang_code'] = language
        if supported_speakers and not ref_audio_path and not design_instruction:
            kwargs['voice'] = _resolve_custom_voice_speaker(supported_speakers, voice_name)
        elif voice_name and tts_model_type != 'voice_design':
            kwargs['voice'] = voice_name
        if design_instruction:
            kwargs['instruct'] = design_instruction

        if ref_audio_path:
            if tts_model_type == 'custom_voice' or supported_speakers:
                raise RuntimeError(
                    'Qwen3 CustomVoice model does not use reference-audio cloning in mlx-audio. '
                    'Use the Base model for local voice cloning.'
                )
            # Keep cloning fully local: provide ref_text + disable STT auto-transcription downloads.
            kwargs['ref_audio'] = str(ref_audio_path)
            kwargs['ref_text'] = (ref_audio_text or '').strip() or 'reference voice sample'
            kwargs['stt_model'] = None

        try:
            result = _call_with_step_variants(generate_audio, kwargs, steps)
        except TypeError:
            kwargs.pop('lang_code', None)
            if language:
                kwargs['language'] = language
            result = _call_with_step_variants(generate_audio, kwargs, steps)

        candidate = _extract_path_from_result(result)
        if candidate is None:
            generated = sorted(Path(tmp_dir).glob('*.wav')) + sorted(Path(tmp_dir).glob('*.mp3')) + sorted(Path(tmp_dir).glob('*.m4a'))
            if generated:
                candidate = generated[0]

        if candidate is None:
            raise RuntimeError('mlx-audio returned no output file.')

        _normalize_output_to_wav(candidate, output_wav)


def synthesize_with_runtime(
    text: str,
    model_dir: Path,
    output_wav: Path,
    language: str | None,
    steps: int | None = None,
    ref_audio_path: Path | None = None,
    ref_audio_text: str | None = None,
    design_instruction: str | None = None,
    voice_name: str | None = None,
    temp_dir: Path | None = None,
) -> None:
    module = _find_runtime_module()
    if module is None:
        raise RuntimeError('No supported MLX TTS runtime module found.')

    if module.__name__ == 'mlx_audio':
        _synthesize_with_mlx_audio(
            text=text,
            model_dir=model_dir,
            output_wav=output_wav,
            language=language,
            steps=steps,
            ref_audio_path=ref_audio_path,
            ref_audio_text=ref_audio_text,
            design_instruction=design_instruction,
            voice_name=voice_name,
            temp_dir=temp_dir,
        )
        return

    # Variant A: module-level synthesize_to_file API
    synthesize_to_file = getattr(module, 'synthesize_to_file', None)
    if callable(synthesize_to_file):
        kwargs = {
            'text': text,
            'model_path': str(model_dir),
            'output_path': str(output_wav),
            'language': language,
        }
        result = _call_with_optional_voice_clone(
            synthesize_to_file,
            kwargs,
            steps,
            ref_audio_path=ref_audio_path,
        )
        if isinstance(result, (str, Path)):
            _normalize_output_to_wav(result, output_wav)
        return

    # Variant B: module-level synthesize returning path
    synthesize = getattr(module, 'synthesize', None)
    if callable(synthesize):
        kwargs = {
            'text': text,
            'model_path': str(model_dir),
            'language': language,
        }
        result = _call_with_optional_voice_clone(
            synthesize,
            kwargs,
            steps,
            ref_audio_path=ref_audio_path,
        )
        if isinstance(result, (str, Path)):
            _normalize_output_to_wav(result, output_wav)
            return

    # Variant C: class-based API
    engine_cls = getattr(module, 'QwenTTS', None)
    if engine_cls is not None:
        engine = engine_cls(model_path=str(model_dir))
        if hasattr(engine, 'synthesize_to_file'):
            kwargs = {
                'text': text,
                'output_path': str(output_wav),
                'language': language,
            }
            result = _call_with_optional_voice_clone(
                engine.synthesize_to_file,
                kwargs,
                steps,
                ref_audio_path=ref_audio_path,
            )
            if isinstance(result, (str, Path)):
                _normalize_output_to_wav(result, output_wav)
            return
        if hasattr(engine, 'synthesize'):
            kwargs = {
                'text': text,
                'language': language,
            }
            result = _call_with_optional_voice_clone(
                engine.synthesize,
                kwargs,
                steps,
                ref_audio_path=ref_audio_path,
            )
            if isinstance(result, (str, Path)):
                _normalize_output_to_wav(result, output_wav)
                return

    raise RuntimeError('Installed MLX runtime does not expose a supported API shape.')

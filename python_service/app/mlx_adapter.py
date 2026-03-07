from __future__ import annotations

import importlib
import random
import re
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
_MLX_AUDIO_LANGUAGE_ALIASES = {
    'auto': 'auto',
    'de': 'german',
    'de-de': 'german',
    'de_at': 'german',
    'de-at': 'german',
    'de_ch': 'german',
    'de-ch': 'german',
    'german': 'german',
    'en': 'english',
    'en-us': 'english',
    'en_us': 'english',
    'en-gb': 'english',
    'en_gb': 'english',
    'english': 'english',
    'fr': 'french',
    'fr-fr': 'french',
    'fr_fr': 'french',
    'french': 'french',
    'it': 'italian',
    'it-it': 'italian',
    'it_it': 'italian',
    'italian': 'italian',
    'es': 'spanish',
    'es-es': 'spanish',
    'es_es': 'spanish',
    'spanish': 'spanish',
    'pt': 'portuguese',
    'pt-pt': 'portuguese',
    'pt_pt': 'portuguese',
    'pt-br': 'portuguese',
    'pt_br': 'portuguese',
    'portuguese': 'portuguese',
    'zh': 'chinese',
    'zh-cn': 'chinese',
    'zh_cn': 'chinese',
    'zh-tw': 'chinese',
    'zh_tw': 'chinese',
    'chinese': 'chinese',
    'ja': 'japanese',
    'ja-jp': 'japanese',
    'ja_jp': 'japanese',
    'japanese': 'japanese',
    'ko': 'korean',
    'ko-kr': 'korean',
    'ko_kr': 'korean',
    'korean': 'korean',
    'ru': 'russian',
    'ru-ru': 'russian',
    'ru_ru': 'russian',
    'russian': 'russian',
}


def _find_runtime_module():
    for name in RUNTIME_MODULES:
        try:
            return importlib.import_module(name)
        except Exception:  # noqa: BLE001
            continue
    return None


def runtime_available() -> bool:
    return _find_runtime_module() is not None


def _normalize_mlx_audio_language(language: str | None) -> str | None:
    if language is None:
        return None

    normalized = str(language).strip().lower()
    if not normalized:
        return None

    if normalized in _MLX_AUDIO_LANGUAGE_ALIASES:
        return _MLX_AUDIO_LANGUAGE_ALIASES[normalized]

    root = normalized.split('-', 1)[0].split('_', 1)[0]
    if root in _MLX_AUDIO_LANGUAGE_ALIASES:
        return _MLX_AUDIO_LANGUAGE_ALIASES[root]

    return 'auto'


def _extract_paths_from_result(result: Any) -> list[Path]:
    paths: list[Path] = []

    if isinstance(result, (str, Path)):
        paths.append(Path(result))
        return paths
    if isinstance(result, (list, tuple)):
        for item in result:
            paths.extend(_extract_paths_from_result(item))
        return paths
    if isinstance(result, dict):
        path_keys = {'path', 'audio_path', 'output_path', 'file', 'audio', 'wav'}
        list_keys = {'files', 'paths', 'outputs', 'segments'}
        for key, value in result.items():
            if key in path_keys and isinstance(value, (str, Path)):
                paths.append(Path(value))
                continue
            if key in list_keys:
                paths.extend(_extract_paths_from_result(value))
                continue
            if isinstance(value, (dict, list, tuple)):
                paths.extend(_extract_paths_from_result(value))
    return paths


def _dedupe_paths(paths: list[Path]) -> list[Path]:
    seen: set[str] = set()
    deduped: list[Path] = []
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _segment_sort_key(path: Path) -> tuple[int, int, str]:
    match = re.search(r'(\d+)(?!.*\d)', path.stem)
    if match:
        return (0, int(match.group(1)), path.name)
    return (1, 0, path.name)


def _collect_generated_audio_paths(tmp_dir: Path, explicit_paths: list[Path]) -> list[Path]:
    resolved_explicit: list[Path] = []
    for path in explicit_paths:
        if not path.is_absolute():
            candidate = (tmp_dir / path).resolve()
        else:
            candidate = path.resolve()
        if candidate.exists():
            resolved_explicit.append(candidate)

    if resolved_explicit:
        return sorted(_dedupe_paths(resolved_explicit), key=_segment_sort_key)

    collected: list[Path] = []
    for pattern in ('out_*.wav', 'out_*.mp3', 'out_*.m4a', '*.wav', '*.mp3', '*.m4a'):
        collected.extend(sorted(tmp_dir.glob(pattern), key=_segment_sort_key))
    existing = [path.resolve() for path in collected if path.exists()]
    return sorted(_dedupe_paths(existing), key=_segment_sort_key)


def _call_with_step_variants(func: Any, kwargs: dict[str, Any], steps: int | None):
    if steps is None:
        return func(**kwargs)

    for key in ('steps', 'num_steps', 'inference_steps', 'n_steps', 'diffusion_steps'):
        try:
            return func(**{**kwargs, key: steps})
        except TypeError:
            continue
    return func(**kwargs)


def _seed_runtime_random_generators(random_seed: int | None) -> None:
    if random_seed is None:
        return
    seed_value = int(random_seed)
    random.seed(seed_value)
    try:
        numpy_mod = importlib.import_module('numpy')
        if hasattr(numpy_mod, 'random') and hasattr(numpy_mod.random, 'seed'):
            numpy_mod.random.seed(seed_value % (2**32))
    except Exception:  # noqa: BLE001
        pass
    try:
        mlx_core = importlib.import_module('mlx.core')
        mlx_random = getattr(mlx_core, 'random', None)
        if mlx_random is not None and hasattr(mlx_random, 'seed'):
            mlx_random.seed(seed_value)
    except Exception:  # noqa: BLE001
        pass


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


def _normalize_segment_to_pcm_wav(path_like: str | Path, output_wav: Path) -> None:
    source = Path(path_like)
    if not source.exists():
        raise RuntimeError(f'Runtime produced missing segment file: {source}')

    subprocess.run(
        [
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            str(source),
            '-ar',
            '24000',
            '-ac',
            '1',
            '-c:a',
            'pcm_s16le',
            str(output_wav),
        ],
        check=True,
    )


def _concat_wav_segments(segment_paths: list[Path], output_wav: Path, temp_dir: Path) -> None:
    if not segment_paths:
        raise RuntimeError('No segment paths available for concatenation.')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=temp_dir) as list_file:
        for segment in segment_paths:
            escaped = str(segment).replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        list_path = Path(list_file.name)

    try:
        subprocess.run(
            [
                'ffmpeg',
                '-y',
                '-hide_banner',
                '-loglevel',
                'error',
                '-f',
                'concat',
                '-safe',
                '0',
                '-i',
                str(list_path),
                '-c',
                'copy',
                str(output_wav),
            ],
            check=True,
        )
    finally:
        list_path.unlink(missing_ok=True)


def _probe_duration_ms(audio_path: Path) -> int:
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
        available = ', '.join(sorted(supported_speakers))
        raise RuntimeError(
            f'Unknown CustomVoice speaker "{requested_voice}". Available speakers: {available}'
        )

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
    max_tokens: int | None,
    temperature: float | None,
    top_k: int | None,
    top_p: float | None,
    repetition_penalty: float | None,
    random_seed: int | None,
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
    normalized_language = _normalize_mlx_audio_language(language)
    tts_model_type = str(getattr(getattr(model, 'config', None), 'tts_model_type', '')).strip().lower()
    supported_speakers = [str(item) for item in (getattr(model, 'supported_speakers', None) or []) if str(item).strip()]

    with tempfile.TemporaryDirectory(prefix='mlx-audio-gen-', dir=temp_dir) as tmp_dir:
        prefix = Path(tmp_dir) / 'out'
        kwargs: dict[str, Any] = {
            'model': model,
            'text': text,
            'file_prefix': str(prefix),
        }
        _seed_runtime_random_generators(random_seed)
        if max_tokens is not None:
            kwargs['max_tokens'] = int(max_tokens)
        if temperature is not None:
            kwargs['temperature'] = float(temperature)
        if top_k is not None:
            kwargs['top_k'] = int(top_k)
        if top_p is not None:
            kwargs['top_p'] = float(top_p)
        if repetition_penalty is not None:
            kwargs['repetition_penalty'] = float(repetition_penalty)
        if normalized_language:
            # mlx-audio expects lang_code on several runtimes; language is kept as fallback.
            kwargs['lang_code'] = normalized_language
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
            if normalized_language:
                kwargs['language'] = normalized_language
            result = _call_with_step_variants(generate_audio, kwargs, steps)

        explicit_paths = _extract_paths_from_result(result)
        generated_paths = _collect_generated_audio_paths(Path(tmp_dir), explicit_paths)
        if not generated_paths:
            raise RuntimeError('mlx-audio returned no output file.')
        if len(generated_paths) == 1:
            _normalize_output_to_wav(generated_paths[0], output_wav)
            duration_ms = _probe_duration_ms(output_wav)
            print(
                f'[mlx_adapter] merged 1 segment for output={output_wav.name}, durationMs={duration_ms}',
                flush=True,
            )
            return

        normalized_segments: list[Path] = []
        for idx, segment_path in enumerate(generated_paths):
            normalized_path = Path(tmp_dir) / f'merged-segment-{idx:04d}.wav'
            _normalize_segment_to_pcm_wav(segment_path, normalized_path)
            normalized_segments.append(normalized_path)

        _concat_wav_segments(normalized_segments, output_wav, Path(tmp_dir))
        duration_ms = _probe_duration_ms(output_wav)
        print(
            f'[mlx_adapter] merged {len(normalized_segments)} segments for output={output_wav.name}, durationMs={duration_ms}',
            flush=True,
        )


def synthesize_with_runtime(
    text: str,
    model_dir: Path,
    output_wav: Path,
    language: str | None,
    steps: int | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
    top_k: int | None = None,
    top_p: float | None = None,
    repetition_penalty: float | None = None,
    random_seed: int | None = None,
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
            max_tokens=max_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            random_seed=random_seed,
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

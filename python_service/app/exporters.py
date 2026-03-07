from __future__ import annotations

import os
import json
import math
import shutil
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from uuid import uuid4

from .config import Settings
from .schemas import WordTiming

CRAWL_RENDER_FPS = 30
AUDIO_SYNC_TOLERANCE_MS = 33
MP4_MIN_RENDER_SIDE = 320


ProgressCallback = Callable[[float, str], None]


@dataclass(frozen=True, slots=True)
class CrawlRenderProfile:
    render_fps: int
    output_fps: int
    render_resolution: tuple[int, int]
    label: str


def _run_ffmpeg(args: list[str]) -> None:
    try:
        subprocess.run(args, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        details = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(details) from exc


def concat_audio_chunks(chunks: list[Path], output_wav: Path, temp_dir: Path) -> None:
    if not chunks:
        raise ValueError('No chunks to concatenate')

    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False, dir=temp_dir) as list_file:
        for chunk in chunks:
            list_file.write(f"file '{chunk.as_posix()}'\n")
        list_path = Path(list_file.name)

    try:
        _run_ffmpeg(
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
            ]
        )
    finally:
        list_path.unlink(missing_ok=True)


def export_mp3(input_wav: Path, output_mp3: Path, bitrate: str) -> None:
    _run_ffmpeg(
        [
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel',
            'error',
            '-i',
            str(input_wav),
            '-codec:a',
            'libmp3lame',
            '-b:a',
            bitrate,
            str(output_mp3),
        ]
    )


def create_silence_wav(duration_ms: int, sample_rate: int, output_wav: Path) -> None:
    _run_ffmpeg(
        [
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel',
            'error',
            '-f',
            'lavfi',
            '-i',
            f'anullsrc=r={max(8000, sample_rate)}:cl=mono',
            '-t',
            _format_seconds(max(1, duration_ms)),
            '-c:a',
            'pcm_s16le',
            str(output_wav),
        ]
    )


def write_alignment(words: Iterable[WordTiming], output_json: Path) -> None:
    payload = [word.model_dump(mode='json') for word in words]
    output_json.write_text(json.dumps(payload, indent=2), 'utf-8')


def write_transcript(text: str, output_txt: Path) -> None:
    output_txt.write_text(text, 'utf-8')


def _ass_time(ms: int) -> str:
    cs = int(round(ms / 10))
    h = cs // 360000
    m = (cs % 360000) // 6000
    s = (cs % 6000) // 100
    centisec = cs % 100
    return f'{h}:{m:02d}:{s:02d}.{centisec:02d}'


def _escape_ass_text(text: str) -> str:
    return text.replace('\\', '\\\\').replace('{', '\\{').replace('}', '\\}')


def probe_audio_duration_ms(audio_path: Path) -> int:
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


def _probe_stream_duration_ms(media_path: Path, stream_selector: str) -> int:
    probe = subprocess.run(
        [
            'ffprobe',
            '-v',
            'error',
            '-select_streams',
            stream_selector,
            '-show_entries',
            'stream=duration',
            '-of',
            'default=noprint_wrappers=1:nokey=1',
            str(media_path),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    raw = (probe.stdout or '').strip()
    if probe.returncode == 0 and raw:
        try:
            duration_sec = float(raw)
            if duration_sec > 0:
                return max(1, int(round(duration_sec * 1000)))
        except ValueError:
            pass
    return probe_audio_duration_ms(media_path)


def _format_seconds(ms: int) -> str:
    return f'{max(1, ms) / 1000.0:.3f}'


def _notify_progress(callback: ProgressCallback | None, progress: float, message: str) -> None:
    if callback is None:
        return
    callback(min(1.0, max(0.0, float(progress))), message)


def _even(value: int) -> int:
    return max(2, int(value) - (int(value) % 2))


def _scale_resolution(base: tuple[int, int], scale: float) -> tuple[int, int]:
    width, height = base
    scaled_width = _even(max(MP4_MIN_RENDER_SIDE, int(round(width * scale))))
    scaled_height = _even(max(MP4_MIN_RENDER_SIDE, int(round(height * scale))))
    return (scaled_width, scaled_height)


def _select_crawl_profile(settings: Settings, audio_duration_ms: int) -> CrawlRenderProfile:
    duration_ms = max(1, int(audio_duration_ms))
    output_fps = max(12, int(settings.mp4_fps))
    base_resolution = _parse_resolution(settings.mp4_resolution)

    if duration_ms <= 45_000:
        return CrawlRenderProfile(
            render_fps=30,
            output_fps=output_fps,
            render_resolution=base_resolution,
            label='full',
        )
    if duration_ms <= 120_000:
        return CrawlRenderProfile(
            render_fps=20,
            output_fps=output_fps,
            render_resolution=_scale_resolution(base_resolution, 0.78),
            label='balanced',
        )
    return CrawlRenderProfile(
        render_fps=12,
        output_fps=output_fps,
        render_resolution=_scale_resolution(base_resolution, 0.62),
        label='fast',
    )


def trim_chunk_silence_inplace(
    audio_path: Path,
    temp_dir: Path,
    max_trim_ratio: float = 0.35,
    max_trim_ms: int = 1400,
) -> tuple[int, int]:
    before_ms = probe_audio_duration_ms(audio_path)
    if before_ms < 260:
        return before_ms, before_ms

    with tempfile.TemporaryDirectory(prefix='qwen3-trim-', dir=temp_dir) as tmp_dir:
        tmp_root = Path(tmp_dir)
        lead_trim = tmp_root / 'lead.wav'
        trimmed = tmp_root / 'trimmed.wav'

        _run_ffmpeg(
            [
                'ffmpeg',
                '-y',
                '-hide_banner',
                '-loglevel',
                'error',
                '-i',
                str(audio_path),
                '-af',
                (
                    'silenceremove='
                    'start_periods=1:'
                    'start_duration=0.08:'
                    'start_threshold=-42dB:'
                    'start_silence=0.03'
                ),
                str(lead_trim),
            ]
        )

        _run_ffmpeg(
            [
                'ffmpeg',
                '-y',
                '-hide_banner',
                '-loglevel',
                'error',
                '-i',
                str(lead_trim),
                '-af',
                (
                    'areverse,'
                    'silenceremove='
                    'start_periods=1:'
                    'start_duration=0.08:'
                    'start_threshold=-42dB:'
                    'start_silence=0.03,'
                    'areverse'
                ),
                str(trimmed),
            ]
        )

        after_ms = probe_audio_duration_ms(trimmed)
        trimmed_ms = before_ms - after_ms
        if trimmed_ms <= 0:
            return before_ms, before_ms
        if trimmed_ms > max_trim_ms:
            return before_ms, before_ms
        if before_ms > 0 and (trimmed_ms / before_ms) > max_trim_ratio:
            return before_ms, before_ms

        shutil.move(str(trimmed), str(audio_path))
        return before_ms, after_ms


def _clip_words_to_audio(words: list[WordTiming], audio_duration_ms: int) -> list[WordTiming]:
    if not words:
        return []

    sorted_words = sorted(words, key=lambda word: (word.startMs, word.endMs))
    clipped: list[WordTiming] = []
    cursor = 0
    safe_duration = max(1, audio_duration_ms)

    for word in sorted_words:
        start = min(max(0, word.startMs), safe_duration - 1)
        end = min(max(start + 1, word.endMs), safe_duration)
        if start < cursor:
            start = cursor
        if end <= start:
            end = min(safe_duration, start + 1)
        if start >= safe_duration:
            break

        clipped.append(
            word.model_copy(
                update={
                    'startMs': start,
                    'endMs': end,
                }
            )
        )
        cursor = end

    return clipped


def _normalize_words_to_audio(words: list[WordTiming], audio_duration_ms: int) -> list[WordTiming]:
    clipped = _clip_words_to_audio(words, audio_duration_ms)
    if not clipped:
        return []
    if clipped[-1].endMs < audio_duration_ms:
        clipped[-1] = clipped[-1].model_copy(update={'endMs': audio_duration_ms})
    return clipped


def _validate_timeline(words: list[WordTiming], audio_duration_ms: int) -> None:
    if not words:
        raise RuntimeError('No aligned words available for karaoke export.')

    previous_end = 0
    for index, word in enumerate(words):
        if word.startMs < 0:
            raise RuntimeError(f'Invalid word start at index {index}: {word.startMs}')
        if word.endMs <= word.startMs:
            raise RuntimeError(
                f'Invalid word time range at index {index}: {word.startMs}..{word.endMs}'
            )
        if word.startMs < previous_end:
            raise RuntimeError(
                f'Non-monotonic word timing at index {index}: {word.startMs} < {previous_end}'
            )
        previous_end = word.endMs

    if words[-1].endMs > audio_duration_ms:
        raise RuntimeError(
            f'Word timeline exceeds audio duration ({words[-1].endMs}ms > {audio_duration_ms}ms).'
        )


def _segment_karaoke_cs(segment: list[WordTiming]) -> list[int]:
    if not segment:
        return []

    segment_total_cs = max(1, int(round((segment[-1].endMs - segment[0].startMs) / 10)))
    raw_values = [max(1.0, (word.endMs - word.startMs) / 10.0) for word in segment]
    values = [max(1, int(value)) for value in raw_values]

    diff = segment_total_cs - sum(values)
    if diff > 0:
        order = sorted(
            range(len(raw_values)),
            key=lambda idx: raw_values[idx] - int(raw_values[idx]),
            reverse=True,
        )
        for idx in range(diff):
            values[order[idx % len(order)]] += 1
    elif diff < 0:
        order = sorted(range(len(values)), key=lambda idx: values[idx], reverse=True)
        remaining = -diff
        pointer = 0
        guard = 0
        while remaining > 0 and guard < 100000:
            idx = order[pointer % len(order)]
            if values[idx] > 1:
                values[idx] -= 1
                remaining -= 1
            pointer += 1
            guard += 1

    return values


def _build_ass(words: list[WordTiming], ass_path: Path) -> None:
    header = """[Script Info]
Title: Qwen3 TTS Karaoke
ScriptType: v4.00+
PlayResX: 1920
PlayResY: 1080
WrapStyle: 2
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Arial,56,&H00F2F2F2,&H0000D7FF,&H00141414,&H64000000,0,0,0,0,100,100,0,0,1,2,0,2,80,80,140,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    segments: list[list[WordTiming]] = []
    bucket: list[WordTiming] = []
    bucket_start = 0
    for word in words:
        if not bucket:
            bucket_start = word.startMs
        bucket.append(word)
        if len(bucket) >= 10 or (word.endMs - bucket_start) >= 5200:
            segments.append(bucket)
            bucket = []

    if bucket:
        segments.append(bucket)

    lines: list[str] = [header]
    for segment in segments:
        start = segment[0].startMs
        end = segment[-1].endMs
        durations_cs = _segment_karaoke_cs(segment)
        karaoke_parts = []
        for word, duration_cs in zip(segment, durations_cs, strict=True):
            karaoke_parts.append(f"{{\\k{duration_cs}}}{_escape_ass_text(word.word)}")
        text = ' '.join(karaoke_parts)
        lines.append(
            f"Dialogue: 0,{_ass_time(start)},{_ass_time(end)},Default,,0,0,0,,{text}"
        )

    ass_path.write_text('\n'.join(lines), 'utf-8')


def _file_for_concat(path: Path) -> str:
    escaped = str(path).replace("'", "'\\''")
    return f"file '{escaped}'"


def _load_font(image_font: Any, size: int):
    candidates = [
        '/System/Library/Fonts/Supplemental/Arial.ttf',
        '/System/Library/Fonts/Supplemental/Helvetica.ttc',
        '/System/Library/Fonts/Supplemental/Arial Unicode.ttf',
        '/System/Library/Fonts/Supplemental/NotoSans-Regular.ttf',
    ]
    for candidate in candidates:
        try:
            return image_font.truetype(candidate, size=size)
        except Exception:
            continue
    return image_font.load_default()


def _resolve_font(image_font: Any, cache: dict[int, Any], size: int):
    target = max(14, int(size))
    if target in cache:
        return cache[target]
    cache[target] = _load_font(image_font, target)
    return cache[target]


def _render_crawl_frame(
    frame_path: Path,
    words: list[WordTiming],
    active_index: int,
    current_ms: int,
    audio_duration_ms: int,
    resolution: tuple[int, int],
    font_cache: dict[int, Any],
) -> None:
    from PIL import Image, ImageDraw, ImageFont  # imported lazily to keep backend boot resilient

    width, height = resolution
    image = Image.new('RGB', (width, height), (10, 14, 24))
    draw = ImageDraw.Draw(image)

    if not words:
        image.save(frame_path, format='PNG')
        return

    words_per_line = 8
    line_gap = 28
    baseline_y = int(height * 0.88)
    rows: list[list[WordTiming]] = [
        words[index : index + words_per_line] for index in range(0, len(words), words_per_line)
    ]
    total_lines = len(rows)
    timeline_progress = min(1.0, max(0.0, current_ms / max(1, audio_duration_ms)))
    max_scroll = max(line_gap, (total_lines + 6) * line_gap)
    scroll_px = timeline_progress * max_scroll

    for line_idx, row_words in enumerate(rows):
        if not row_words:
            continue
        y = int(baseline_y + (line_idx * line_gap) - scroll_px)
        if y < -64 or y > (height + 64):
            continue

        depth = min(1.0, max(0.0, y / max(1, height)))
        scale = max(0.42, 0.74 - depth * 0.28)
        font_size = int(20 * scale)
        line_font = _resolve_font(ImageFont, font_cache, font_size)
        global_line_start = line_idx * words_per_line

        token_measures: list[int] = []
        for row_word in row_words:
            token = f'{row_word.word} '
            token_box = draw.textbbox((0, 0), token, font=line_font)
            token_measures.append(token_box[2] - token_box[0])

        line_width = sum(token_measures)
        x = int((width - line_width) / 2)
        x = int((x - width / 2) * scale + width / 2)

        for token_idx, row_word in enumerate(row_words):
            global_idx = global_line_start + token_idx
            token = f'{row_word.word} '
            if abs(global_idx - active_index) <= 2:
                fill = (176, 205, 245)
            else:
                shade = max(68, int(152 - (1.0 - depth) * 78))
                fill = (shade, min(188, shade + 18), min(222, shade + 28))
            draw.text((x, y), token, fill=fill, font=line_font)
            x += token_measures[token_idx]
    image.save(frame_path, format='PNG')


def _parse_resolution(resolution: str) -> tuple[int, int]:
    if 'x' not in resolution:
        return (1920, 1080)
    left, right = resolution.split('x', maxsplit=1)
    try:
        return (int(left), int(right))
    except ValueError:
        return (1920, 1080)


def _build_crawl_frame_plan(
    words: list[WordTiming],
    audio_duration_ms: int,
    render_fps: int = CRAWL_RENDER_FPS,
) -> list[int]:
    del words
    step_ms = 1000.0 / max(1, render_fps)
    frame_count = max(1, int(math.ceil(audio_duration_ms / step_ms)) + 1)
    plan: list[int] = []
    for idx in range(frame_count):
        value = min(audio_duration_ms, int(round(idx * step_ms)))
        if plan and value < plan[-1]:
            value = plan[-1]
        plan.append(value)
    if plan[-1] != audio_duration_ms:
        plan.append(audio_duration_ms)
    return plan


def _validate_output_audio_duration(
    output_mp4: Path,
    target_audio_ms: int,
    tolerance_ms: int = AUDIO_SYNC_TOLERANCE_MS,
) -> int:
    rendered_audio_ms = _probe_stream_duration_ms(output_mp4, 'a:0')
    if rendered_audio_ms + tolerance_ms < target_audio_ms:
        raise RuntimeError(
            f'MP4 audio underrun detected ({rendered_audio_ms}ms < {target_audio_ms}ms, tolerance {tolerance_ms}ms).'
        )
    return rendered_audio_ms


def _export_plain_mp4(settings: Settings, input_audio: Path, output_mp4: Path, target_audio_ms: int) -> None:
    _run_ffmpeg(
        [
            'ffmpeg',
            '-y',
            '-hide_banner',
            '-loglevel',
            'error',
            '-f',
            'lavfi',
            '-i',
            f'color=c=#101612:s={settings.mp4_resolution}:r={settings.mp4_fps}',
            '-i',
            str(input_audio),
            '-map',
            '0:v:0',
            '-map',
            '1:a:0',
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-preset',
            'veryfast',
            '-movflags',
            '+faststart',
            '-c:a',
            'aac',
            '-af',
            'apad=pad_dur=1',
            '-t',
            _format_seconds(target_audio_ms),
            str(output_mp4),
        ]
    )


def _export_ass_karaoke_mp4(
    settings: Settings,
    input_audio: Path,
    words: list[WordTiming],
    output_mp4: Path,
    target_audio_ms: int,
) -> None:
    with tempfile.TemporaryDirectory(prefix='qwen3-ass-', dir=settings.runtime_tmp_dir) as tmp_dir:
        ass_path = Path(tmp_dir) / 'karaoke.ass'
        _build_ass(words, ass_path)

        _run_ffmpeg(
            [
                'ffmpeg',
                '-y',
                '-hide_banner',
                '-loglevel',
                'error',
                '-f',
                'lavfi',
                '-i',
                f'color=c=#101612:s={settings.mp4_resolution}:r={settings.mp4_fps}',
                '-i',
                str(input_audio),
                '-map',
                '0:v:0',
                '-map',
                '1:a:0',
                '-vf',
                f'ass={ass_path}',
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                '-preset',
                'veryfast',
                '-movflags',
                '+faststart',
                '-c:a',
                'aac',
                '-af',
                'apad=pad_dur=1',
                '-t',
                _format_seconds(target_audio_ms),
                str(output_mp4),
            ]
        )


def _export_crawl_karaoke_mp4(
    settings: Settings,
    input_audio: Path,
    words: list[WordTiming],
    output_mp4: Path,
    target_audio_ms: int,
    progress_callback: ProgressCallback | None = None,
) -> None:
    try:
        import PIL  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f'Pillow not installed for image karaoke fallback: {exc}') from exc

    if not words:
        _export_plain_mp4(settings, input_audio, output_mp4, target_audio_ms)
        return

    profile = _select_crawl_profile(settings, target_audio_ms)
    render_fps = max(1, int(profile.render_fps))
    output_fps = max(1, int(profile.output_fps))
    width, height = profile.render_resolution
    frame_times = _build_crawl_frame_plan(words, target_audio_ms, render_fps=render_fps)
    total_frames = max(1, len(frame_times))
    _notify_progress(
        progress_callback,
        0.08,
        f'mp4_render_profile:{profile.label}:{render_fps}fps:{width}x{height}',
    )

    with tempfile.TemporaryDirectory(prefix='qwen3-frames-', dir=settings.runtime_tmp_dir) as tmp_dir:
        temp_dir = Path(tmp_dir)
        active_indices: list[int] = []
        active_index = 0
        for frame_ms in frame_times:
            while active_index + 1 < len(words) and frame_ms >= words[active_index].endMs:
                active_index += 1
            active_indices.append(active_index)

        jobs = [(idx, frame_times[idx], active_indices[idx]) for idx in range(total_frames)]
        progress_step = max(1, total_frames // 60)
        completed = 0
        local_state = threading.local()

        def render_job(job: tuple[int, int, int]) -> None:
            idx, frame_ms, word_index = job
            frame_path = temp_dir / f'frame-{idx:06d}.png'
            font_cache = getattr(local_state, 'font_cache', None)
            if font_cache is None:
                font_cache = {}
                setattr(local_state, 'font_cache', font_cache)
            _render_crawl_frame(
                frame_path=frame_path,
                words=words,
                active_index=word_index,
                current_ms=frame_ms,
                audio_duration_ms=target_audio_ms,
                resolution=(width, height),
                font_cache=font_cache,
            )

        max_workers = 1
        if total_frames >= 80:
            max_workers = min(8, max(2, (os.cpu_count() or 4) - 1))

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            for _ in pool.map(render_job, jobs, chunksize=8):
                completed += 1
                if completed == 1 or completed == total_frames or (completed % progress_step) == 0:
                    render_progress = 0.08 + (0.76 * (completed / total_frames))
                    _notify_progress(
                        progress_callback,
                        render_progress,
                        f'mp4_render_frames:{completed}/{total_frames}',
                    )

        _notify_progress(progress_callback, 0.87, 'mp4_mux')
        _run_ffmpeg(
            [
                'ffmpeg',
                '-y',
                '-hide_banner',
                '-loglevel',
                'error',
                '-start_number',
                '0',
                '-framerate',
                str(render_fps),
                '-i',
                str(temp_dir / 'frame-%06d.png'),
                '-i',
                str(input_audio),
                '-map',
                '0:v:0',
                '-map',
                '1:a:0',
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                '-movflags',
                '+faststart',
                '-preset',
                'veryfast',
                '-r',
                str(output_fps),
                '-c:a',
                'aac',
                '-af',
                'apad=pad_dur=1',
                '-t',
                _format_seconds(target_audio_ms),
                str(output_mp4),
            ]
        )
        _notify_progress(progress_callback, 0.96, 'mp4_mux_done')


def export_karaoke_mp4(
    settings: Settings,
    input_audio: Path,
    words: list[WordTiming],
    output_mp4: Path,
    progress_callback: ProgressCallback | None = None,
) -> str | None:
    _notify_progress(progress_callback, 0.02, 'mp4_prepare')
    target_audio_ms = probe_audio_duration_ms(input_audio)
    normalized_words = _normalize_words_to_audio(words, target_audio_ms)
    _validate_timeline(normalized_words, target_audio_ms)
    _notify_progress(progress_callback, 0.06, 'mp4_timeline_ready')
    fallback_warning: str | None = None
    output_mp4.parent.mkdir(parents=True, exist_ok=True)
    temp_output_mp4 = output_mp4.with_name(f'{output_mp4.stem}.{uuid4().hex[:8]}.tmp{output_mp4.suffix}')
    try:
        try:
            _export_crawl_karaoke_mp4(
                settings=settings,
                input_audio=input_audio,
                words=normalized_words,
                output_mp4=temp_output_mp4,
                target_audio_ms=target_audio_ms,
                progress_callback=progress_callback,
            )
        except Exception as crawl_exc:
            try:
                _notify_progress(progress_callback, 0.60, 'mp4_plain_fallback')
                _export_plain_mp4(settings, input_audio, temp_output_mp4, target_audio_ms)
                fallback_warning = (
                    'Crawl renderer failed; plain fallback used (no burned text). '
                    f'Reason: {crawl_exc}'
                )
            except Exception as plain_exc:
                raise RuntimeError(
                    f'Karaoke export failed (crawl={crawl_exc}; plain={plain_exc})'
                ) from plain_exc

        _validate_output_audio_duration(temp_output_mp4, target_audio_ms)
        _notify_progress(progress_callback, 0.99, 'mp4_validate_done')
        temp_output_mp4.replace(output_mp4)
    finally:
        if temp_output_mp4.exists():
            temp_output_mp4.unlink(missing_ok=True)
    _notify_progress(progress_callback, 1.0, 'mp4_done')
    return fallback_warning

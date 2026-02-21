from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Iterable

from .config import Settings
from .schemas import WordTiming


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


def _render_karaoke_frame(frame_path: Path, words: list[WordTiming], active_index: int, resolution: tuple[int, int]) -> None:
    from PIL import Image, ImageDraw, ImageFont  # imported lazily to keep backend boot resilient

    width, height = resolution
    image = Image.new('RGB', (width, height), (16, 22, 18))
    draw = ImageDraw.Draw(image)

    title_font = _load_font(ImageFont, 42)
    text_font = _load_font(ImageFont, 56)
    tiny_font = _load_font(ImageFont, 28)

    header = 'Qwen3 TTS'
    header_bbox = draw.textbbox((0, 0), header, font=title_font)
    draw.text(
        ((width - (header_bbox[2] - header_bbox[0])) / 2, 120),
        header,
        fill=(210, 236, 219),
        font=title_font,
    )

    start = max(0, active_index - 4)
    end = min(len(words), active_index + 5)
    context = words[start:end]

    cursor_x = 0
    word_widths: list[int] = []
    for word in context:
        token = f'{word.word} '
        box = draw.textbbox((0, 0), token, font=text_font)
        width_px = box[2] - box[0]
        word_widths.append(width_px)
        cursor_x += width_px

    x = (width - cursor_x) / 2
    y = height / 2
    for idx, word in enumerate(context):
        token = f'{word.word} '
        global_idx = start + idx
        fill = (252, 182, 74) if global_idx == active_index else (235, 240, 228)
        draw.text((x, y), token, fill=fill, font=text_font)
        x += word_widths[idx]

    active = words[active_index]
    total_ms = max(1, words[-1].endMs)
    progress = min(1.0, max(0.0, active.endMs / total_ms))
    bar_margin = 220
    bar_y = height - 170
    bar_h = 12
    draw.rounded_rectangle((bar_margin, bar_y, width - bar_margin, bar_y + bar_h), radius=6, fill=(59, 80, 65))
    draw.rounded_rectangle(
        (bar_margin, bar_y, bar_margin + int((width - 2 * bar_margin) * progress), bar_y + bar_h),
        radius=6,
        fill=(84, 186, 123),
    )

    footer = f'{active.startMs/1000:.1f}s'
    footer_bbox = draw.textbbox((0, 0), footer, font=tiny_font)
    draw.text(
        ((width - (footer_bbox[2] - footer_bbox[0])) / 2, bar_y + 26),
        footer,
        fill=(175, 198, 179),
        font=tiny_font,
    )
    image.save(frame_path, format='PNG')


def _export_plain_mp4(settings: Settings, input_audio: Path, output_mp4: Path) -> None:
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
            '-c:v',
            'libx264',
            '-pix_fmt',
            'yuv420p',
            '-preset',
            'veryfast',
            '-c:a',
            'aac',
            '-shortest',
            str(output_mp4),
        ]
    )


def _export_ass_karaoke_mp4(settings: Settings, input_audio: Path, words: list[WordTiming], output_mp4: Path) -> None:
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
                '-vf',
                f'ass={ass_path}',
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                '-preset',
                'veryfast',
                '-c:a',
                'aac',
                '-shortest',
                str(output_mp4),
            ]
        )


def _export_image_karaoke_mp4(settings: Settings, input_audio: Path, words: list[WordTiming], output_mp4: Path) -> None:
    try:
        import PIL  # noqa: F401
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(f'Pillow not installed for image karaoke fallback: {exc}') from exc

    if not words:
        _export_plain_mp4(settings, input_audio, output_mp4)
        return

    width, height = (1920, 1080)
    if 'x' in settings.mp4_resolution:
        left, right = settings.mp4_resolution.split('x', maxsplit=1)
        width, height = int(left), int(right)

    with tempfile.TemporaryDirectory(prefix='qwen3-frames-', dir=settings.runtime_tmp_dir) as tmp_dir:
        temp_dir = Path(tmp_dir)
        concat_path = temp_dir / 'frames.txt'

        lines: list[str] = []
        frame_paths: list[Path] = []
        for idx, word in enumerate(words):
            duration_sec = max(0.08, (word.endMs - word.startMs) / 1000.0)
            frame_path = temp_dir / f'frame-{idx:05d}.png'
            _render_karaoke_frame(frame_path, words, idx, (width, height))
            frame_paths.append(frame_path)
            lines.append(_file_for_concat(frame_path))
            lines.append(f'duration {duration_sec:.3f}')

        lines.append(_file_for_concat(frame_paths[-1]))
        concat_path.write_text('\n'.join(lines), 'utf-8')

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
                str(concat_path),
                '-i',
                str(input_audio),
                '-c:v',
                'libx264',
                '-pix_fmt',
                'yuv420p',
                '-r',
                str(settings.mp4_fps),
                '-c:a',
                'aac',
                '-shortest',
                str(output_mp4),
            ]
        )


def export_karaoke_mp4(
    settings: Settings,
    input_audio: Path,
    words: list[WordTiming],
    output_mp4: Path,
) -> None:
    audio_duration_ms = probe_audio_duration_ms(input_audio)
    normalized_words = _clip_words_to_audio(words, audio_duration_ms)
    _validate_timeline(normalized_words, audio_duration_ms)
    try:
        _export_ass_karaoke_mp4(settings, input_audio, normalized_words, output_mp4)
    except Exception as ass_exc:
        try:
            _export_image_karaoke_mp4(settings, input_audio, normalized_words, output_mp4)
        except Exception as image_exc:
            raise RuntimeError(
                f'Karaoke export failed (ass={ass_exc}; image={image_exc})'
            ) from image_exc

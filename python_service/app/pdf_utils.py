from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from statistics import median
from typing import Iterator

import fitz

from .schemas import BoundingBox


@dataclass(slots=True)
class ExtractedWord:
    word: str
    page: int
    bbox: BoundingBox | None


@dataclass(slots=True)
class ChunkBreak:
    afterWordIndex: int
    durationMs: int
    reason: str  # 'heading' | 'paragraph'


@dataclass(slots=True)
class TextChunk:
    text: str
    words: list[ExtractedWord]
    breaks: list[ChunkBreak] = field(default_factory=list)


@dataclass(slots=True)
class PdfExtractionResult:
    total_pages: int
    chunks: list[TextChunk]


@dataclass(slots=True)
class PdfChunkItem:
    chunk: TextChunk
    pageStart: int
    pageEnd: int
    totalPages: int


HEADING_BREAK_MS = 280
PARAGRAPH_BREAK_MS = 160


def _upper_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    if not letters:
        return 0.0
    upper = sum(1 for char in letters if char.isupper())
    return upper / len(letters)


def _looks_like_heading(text: str, word_count: int, line_height: float, median_height: float) -> bool:
    if not text or word_count <= 0:
        return False

    if word_count > 10:
        return False

    relative_height = (line_height / median_height) if median_height > 0 else 1.0
    uppercase_ratio = _upper_ratio(text)

    title_markers = (':', '-', '–', '—')
    marker_hit = any(text.endswith(marker) for marker in title_markers)
    title_case_like = text[:1].isupper() and text.lower() != text and not text.endswith('.')

    return relative_height >= 1.15 and (uppercase_ratio >= 0.55 or marker_hit or title_case_like)


def _detect_page_breaks(
    lines: list[dict[str, float | int | str]],
    heading_ms: int = HEADING_BREAK_MS,
    paragraph_ms: int = PARAGRAPH_BREAK_MS,
) -> list[ChunkBreak]:
    if not lines:
        return []

    heights = [float(line['height']) for line in lines if float(line['height']) > 0]
    median_height = median(heights) if heights else 1.0

    gaps: list[float] = []
    for idx in range(len(lines) - 1):
        current_bottom = float(lines[idx]['y1'])
        next_top = float(lines[idx + 1]['y0'])
        gap = next_top - current_bottom
        if gap > 0:
            gaps.append(gap)
    median_gap = median(gaps) if gaps else 0.0

    selected: dict[int, ChunkBreak] = {}
    for idx, line in enumerate(lines):
        line_text = str(line['text']).strip()
        word_count = int(line['wordCount'])
        line_height = float(line['height'])
        line_end = int(line['endIndex'])

        if _looks_like_heading(line_text, word_count, line_height, median_height):
            selected[line_end] = ChunkBreak(
                afterWordIndex=line_end,
                durationMs=heading_ms,
                reason='heading',
            )

        if idx < len(lines) - 1:
            gap = float(lines[idx + 1]['y0']) - float(line['y1'])
            if median_gap > 0:
                paragraph_break = gap >= max(6.0, median_gap * 1.65)
            else:
                paragraph_break = gap >= 10.0
            if paragraph_break:
                existing = selected.get(line_end)
                if existing is None or paragraph_ms > existing.durationMs:
                    selected[line_end] = ChunkBreak(
                        afterWordIndex=line_end,
                        durationMs=paragraph_ms,
                        reason='paragraph',
                    )

    return [selected[index] for index in sorted(selected)]


def _resolve_page_window(doc: fitz.Document, page_range: tuple[int, int] | None) -> tuple[int, int, int]:
    if doc.page_count == 0:
        return 0, -1, 0

    start_page = 0
    end_page = doc.page_count - 1

    if page_range is not None:
        start_page = max(0, page_range[0])
        end_page = min(doc.page_count - 1, page_range[1])
        if end_page < start_page:
            raise ValueError('Invalid page range.')

    selected_pages = max(0, end_page - start_page + 1)
    return start_page, end_page, selected_pages


def iter_pdf_chunks(
    pdf_path: str,
    page_range: tuple[int, int] | None = None,
    pages_per_chunk: int = 6,
) -> Iterator[PdfChunkItem]:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f'PDF not found: {pdf_path}')

    safe_pages_per_chunk = max(1, pages_per_chunk)
    with fitz.open(path) as doc:
        start_page, end_page, selected_pages = _resolve_page_window(doc, page_range)
        if selected_pages == 0:
            return

        bucket_text_parts: list[str] = []
        bucket_words: list[ExtractedWord] = []
        bucket_breaks: list[ChunkBreak] = []
        bucket_start_page: int | None = None

        for page_idx in range(start_page, end_page + 1):
            page = doc.load_page(page_idx)
            words_raw = page.get_text('words')
            words_raw.sort(key=lambda item: (item[5], item[6], item[7]))

            page_words: list[ExtractedWord] = []
            text_parts: list[str] = []
            line_entries: list[dict[str, float | int | str]] = []
            current_line_key: tuple[int, int] | None = None
            current_line_raw: list[tuple] = []

            def flush_line() -> None:
                nonlocal current_line_key, current_line_raw
                if not current_line_raw:
                    return

                line_tokens: list[str] = []
                line_start = len(page_words)
                y0 = float('inf')
                y1 = float('-inf')

                for line_item in current_line_raw:
                    token = str(line_item[4]).strip()
                    if not token:
                        continue
                    x0 = float(line_item[0])
                    top = float(line_item[1])
                    x1 = float(line_item[2])
                    bottom = float(line_item[3])
                    y0 = min(y0, top)
                    y1 = max(y1, bottom)
                    text_parts.append(token)
                    line_tokens.append(token)
                    page_words.append(
                        ExtractedWord(
                            word=token,
                            page=page_idx,
                            bbox=BoundingBox(x0=x0, y0=top, x1=x1, y1=bottom),
                        )
                    )

                if line_tokens:
                    line_end = len(page_words) - 1
                    line_entries.append(
                        {
                            'text': ' '.join(line_tokens),
                            'wordCount': len(line_tokens),
                            'startIndex': line_start,
                            'endIndex': line_end,
                            'y0': y0 if y0 != float('inf') else 0.0,
                            'y1': y1 if y1 != float('-inf') else 0.0,
                            'height': max(0.1, y1 - y0),
                        }
                    )

                current_line_raw = []
                current_line_key = None

            for item in words_raw:
                line_key = (int(item[5]), int(item[6]))
                if current_line_key is None:
                    current_line_key = line_key
                if line_key != current_line_key:
                    flush_line()
                    current_line_key = line_key
                current_line_raw.append(item)

            flush_line()

            page_text = ' '.join(text_parts)
            page_breaks = _detect_page_breaks(line_entries)
            if page_text:
                if bucket_start_page is None:
                    bucket_start_page = page_idx
                bucket_offset = len(bucket_words)
                bucket_text_parts.append(page_text)
                bucket_words.extend(page_words)
                for page_break in page_breaks:
                    mapped_index = bucket_offset + page_break.afterWordIndex
                    if mapped_index < bucket_offset or mapped_index >= len(bucket_words):
                        continue
                    # De-duplicate same marker index by keeping the longer pause.
                    existing_idx = next(
                        (
                            idx
                            for idx, item in enumerate(bucket_breaks)
                            if item.afterWordIndex == mapped_index
                        ),
                        None,
                    )
                    mapped_break = ChunkBreak(
                        afterWordIndex=mapped_index,
                        durationMs=page_break.durationMs,
                        reason=page_break.reason,
                    )
                    if existing_idx is None:
                        bucket_breaks.append(mapped_break)
                    elif bucket_breaks[existing_idx].durationMs < mapped_break.durationMs:
                        bucket_breaks[existing_idx] = mapped_break

            local_page_number = page_idx - start_page + 1
            reached_boundary = (local_page_number % safe_pages_per_chunk) == 0
            is_last_page = page_idx == end_page
            if reached_boundary or is_last_page:
                if bucket_text_parts:
                    deduped_breaks = sorted(
                        {
                            item.afterWordIndex: item for item in bucket_breaks
                        }.values(),
                        key=lambda item: item.afterWordIndex,
                    )
                    yield PdfChunkItem(
                        chunk=TextChunk(
                            text='\n\n'.join(bucket_text_parts),
                            words=bucket_words.copy(),
                            breaks=deduped_breaks,
                        ),
                        pageStart=bucket_start_page if bucket_start_page is not None else page_idx,
                        pageEnd=page_idx,
                        totalPages=selected_pages,
                    )
                bucket_text_parts.clear()
                bucket_words.clear()
                bucket_breaks.clear()
                bucket_start_page = None


def extract_pdf_chunks(
    pdf_path: str,
    page_range: tuple[int, int] | None = None,
    pages_per_chunk: int = 6,
) -> PdfExtractionResult:
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f'PDF not found: {pdf_path}')

    with fitz.open(path) as doc:
        _, _, selected_pages = _resolve_page_window(doc, page_range)

    chunks = [item.chunk for item in iter_pdf_chunks(pdf_path, page_range=page_range, pages_per_chunk=pages_per_chunk)]
    return PdfExtractionResult(total_pages=selected_pages, chunks=chunks)

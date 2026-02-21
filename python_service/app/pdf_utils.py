from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import fitz

from .schemas import BoundingBox


@dataclass(slots=True)
class ExtractedWord:
    word: str
    page: int
    bbox: BoundingBox | None


@dataclass(slots=True)
class TextChunk:
    text: str
    words: list[ExtractedWord]


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
        bucket_start_page: int | None = None

        for page_idx in range(start_page, end_page + 1):
            page = doc.load_page(page_idx)
            words_raw = page.get_text('words')
            words_raw.sort(key=lambda item: (item[5], item[6], item[7]))

            page_words: list[ExtractedWord] = []
            text_parts: list[str] = []
            for item in words_raw:
                token = str(item[4]).strip()
                if not token:
                    continue
                text_parts.append(token)
                page_words.append(
                    ExtractedWord(
                        word=token,
                        page=page_idx,
                        bbox=BoundingBox(x0=item[0], y0=item[1], x1=item[2], y1=item[3]),
                    )
                )

            page_text = ' '.join(text_parts)
            if page_text:
                if bucket_start_page is None:
                    bucket_start_page = page_idx
                bucket_text_parts.append(page_text)
                bucket_words.extend(page_words)

            local_page_number = page_idx - start_page + 1
            reached_boundary = (local_page_number % safe_pages_per_chunk) == 0
            is_last_page = page_idx == end_page
            if reached_boundary or is_last_page:
                if bucket_text_parts:
                    yield PdfChunkItem(
                        chunk=TextChunk(
                            text='\n\n'.join(bucket_text_parts),
                            words=bucket_words.copy(),
                        ),
                        pageStart=bucket_start_page if bucket_start_page is not None else page_idx,
                        pageEnd=page_idx,
                        totalPages=selected_pages,
                    )
                bucket_text_parts.clear()
                bucket_words.clear()
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

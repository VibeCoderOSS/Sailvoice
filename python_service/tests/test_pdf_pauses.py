from __future__ import annotations

from app.manager import _build_pdf_synthesis_text, _split_pdf_chunk_for_synthesis
from app.pdf_utils import ChunkBreak, ExtractedWord, TextChunk, _detect_page_breaks


def _build_words(tokens: list[str]) -> list[ExtractedWord]:
    return [ExtractedWord(word=token, page=0, bbox=None) for token in tokens]


def test_detect_page_breaks_heading_and_paragraph() -> None:
    lines = [
        {'text': 'CHAPTER ONE', 'wordCount': 2, 'startIndex': 0, 'endIndex': 1, 'y0': 10.0, 'y1': 28.0, 'height': 18.0},
        {'text': 'This is body', 'wordCount': 3, 'startIndex': 2, 'endIndex': 4, 'y0': 30.0, 'y1': 42.0, 'height': 12.0},
        {'text': 'Still body text', 'wordCount': 3, 'startIndex': 5, 'endIndex': 7, 'y0': 44.0, 'y1': 56.0, 'height': 12.0},
        {'text': 'New paragraph', 'wordCount': 2, 'startIndex': 8, 'endIndex': 9, 'y0': 76.0, 'y1': 88.0, 'height': 12.0},
    ]

    breaks = _detect_page_breaks(lines)
    assert [(item.afterWordIndex, item.durationMs, item.reason) for item in breaks] == [
        (1, 280, 'heading'),
        (7, 160, 'paragraph'),
    ]


def test_pdf_synthesis_text_inserts_structural_newlines() -> None:
    chunk = TextChunk(
        text='Alpha Beta Gamma Delta Epsilon',
        words=_build_words(['Alpha', 'Beta', 'Gamma', 'Delta', 'Epsilon']),
        breaks=[
            ChunkBreak(afterWordIndex=1, durationMs=280, reason='heading'),
            ChunkBreak(afterWordIndex=3, durationMs=160, reason='paragraph'),
        ],
    )

    rendered = _build_pdf_synthesis_text(chunk)
    assert rendered == 'Alpha Beta. Gamma Delta, Epsilon'
    assert '\n' not in rendered


def test_pdf_synthesis_text_keeps_word_sequence() -> None:
    words = ['One', 'Two', 'Three', 'Four', 'Five']
    chunk = TextChunk(
        text=' '.join(words),
        words=_build_words(words),
        breaks=[
            ChunkBreak(afterWordIndex=1, durationMs=280, reason='heading'),
            ChunkBreak(afterWordIndex=3, durationMs=160, reason='paragraph'),
        ],
    )

    rendered = _build_pdf_synthesis_text(chunk)
    rendered_words = rendered.replace('\n', ' ').split()
    normalized_words = [token.strip('.,;:!?') for token in rendered_words]
    assert normalized_words == words


def test_pdf_subchunk_split_respects_word_order_and_limits() -> None:
    tokens = [f'Wort{i}' for i in range(1, 19)]
    chunk = TextChunk(
        text=' '.join(tokens),
        words=_build_words(tokens),
        breaks=[
            ChunkBreak(afterWordIndex=5, durationMs=280, reason='heading'),
            ChunkBreak(afterWordIndex=11, durationMs=160, reason='paragraph'),
        ],
    )

    subchunks = _split_pdf_chunk_for_synthesis(
        chunk,
        max_words=7,
        max_chars=45,
    )

    assert len(subchunks) >= 3
    flattened = [word.word for part in subchunks for word in part.words]
    assert flattened == tokens
    assert all(len(part.words) <= 7 for part in subchunks)
    assert all(len(part.text) <= 45 for part in subchunks)

from app.tts_engine import chunk_text


def test_chunk_text_keeps_content():
    source = (
        'Sentence one is short. Sentence two is a little longer than the first one. '
        'Sentence three ensures we split the input into multiple chunks for generation.'
    )
    chunks = chunk_text(source, max_chars=50)
    assert len(chunks) >= 2
    merged = ' '.join(chunks)
    assert 'Sentence one' in merged
    assert 'Sentence three' in merged

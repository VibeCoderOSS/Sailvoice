from app.language import detect_language


def test_detect_language_german_text():
    result = detect_language('Dies ist ein deutscher Testsatz mit mehreren Wörtern und Kontext.')
    assert result.code in {'de', 'auto'}
    assert isinstance(result.candidates, list)
    assert isinstance(result.needs_confirmation, bool)


def test_detect_language_empty_text():
    result = detect_language('')
    assert result.code == 'auto'
    assert result.needs_confirmation is True

from __future__ import annotations

from dataclasses import dataclass

from langdetect import DetectorFactory, LangDetectException, detect_langs

DetectorFactory.seed = 0


LANG_ALIAS = {
    'de': 'de',
    'deu': 'de',
    'ger': 'de',
    'en': 'en',
    'eng': 'en',
    'fr': 'fr',
    'fra': 'fr',
    'it': 'it',
    'ita': 'it',
    'es': 'es',
    'spa': 'es',
    'pt': 'pt',
    'por': 'pt',
    'ja': 'ja',
    'jpn': 'ja',
    'ko': 'ko',
    'kor': 'ko',
    'ru': 'ru',
    'rus': 'ru',
    'zh-cn': 'zh',
    'zh-tw': 'zh',
    'zh': 'zh',
}


@dataclass(slots=True)
class DetectionResult:
    code: str
    confidence: float
    candidates: list[tuple[str, float]]
    needs_confirmation: bool
    reason: str


def _normalize_code(code: str) -> str:
    token = (code or '').strip().lower()
    return LANG_ALIAS.get(token, token)


def detect_language(text: str) -> DetectionResult:
    sample = (text or '').strip()
    if not sample:
        return DetectionResult(
            code='auto',
            confidence=0.0,
            candidates=[],
            needs_confirmation=True,
            reason='empty_text',
        )

    try:
        candidates = detect_langs(sample[:4000])
    except LangDetectException:
        return DetectionResult(
            code='auto',
            confidence=0.0,
            candidates=[],
            needs_confirmation=True,
            reason='detection_failed',
        )

    if not candidates:
        return DetectionResult(
            code='auto',
            confidence=0.0,
            candidates=[],
            needs_confirmation=True,
            reason='no_candidates',
        )

    normalized: list[tuple[str, float]] = []
    seen: set[str] = set()
    for candidate in candidates:
        code = _normalize_code(candidate.lang)
        confidence = float(candidate.prob)
        if code in seen:
            continue
        seen.add(code)
        normalized.append((code, confidence))
        if len(normalized) >= 3:
            break

    top_code, top_confidence = normalized[0]
    confidence_gap = top_confidence - (normalized[1][1] if len(normalized) > 1 else 0.0)
    short_text = len(sample) < 30
    low_confidence = top_confidence < 0.80
    ambiguous = confidence_gap < 0.20
    needs_confirmation = short_text or low_confidence or ambiguous
    reason = 'ok'
    if short_text:
        reason = 'short_text'
    elif low_confidence:
        reason = 'low_confidence'
    elif ambiguous:
        reason = 'ambiguous'

    return DetectionResult(
        code=top_code,
        confidence=top_confidence,
        candidates=normalized,
        needs_confirmation=needs_confirmation,
        reason=reason,
    )

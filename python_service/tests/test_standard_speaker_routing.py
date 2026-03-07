from __future__ import annotations

from datetime import datetime, timezone

import pytest

from app.manager import JobManager, _resolve_voice_route
from app.schemas import VoiceItem


def _voice_item(*, voice_type: str) -> VoiceItem:
    return VoiceItem(
        id=f'{voice_type}-voice-id',
        name=f'{voice_type}-voice',
        type=voice_type,  # type: ignore[arg-type]
        language='en',
        createdAt=datetime.now(tz=timezone.utc),
    )


def test_resolve_voice_route_defaults_to_customvoice_with_default_speaker() -> None:
    route = _resolve_voice_route(
        requested_model_id='base',
        selected_voice=None,
        requested_speaker=None,
        default_speaker='serena',
    )

    assert route.runtime_model_id == 'customvoice'
    assert route.speaker == 'serena'
    assert route.selected_voice is None


def test_resolve_voice_route_uses_explicit_preset_speaker() -> None:
    route = _resolve_voice_route(
        requested_model_id='customvoice',
        selected_voice=None,
        requested_speaker='vivian',
        default_speaker='serena',
    )

    assert route.runtime_model_id == 'customvoice'
    assert route.speaker == 'vivian'
    assert route.selected_voice is None


def test_resolve_voice_route_prioritizes_clone_voice() -> None:
    clone_voice = _voice_item(voice_type='clone')
    route = _resolve_voice_route(
        requested_model_id='customvoice',
        selected_voice=clone_voice,
        requested_speaker='serena',
        default_speaker='serena',
    )

    assert route.runtime_model_id == 'base'
    assert route.speaker is None
    assert route.selected_voice == clone_voice


def test_resolve_voice_route_prioritizes_design_voice() -> None:
    design_voice = _voice_item(voice_type='design')
    route = _resolve_voice_route(
        requested_model_id='base',
        selected_voice=design_voice,
        requested_speaker='serena',
        default_speaker='serena',
    )

    assert route.runtime_model_id == 'voicedesign'
    assert route.speaker is None
    assert route.selected_voice == design_voice


def test_validate_voice_selection_rejects_voice_and_speaker_conflict() -> None:
    with pytest.raises(ValueError, match='either voiceId or speaker'):
        JobManager._validate_voice_selection(object(), 'voice-id', 'customvoice', 'serena')


def test_validate_voice_selection_rejects_speaker_for_non_customvoice_model() -> None:
    with pytest.raises(ValueError, match='customvoice'):
        JobManager._validate_voice_selection(object(), None, 'base', 'serena')

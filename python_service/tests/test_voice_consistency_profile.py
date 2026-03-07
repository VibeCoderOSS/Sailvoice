from __future__ import annotations

from app.manager import _build_chunk_generation_profile, _compute_job_seed


def test_compute_job_seed_is_deterministic() -> None:
    seed_a = _compute_job_seed(
        job_id='job-1',
        voice_id='voice-1',
        speaker='serena',
        model_id='base',
        language='de',
        seed_base=424242,
    )
    seed_b = _compute_job_seed(
        job_id='job-1',
        voice_id='voice-1',
        speaker='serena',
        model_id='base',
        language='de',
        seed_base=424242,
    )
    seed_c = _compute_job_seed(
        job_id='job-1',
        voice_id='voice-2',
        speaker='serena',
        model_id='base',
        language='de',
        seed_base=424242,
    )

    assert seed_a == seed_b
    assert seed_a != seed_c


def test_compute_job_seed_changes_with_speaker() -> None:
    seed_serena = _compute_job_seed(
        job_id='job-1',
        voice_id=None,
        speaker='serena',
        model_id='customvoice',
        language='en',
        seed_base=424242,
    )
    seed_vivian = _compute_job_seed(
        job_id='job-1',
        voice_id=None,
        speaker='vivian',
        model_id='customvoice',
        language='en',
        seed_base=424242,
    )

    assert seed_serena != seed_vivian


def test_chunk_profile_forces_strict_when_multi_chunk() -> None:
    profile = _build_chunk_generation_profile(
        job_id='job-42',
        voice_id='voice-abc',
        speaker=None,
        model_id='customvoice',
        language='en',
        chunk_count=6,
        consistency_mode='natural',
        seed_base=424242,
        strict_temperature=0.18,
        strict_top_k=20,
        strict_top_p=0.85,
        strict_repetition_penalty=1.12,
    )

    assert profile.mode == 'strict'
    assert profile.random_seed is not None
    assert profile.temperature == 0.18
    assert profile.top_k == 20
    assert profile.top_p == 0.85
    assert profile.repetition_penalty == 1.12


def test_chunk_profile_allows_natural_for_single_chunk() -> None:
    profile = _build_chunk_generation_profile(
        job_id='job-42',
        voice_id='voice-abc',
        speaker=None,
        model_id='customvoice',
        language='en',
        chunk_count=1,
        consistency_mode='natural',
        seed_base=424242,
        strict_temperature=0.18,
        strict_top_k=20,
        strict_top_p=0.85,
        strict_repetition_penalty=1.12,
    )

    assert profile.mode == 'natural'
    assert profile.random_seed is None
    assert profile.temperature is None
    assert profile.top_k is None
    assert profile.top_p is None
    assert profile.repetition_penalty is None

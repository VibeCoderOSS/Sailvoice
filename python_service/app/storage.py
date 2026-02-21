from __future__ import annotations

import base64
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from .config import Settings
from .schemas import JobStatus, VoiceItem


def _utc_now() -> datetime:
    return datetime.now(tz=timezone.utc)


class PersistentStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.jobs: dict[str, JobStatus] = {}
        self.voice_index: dict[str, dict[str, Any]] = {}
        self.asset_index: dict[str, str] = {}
        self._voice_key = self._derive_voice_key(settings.voice_secret_b64)
        self._load_jobs()
        self._load_voices()

    def _derive_voice_key(self, secret_b64: str) -> bytes:
        if not secret_b64:
            return hashlib.sha256(os.urandom(32)).digest()

        try:
            decoded = base64.b64decode(secret_b64)
            if len(decoded) == 32:
                return decoded
        except Exception:
            pass

        return hashlib.sha256(secret_b64.encode('utf-8')).digest()

    def _load_jobs(self) -> None:
        path = self.settings.jobs_index_path
        if not path.exists():
            return

        data = json.loads(path.read_text('utf-8'))
        for item in data.get('jobs', []):
            job = JobStatus.model_validate(item)
            self.jobs[job.id] = job
            for artifact in job.artifacts:
                self.asset_index[artifact.id] = artifact.path

    def save_jobs(self) -> None:
        payload = {'jobs': [job.model_dump(mode='json') for job in self.list_jobs()]}
        self.settings.jobs_index_path.write_text(json.dumps(payload, indent=2), 'utf-8')

    def list_jobs(self) -> list[JobStatus]:
        return sorted(self.jobs.values(), key=lambda item: item.createdAt, reverse=True)

    def upsert_job(self, job: JobStatus) -> None:
        self.jobs[job.id] = job
        for artifact in job.artifacts:
            self.asset_index[artifact.id] = artifact.path
        self.save_jobs()

    def get_job(self, job_id: str) -> JobStatus | None:
        return self.jobs.get(job_id)

    def delete_job(self, job_id: str) -> list[str]:
        job = self.jobs.get(job_id)
        if not job:
            raise KeyError('Job not found')

        artifact_paths: set[str] = set()
        for artifact in job.artifacts:
            self.asset_index.pop(artifact.id, None)
            if artifact.path:
                artifact_paths.add(artifact.path)

        self.jobs.pop(job_id, None)
        self.save_jobs()
        return sorted(artifact_paths)

    def register_asset(self, file_path: Path) -> str:
        asset_id = uuid4().hex
        self.asset_index[asset_id] = str(file_path)
        return asset_id

    def get_asset_path(self, asset_id: str) -> str | None:
        return self.asset_index.get(asset_id)

    def _load_voices(self) -> None:
        path = self.settings.voices_index_path
        if not path.exists():
            self.voice_index = {}
            return
        payload = json.loads(path.read_text('utf-8'))
        voices = payload.get('voices', {})
        changed = False
        normalized: dict[str, dict[str, Any]] = {}
        for voice_id, voice in voices.items():
            record = dict(voice)
            voice_type = str(record.get('type') or 'clone').strip().lower()
            if voice_type not in {'clone', 'design'}:
                voice_type = 'clone'
            if record.get('type') != voice_type:
                record['type'] = voice_type
                changed = True
            normalized[voice_id] = record
        self.voice_index = normalized
        if changed:
            self._save_voices()

    def _save_voices(self) -> None:
        payload = {'voices': self.voice_index}
        self.settings.voices_index_path.write_text(json.dumps(payload, indent=2), 'utf-8')

    def _to_voice_item(self, voice: dict[str, Any]) -> VoiceItem:
        return VoiceItem(
            id=voice['id'],
            name=voice['name'],
            type=voice.get('type') or 'clone',
            language=voice.get('language'),
            referenceText=voice.get('referenceText'),
            description=voice.get('description'),
            createdAt=datetime.fromisoformat(voice['createdAt']),
        )

    def list_voices(self) -> list[VoiceItem]:
        items = [self._to_voice_item(voice) for voice in self.voice_index.values()]
        return sorted(items, key=lambda item: item.createdAt, reverse=True)

    def get_voice(self, voice_id: str) -> VoiceItem:
        voice = self.voice_index.get(voice_id)
        if not voice:
            raise KeyError('Voice not found')
        return self._to_voice_item(voice)

    def create_voice(self, name: str, language: str | None, audio_path: str, reference_text: str | None = None) -> VoiceItem:
        path = Path(audio_path)
        if not path.exists():
            raise FileNotFoundError(f'Audio not found: {audio_path}')

        audio_bytes = path.read_bytes()
        nonce = os.urandom(12)
        aes = AESGCM(self._voice_key)
        encrypted = aes.encrypt(nonce, audio_bytes, None)

        voice_id = uuid4().hex
        encrypted_path = self.settings.voices_dir / f'{voice_id}.bin'
        encrypted_path.write_bytes(encrypted)

        created_at = _utc_now()
        self.voice_index[voice_id] = {
            'id': voice_id,
            'name': name,
            'type': 'clone',
            'language': language,
            'referenceText': (reference_text or '').strip() or None,
            'description': None,
            'createdAt': created_at.isoformat(),
            'nonce': base64.b64encode(nonce).decode('utf-8'),
            'encryptedPath': str(encrypted_path),
            'sourceExtension': path.suffix.lower() or '.wav',
        }
        self._save_voices()

        return VoiceItem(
            id=voice_id,
            name=name,
            type='clone',
            language=language,
            referenceText=(reference_text or '').strip() or None,
            description=None,
            createdAt=created_at,
        )

    def create_design_voice(self, name: str, language: str | None, description: str) -> VoiceItem:
        voice_id = uuid4().hex
        created_at = _utc_now()
        self.voice_index[voice_id] = {
            'id': voice_id,
            'name': name,
            'type': 'design',
            'language': language,
            'referenceText': None,
            'description': description.strip(),
            'createdAt': created_at.isoformat(),
        }
        self._save_voices()
        return VoiceItem(
            id=voice_id,
            name=name,
            type='design',
            language=language,
            referenceText=None,
            description=description.strip(),
            createdAt=created_at,
        )

    def rename_voice(self, voice_id: str, name: str) -> VoiceItem:
        voice = self.voice_index.get(voice_id)
        if not voice:
            raise KeyError('Voice not found')

        voice['name'] = name
        self._save_voices()
        return self._to_voice_item(voice)

    def delete_voice(self, voice_id: str) -> None:
        voice = self.voice_index.get(voice_id)
        if not voice:
            return

        encrypted_path_raw = voice.get('encryptedPath')
        if encrypted_path_raw:
            encrypted_path = Path(encrypted_path_raw)
            if encrypted_path.exists():
                encrypted_path.unlink()

        self.voice_index.pop(voice_id, None)
        self._save_voices()

    def materialize_voice_reference(self, voice_id: str) -> Path:
        voice = self.voice_index.get(voice_id)
        if not voice:
            raise KeyError('Voice not found')

        if str(voice.get('type') or 'clone') != 'clone':
            raise RuntimeError('Selected voice is not a clone voice.')

        encrypted_path_raw = voice.get('encryptedPath')
        if not encrypted_path_raw:
            raise RuntimeError('Voice metadata is missing encryptedPath.')
        encrypted_path = Path(encrypted_path_raw)
        if not encrypted_path.exists():
            raise FileNotFoundError(f'Encrypted voice file missing: {encrypted_path}')

        nonce_raw = voice.get('nonce')
        if not nonce_raw:
            raise RuntimeError('Voice metadata is missing nonce.')

        nonce = base64.b64decode(nonce_raw)
        encrypted = encrypted_path.read_bytes()
        aes = AESGCM(self._voice_key)
        decrypted = aes.decrypt(nonce, encrypted, None)

        suffix = str(voice.get('sourceExtension') or '.wav').strip()
        if not suffix.startswith('.'):
            suffix = f'.{suffix}'
        target = self.settings.runtime_tmp_dir / f'voice-ref-{voice_id}-{uuid4().hex[:8]}{suffix}'
        target.write_bytes(decrypted)
        return target

    def get_voice_reference_text(self, voice_id: str) -> str | None:
        voice = self.voice_index.get(voice_id)
        if not voice:
            raise KeyError('Voice not found')
        value = voice.get('referenceText')
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    def update_clone_voice_metadata(
        self,
        voice_id: str,
        reference_text: str | None = None,
        language: str | None = None,
    ) -> VoiceItem:
        voice = self.voice_index.get(voice_id)
        if not voice:
            raise KeyError('Voice not found')
        if str(voice.get('type') or 'clone') != 'clone':
            raise RuntimeError('Selected voice is not a clone voice.')

        changed = False

        if reference_text is not None:
            normalized_ref = reference_text.strip() or None
            if normalized_ref != voice.get('referenceText'):
                voice['referenceText'] = normalized_ref
                changed = True

        if language is not None:
            normalized_language = language.strip().lower() or None
            if normalized_language and not voice.get('language'):
                voice['language'] = normalized_language
                changed = True

        if changed:
            self._save_voices()

        return self._to_voice_item(voice)

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


ModelId = Literal['base', 'customvoice', 'voicedesign']
ModelCatalogId = Literal['base', 'customvoice', 'voicedesign', 'whisperx']
ModelSource = Literal['mlx', 'official']
OutputFormat = Literal['mp3', 'mp4']
JobState = Literal['queued', 'running', 'waiting_language', 'done', 'failed']
Mp4State = Literal['not_requested', 'queued', 'running', 'done', 'failed']
AlignmentState = Literal['queued', 'running', 'done', 'failed']
VoiceType = Literal['clone', 'design']
JobPhase = Literal[
    'queued',
    'model_check',
    'model_load_ram',
    'synthesizing',
    'aligning',
    'merge_export',
    'done',
    'failed',
]


class PageRange(BaseModel):
    start: int = Field(ge=1)
    end: int = Field(ge=1)


class JobRequestText(BaseModel):
    text: str = Field(min_length=1)
    model: ModelId
    voiceId: str | None = None
    language: str | None = None
    outputFormats: list[OutputFormat] = Field(default_factory=lambda: ['mp3', 'mp4'])


class JobRequestPdf(BaseModel):
    pdfPath: str = Field(min_length=1)
    model: ModelId
    voiceId: str | None = None
    language: str | None = None
    pageRange: PageRange | None = None
    outputFormats: list[OutputFormat] = Field(default_factory=lambda: ['mp3', 'mp4'])


class BoundingBox(BaseModel):
    x0: float
    y0: float
    x1: float
    y1: float


class WordTiming(BaseModel):
    word: str
    startMs: int
    endMs: int
    confidence: float | None = None
    page: int | None = None
    bbox: BoundingBox | None = None


class Artifact(BaseModel):
    id: str
    type: Literal['mp3', 'mp4', 'alignment', 'transcript', 'audio_chunk']
    path: str
    sizeBytes: int


class JobStatus(BaseModel):
    id: str
    state: JobState
    progress: float
    createdAt: datetime
    updatedAt: datetime
    sourceType: Literal['text', 'pdf']
    sourceLabel: str
    model: ModelId
    detectedLanguage: str
    statusMessage: str | None = None
    mp4State: Mp4State = 'not_requested'
    mp4Error: str | None = None
    alignmentState: AlignmentState = 'queued'
    alignmentError: str | None = None
    alignmentCoverage: float | None = None
    audioDurationMs: int | None = None
    artifacts: list[Artifact] = Field(default_factory=list)
    warning: str | None = None
    error: str | None = None
    phase: JobPhase = 'queued'
    phaseProgress: float | None = None
    alignmentWarning: str | None = None
    words: list[WordTiming] = Field(default_factory=list)


class JobCreatedResponse(BaseModel):
    id: str


class LanguageCandidate(BaseModel):
    code: str
    confidence: float


class LanguageDecisionRequest(BaseModel):
    language: str = Field(min_length=2, max_length=12)


class VoiceCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    language: str | None = None
    audioPath: str = Field(min_length=1)
    referenceText: str | None = Field(default=None, max_length=1000)


class VoiceDesignCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)
    language: str | None = None
    description: str = Field(min_length=1, max_length=500)


class VoiceUpdateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=80)


class VoiceItem(BaseModel):
    id: str
    name: str
    type: VoiceType = 'clone'
    language: str | None = None
    referenceText: str | None = None
    description: str | None = None
    createdAt: datetime


class VoicePreviewRequest(BaseModel):
    voiceId: str | None = None
    text: str = Field(min_length=1)
    model: ModelId = 'base'


class VoicePreviewResponse(BaseModel):
    assetId: str


class ModelInfo(BaseModel):
    model: ModelCatalogId
    source: ModelSource
    repo: str
    localPath: str
    downloaded: bool


class ModelDownloadRequest(BaseModel):
    model: ModelCatalogId
    source: ModelSource = 'mlx'


class ModelDownloadResponse(BaseModel):
    model: ModelCatalogId
    source: ModelSource
    repo: str
    localPath: str
    downloaded: bool
    warning: str | None = None


class JobEventProgress(BaseModel):
    type: Literal['progress'] = 'progress'
    progress: float
    message: str
    phase: JobPhase | None = None
    phaseProgress: float | None = None


class JobEventWarning(BaseModel):
    type: Literal['warning'] = 'warning'
    message: str


class JobEventWord(BaseModel):
    type: Literal['word'] = 'word'
    payload: WordTiming


class JobEventAudioChunk(BaseModel):
    type: Literal['audio_chunk'] = 'audio_chunk'
    assetId: str
    startMs: int
    endMs: int


class JobEventDone(BaseModel):
    type: Literal['done'] = 'done'


class JobEventError(BaseModel):
    type: Literal['error'] = 'error'
    message: str


class JobEventAlignmentProgress(BaseModel):
    type: Literal['alignment_progress'] = 'alignment_progress'
    progress: float
    message: str


class JobEventAlignmentDone(BaseModel):
    type: Literal['alignment_done'] = 'alignment_done'
    coverage: float


class JobEventAlignmentFailed(BaseModel):
    type: Literal['alignment_failed'] = 'alignment_failed'
    message: str


class JobEventLanguageRequired(BaseModel):
    type: Literal['language_required'] = 'language_required'
    candidates: list[LanguageCandidate]
    suggestedLanguage: str | None = None
    confidence: float = 0.0
    reason: str


class RuntimeStatusResponse(BaseModel):
    mlxCompatible: bool
    fallbackEnabled: bool
    runtimeVersion: str | None = None
    supportsQwen3Tts: bool
    minRequiredVersion: str
    reason: str
    alignmentRuntimeReady: bool = False
    alignmentModelsReady: bool = False
    alignmentReason: str | None = None
    alignmentProbeAt: str | None = None
    alignmentProbeError: str | None = None

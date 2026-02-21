export type ModelId = 'base' | 'customvoice' | 'voicedesign';
export type ModelCatalogId = ModelId | 'whisperx';
export type ModelSource = 'mlx' | 'official';
export type OutputFormat = 'mp3' | 'mp4';
export type JobState = 'queued' | 'running' | 'waiting_language' | 'done' | 'failed';
export type Mp4State = 'not_requested' | 'queued' | 'running' | 'done' | 'failed';
export type AlignmentState = 'queued' | 'running' | 'done' | 'failed';
export type VoiceType = 'clone' | 'design';
export type JobPhase =
  | 'queued'
  | 'model_check'
  | 'model_load_ram'
  | 'synthesizing'
  | 'aligning'
  | 'merge_export'
  | 'done'
  | 'failed';

export interface BoundingBox {
  x0: number;
  y0: number;
  x1: number;
  y1: number;
}

export interface WordTiming {
  word: string;
  startMs: number;
  endMs: number;
  confidence?: number | null;
  page: number | null;
  bbox: BoundingBox | null;
}

export interface Artifact {
  id: string;
  type: OutputFormat | 'alignment' | 'transcript' | 'audio_chunk';
  path: string;
  sizeBytes: number;
}

export interface JobStatus {
  id: string;
  state: JobState;
  progress: number;
  statusMessage: string | null;
  mp4State: Mp4State;
  mp4Error: string | null;
  alignmentState: AlignmentState;
  alignmentError: string | null;
  alignmentCoverage?: number | null;
  audioDurationMs?: number | null;
  createdAt: string;
  updatedAt: string;
  sourceType: 'text' | 'pdf';
  sourceLabel: string;
  model: ModelId;
  detectedLanguage: string;
  artifacts: Artifact[];
  warning: string | null;
  error: string | null;
  phase: JobPhase;
  phaseProgress: number | null;
  alignmentWarning: string | null;
  words: WordTiming[];
}

export interface VoiceItem {
  id: string;
  name: string;
  type: VoiceType;
  language?: string | null;
  referenceText?: string | null;
  description?: string | null;
  createdAt: string;
}

export interface AppConfig {
  locale: 'de' | 'en';
  outputDir: string;
  modelCacheDir: string;
  performanceProfile: 'standard' | 'memory';
  qualityPreset: 'speed' | 'balanced' | 'quality';
  allowFallback: boolean;
}

export interface ModelInfo {
  model: ModelCatalogId;
  source: ModelSource;
  repo: string;
  localPath: string;
  downloaded: boolean;
}

export interface JobRequestText {
  text: string;
  model: ModelId;
  voiceId?: string;
  language?: string;
  outputFormats: OutputFormat[];
}

export interface JobRequestPdf {
  pdfPath: string;
  model: ModelId;
  voiceId?: string;
  language?: string;
  pageRange?: { start: number; end: number };
  outputFormats: OutputFormat[];
}

export type JobEvent =
  | { type: 'progress'; progress: number; message: string; phase?: JobPhase | null; phaseProgress?: number | null }
  | { type: 'warning'; message: string }
  | { type: 'alignment_progress'; progress: number; message: string }
  | { type: 'alignment_done'; coverage: number }
  | { type: 'alignment_failed'; message: string }
  | { type: 'audio_chunk'; assetId: string; startMs: number; endMs: number }
  | { type: 'word'; payload: WordTiming }
  | {
      type: 'language_required';
      candidates: Array<{ code: string; confidence: number }>;
      suggestedLanguage: string | null;
      confidence: number;
      reason: string;
    }
  | { type: 'done' }
  | { type: 'error'; message: string };

export interface RuntimeStatus {
  mlxCompatible: boolean;
  fallbackEnabled: boolean;
  runtimeVersion: string | null;
  supportsQwen3Tts: boolean;
  minRequiredVersion: string;
  reason: string;
  alignmentRuntimeReady: boolean;
  alignmentModelsReady: boolean;
  alignmentReason?: string | null;
  alignmentProbeAt?: string | null;
  alignmentProbeError?: string | null;
}

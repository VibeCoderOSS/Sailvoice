import { create } from 'zustand';
import type { AppConfig, JobStatus, VoiceItem } from '../types/models';

type AudioQueueItem = {
  assetId: string;
  startMs: number;
  endMs: number;
};

type AppState = {
  serviceUrl: string;
  config: AppConfig | null;
  jobs: JobStatus[];
  voices: VoiceItem[];
  selectedPdfPath: string | null;
  selectedPdfPage: number;
  currentJobId: string | null;
  currentJobSelectionMode: 'auto' | 'manual';
  currentTimeMs: number;
  pendingSeekMs: number | null;
  isPlaying: boolean;
  audioQueue: AudioQueueItem[];
  setServiceUrl: (serviceUrl: string) => void;
  setConfig: (config: AppConfig) => void;
  setJobs: (jobs: JobStatus[]) => void;
  upsertJob: (job: JobStatus) => void;
  setVoices: (voices: VoiceItem[]) => void;
  setSelectedPdfPath: (path: string | null) => void;
  setSelectedPdfPage: (page: number) => void;
  setCurrentJobId: (jobId: string | null) => void;
  setCurrentJobIdManual: (jobId: string | null) => void;
  setCurrentTimeMs: (ms: number) => void;
  setPendingSeekMs: (ms: number | null) => void;
  setIsPlaying: (value: boolean) => void;
  enqueueAudioChunk: (item: AudioQueueItem) => void;
  clearAudioQueue: () => void;
  dequeueAudioChunk: () => AudioQueueItem | undefined;
};

export const useAppStore = create<AppState>((set, get) => ({
  serviceUrl: '',
  config: null,
  jobs: [],
  voices: [],
  selectedPdfPath: null,
  selectedPdfPage: 1,
  currentJobId: null,
  currentJobSelectionMode: 'auto',
  currentTimeMs: 0,
  pendingSeekMs: null,
  isPlaying: false,
  audioQueue: [],
  setServiceUrl: (serviceUrl) => set({ serviceUrl }),
  setConfig: (config) => set({ config }),
  setJobs: (jobs) => set({ jobs }),
  upsertJob: (job) =>
    set((state) => {
      const index = state.jobs.findIndex((item) => item.id === job.id);
      if (index === -1) {
        return { jobs: [job, ...state.jobs] };
      }
      const next = [...state.jobs];
      next[index] = job;
      return { jobs: next };
    }),
  setVoices: (voices) => set({ voices }),
  setSelectedPdfPath: (path) => set({ selectedPdfPath: path, selectedPdfPage: 1 }),
  setSelectedPdfPage: (page) => set({ selectedPdfPage: page }),
  setCurrentJobId: (jobId) => set({ currentJobId: jobId, currentJobSelectionMode: 'auto' }),
  setCurrentJobIdManual: (jobId) =>
    set({
      currentJobId: jobId,
      currentJobSelectionMode: jobId ? 'manual' : 'auto'
    }),
  setCurrentTimeMs: (ms) => set({ currentTimeMs: ms }),
  setPendingSeekMs: (ms) => set({ pendingSeekMs: ms }),
  setIsPlaying: (value) => set({ isPlaying: value }),
  enqueueAudioChunk: (item) =>
    set((state) => {
      const already = state.audioQueue.some((chunk) => chunk.assetId === item.assetId);
      if (already) {
        return state;
      }
      return { audioQueue: [...state.audioQueue, item].sort((a, b) => a.startMs - b.startMs) };
    }),
  clearAudioQueue: () => set({ audioQueue: [] }),
  dequeueAudioChunk: () => {
    const [first, ...rest] = get().audioQueue;
    set({ audioQueue: rest });
    return first;
  }
}));

import axios from 'axios';
import type {
  JobEvent,
  ModelCatalogId,
  ModelId,
  ModelInfo,
  ModelSource,
  JobRequestPdf,
  JobRequestText,
  JobStatus,
  RuntimeStatus,
  VoiceItem
} from '../types/models';

export const createClient = (baseUrl: string) => {
  return axios.create({
    baseURL: baseUrl,
    timeout: 120000
  });
};

export async function createTextJob(baseUrl: string, payload: JobRequestText) {
  const client = createClient(baseUrl);
  const { data } = await client.post<{ id: string }>('/v1/jobs/text', payload);
  return data;
}

export async function createPdfJob(baseUrl: string, payload: JobRequestPdf) {
  const client = createClient(baseUrl);
  const { data } = await client.post<{ id: string }>('/v1/jobs/pdf', payload);
  return data;
}

export async function getJob(baseUrl: string, jobId: string) {
  const client = createClient(baseUrl);
  const { data } = await client.get<JobStatus>(`/v1/jobs/${jobId}`);
  return data;
}

export async function listJobs(baseUrl: string) {
  const client = createClient(baseUrl);
  const { data } = await client.get<JobStatus[]>('/v1/jobs');
  return data;
}

export async function deleteJob(baseUrl: string, jobId: string) {
  const client = createClient(baseUrl);
  await client.delete(`/v1/jobs/${jobId}`);
}

export async function cancelJob(baseUrl: string, jobId: string) {
  const client = createClient(baseUrl);
  await client.post(`/v1/jobs/${jobId}/cancel`);
}

export async function listVoices(baseUrl: string) {
  const client = createClient(baseUrl);
  const { data } = await client.get<VoiceItem[]>('/v1/voices');
  return data;
}

export async function createVoice(baseUrl: string, payload: { name: string; language?: string; audioPath: string; referenceText?: string }) {
  const client = createClient(baseUrl);
  const { data } = await client.post<VoiceItem>('/v1/voices', payload);
  return data;
}

export async function createDesignVoice(baseUrl: string, payload: { name: string; language?: string; description: string }) {
  const client = createClient(baseUrl);
  const { data } = await client.post<VoiceItem>('/v1/voices/design', payload);
  return data;
}

export async function deleteVoice(baseUrl: string, voiceId: string) {
  const client = createClient(baseUrl);
  await client.delete(`/v1/voices/${voiceId}`);
}

export async function renameVoice(baseUrl: string, voiceId: string, name: string) {
  const client = createClient(baseUrl);
  const { data } = await client.patch<VoiceItem>(`/v1/voices/${voiceId}`, { name });
  return data;
}

export async function requestVoicePreview(baseUrl: string, payload: { voiceId?: string; speaker?: string; text: string; model: ModelId }) {
  const client = createClient(baseUrl);
  const { data } = await client.post<{ assetId: string }>('/v1/voices/preview', payload);
  return data;
}

export async function confirmJobLanguage(baseUrl: string, jobId: string, language: string) {
  const client = createClient(baseUrl);
  await client.post(`/v1/jobs/${jobId}/language`, { language });
}

export async function listModels(baseUrl: string) {
  const client = createClient(baseUrl);
  const { data } = await client.get<ModelInfo[]>('/v1/models');
  return data;
}

export async function getRuntimeStatus(baseUrl: string) {
  const client = createClient(baseUrl);
  const { data } = await client.get<RuntimeStatus>('/v1/runtime');
  return data;
}

export async function warmupRuntime(baseUrl: string) {
  const client = createClient(baseUrl);
  const { data } = await client.post<RuntimeStatus>('/v1/runtime/warmup');
  return data;
}

export async function downloadModel(baseUrl: string, payload: { model: ModelCatalogId; source: ModelSource }) {
  const client = createClient(baseUrl);
  const { data } = await client.post<ModelInfo & { warning?: string | null }>('/v1/models/download', payload, { timeout: 0 });
  return data;
}

export function subscribeJobEvents(
  baseUrl: string,
  jobId: string,
  onEvent: (event: JobEvent) => void,
  onError?: (event: Event) => void
) {
  const source = new EventSource(`${baseUrl}/v1/jobs/${jobId}/events`);

  source.onmessage = (message) => {
    try {
      const event = JSON.parse(message.data) as JobEvent;
      onEvent(event);
    } catch (error) {
      console.error('Invalid SSE message', error);
    }
  };

  source.onerror = (error) => {
    onError?.(error);
  };

  return () => source.close();
}

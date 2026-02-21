import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { confirmJobLanguage, createPdfJob, createTextJob, getJob, listVoices, subscribeJobEvents } from '../api/client';
import { useI18n } from '../i18n/I18nProvider';
import { useAppStore } from '../state/appStore';
import type { JobEvent, JobState, ModelId, OutputFormat, WordTiming } from '../types/models';

type InputMode = 'text' | 'pdf';
type LanguagePromptState = {
  jobId: string;
  candidates: Array<{ code: string; confidence: number }>;
  suggestedLanguage: string | null;
  confidence: number;
  reason: string;
  selectedLanguage: string;
};

function isActiveState(state: JobState) {
  return state === 'queued' || state === 'running' || state === 'waiting_language';
}

function isMp4Pending(state: string | null | undefined) {
  return state === 'queued' || state === 'running';
}

function pickPreferredJobId(
  jobs: Array<{
    id: string;
    state: JobState;
    updatedAt: string;
  }>
) {
  if (!jobs.length) {
    return null;
  }

  const stateRank = (state: JobState) => {
    if (state === 'running' || state === 'waiting_language') {
      return 0;
    }
    if (state === 'queued') {
      return 1;
    }
    return 2;
  };

  const sorted = [...jobs].sort((left, right) => {
    const rankDiff = stateRank(left.state) - stateRank(right.state);
    if (rankDiff !== 0) {
      return rankDiff;
    }
    return new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime();
  });

  return sorted[0]?.id ?? null;
}

function updateCurrentJobWord(word: WordTiming) {
  const { jobs, currentJobId, upsertJob } = useAppStore.getState();
  const current = jobs.find((job) => job.id === currentJobId);
  if (!current) {
    return;
  }
  const exists = current.words.some((item) => item.startMs === word.startMs && item.word === word.word);
  if (exists) {
    return;
  }
  upsertJob({
    ...current,
    words: [...current.words, word].sort((a, b) => a.startMs - b.startMs)
  });
}

export function StudioPage() {
  const { t } = useI18n();

  const serviceUrl = useAppStore((state) => state.serviceUrl);
  const selectedPdfPath = useAppStore((state) => state.selectedPdfPath);
  const voices = useAppStore((state) => state.voices);
  const jobs = useAppStore((state) => state.jobs);
  const currentJobId = useAppStore((state) => state.currentJobId);
  const currentJobSelectionMode = useAppStore((state) => state.currentJobSelectionMode);

  const setCurrentJobId = useAppStore((state) => state.setCurrentJobId);
  const setSelectedPdfPath = useAppStore((state) => state.setSelectedPdfPath);
  const setCurrentTimeMs = useAppStore((state) => state.setCurrentTimeMs);
  const clearAudioQueue = useAppStore((state) => state.clearAudioQueue);
  const enqueueAudioChunk = useAppStore((state) => state.enqueueAudioChunk);
  const upsertJob = useAppStore((state) => state.upsertJob);

  const [mode, setMode] = useState<InputMode>('text');
  const [text, setText] = useState(
    'Willkommen im lokalen Text to Speech Studio. Diese Stimme läuft vollständig auf deinem Mac. Es bleibt für immer gratis. Viel Spass!'
  );
  const [selectedVoiceId, setSelectedVoiceId] = useState<string>('');
  const [includeMp3, setIncludeMp3] = useState(true);
  const [includeMp4, setIncludeMp4] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [languagePrompt, setLanguagePrompt] = useState<LanguagePromptState | null>(null);

  const eventUnsubscribers = useRef<Map<string, () => void>>(new Map());

  const currentJob = useMemo(() => jobs.find((job) => job.id === currentJobId) ?? null, [jobs, currentJobId]);
  const selectedVoice = useMemo(() => voices.find((voice) => voice.id === selectedVoiceId) ?? null, [selectedVoiceId, voices]);
  const effectiveModel = useMemo<ModelId>(() => {
    if (selectedVoice?.type === 'design') {
      return 'voicedesign';
    }
    return 'base';
  }, [selectedVoice?.type]);

  const refreshJob = useCallback(
    async (jobId: string) => {
      if (!serviceUrl) {
        return;
      }
      const job = await getJob(serviceUrl, jobId);
      upsertJob(job);
    },
    [serviceUrl, upsertJob]
  );

  const handleJobEvent = useCallback(
    (jobId: string, event: JobEvent) => {
      const { jobs: allJobs } = useAppStore.getState();
      const current = allJobs.find((item) => item.id === jobId);
      if (!current) {
        return;
      }

      if (event.type === 'progress') {
        const nextState =
          current.state === 'done' ? 'done' : current.state === 'waiting_language' ? 'waiting_language' : 'running';
        const nextProgress = current.state === 'done' ? current.progress : event.progress;
        useAppStore.getState().upsertJob({
          ...current,
          state: nextState,
          progress: nextProgress,
          statusMessage: event.message,
          phase: event.phase || current.phase,
          phaseProgress: event.phaseProgress ?? current.phaseProgress
        });
      }

      if (event.type === 'warning') {
        useAppStore.getState().upsertJob({
          ...current,
          warning: event.message
        });
      }

      if (event.type === 'alignment_progress') {
        useAppStore.getState().upsertJob({
          ...current,
          alignmentState: 'running',
          statusMessage: event.message
        });
      }

      if (event.type === 'alignment_done') {
        useAppStore.getState().upsertJob({
          ...current,
          alignmentState: current.state === 'done' ? 'done' : 'running',
          alignmentCoverage: event.coverage
        });
      }

      if (event.type === 'alignment_failed') {
        useAppStore.getState().upsertJob({
          ...current,
          alignmentState: 'failed',
          alignmentError: event.message,
          error: event.message,
          state: 'failed',
          statusMessage: event.message
        });
      }

      if (event.type === 'audio_chunk') {
        if (useAppStore.getState().currentJobId === jobId) {
          enqueueAudioChunk({
            assetId: event.assetId,
            startMs: event.startMs,
            endMs: event.endMs
          });
        }
      }

      if (event.type === 'word') {
        if (useAppStore.getState().currentJobId === jobId) {
          updateCurrentJobWord(event.payload);
        }
      }

      if (event.type === 'language_required') {
        const suggested = event.suggestedLanguage || event.candidates[0]?.code || '';
        setLanguagePrompt({
          jobId,
          candidates: event.candidates,
          suggestedLanguage: event.suggestedLanguage,
          confidence: event.confidence,
          reason: event.reason,
          selectedLanguage: suggested
        });
        useAppStore.getState().upsertJob({
          ...current,
          state: 'waiting_language',
          detectedLanguage: suggested || 'auto',
          statusMessage: 'waiting_language_confirmation'
        });
      }

      if (event.type === 'done') {
        refreshJob(jobId).catch(console.error);
        const unsubscribe = eventUnsubscribers.current.get(jobId);
        if (unsubscribe) {
          unsubscribe();
          eventUnsubscribers.current.delete(jobId);
        }
        setLanguagePrompt((currentPrompt) => (currentPrompt?.jobId === jobId ? null : currentPrompt));
      }

      if (event.type === 'error') {
        useAppStore.getState().upsertJob({
          ...current,
          state: 'failed',
          error: event.message,
          statusMessage: event.message
        });
        const unsubscribe = eventUnsubscribers.current.get(jobId);
        if (unsubscribe) {
          unsubscribe();
          eventUnsubscribers.current.delete(jobId);
        }
        setLanguagePrompt((currentPrompt) => (currentPrompt?.jobId === jobId ? null : currentPrompt));
      }
    },
    [enqueueAudioChunk, refreshJob]
  );

  const attachStream = useCallback(
    (jobId: string) => {
      if (!serviceUrl || eventUnsubscribers.current.has(jobId)) {
        return;
      }

      const unsubscribe = subscribeJobEvents(
        serviceUrl,
        jobId,
        (event) => handleJobEvent(jobId, event),
        () => {
          refreshJob(jobId).catch(console.error);
        }
      );

      eventUnsubscribers.current.set(jobId, unsubscribe);
    },
    [handleJobEvent, refreshJob, serviceUrl]
  );

  const submit = async () => {
    setError(null);
    if (!serviceUrl) {
      setError('Backend not ready');
      return;
    }

    if (mode === 'text' && !text.trim()) {
      setError('Please provide text input');
      return;
    }

    if (mode === 'pdf' && !selectedPdfPath) {
      setError('Please select a PDF');
      return;
    }

    const formats: OutputFormat[] = [];
    if (includeMp3) {
      formats.push('mp3');
    }
    if (includeMp4) {
      formats.push('mp4');
    }
    if (!formats.length) {
      setError(t('chooseAtLeastOneFormat'));
      return;
    }
    setIsSubmitting(true);

    try {
      clearAudioQueue();
      setCurrentTimeMs(0);

      const selectedVoiceIdPayload = selectedVoiceId || undefined;
      const response =
        mode === 'text'
          ? await createTextJob(serviceUrl, {
              text,
              model: effectiveModel,
              voiceId: selectedVoiceIdPayload || undefined,
              language: selectedVoice?.language || undefined,
              outputFormats: formats
            })
          : await createPdfJob(serviceUrl, {
              pdfPath: selectedPdfPath!,
              model: effectiveModel,
              voiceId: selectedVoiceIdPayload || undefined,
              language: selectedVoice?.language || undefined,
              outputFormats: formats
            });

      await refreshJob(response.id);
      setCurrentJobId(response.id);
      attachStream(response.id);
    } catch (submitError) {
      console.error(submitError);
      setError((submitError as Error).message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const pickPdf = async () => {
    const path = await window.desktopApi.pickPdf();
    if (path) {
      setSelectedPdfPath(path);
      setMode('pdf');
    }
  };

  const confirmLanguageSelection = async () => {
    if (!languagePrompt || !serviceUrl || !languagePrompt.selectedLanguage.trim()) {
      return;
    }
    try {
      await confirmJobLanguage(serviceUrl, languagePrompt.jobId, languagePrompt.selectedLanguage.trim());
      setLanguagePrompt(null);
      await refreshJob(languagePrompt.jobId);
    } catch (confirmError) {
      console.error(confirmError);
      setError((confirmError as Error).message);
    }
  };

  useEffect(() => {
    const exists = voices.some((voice) => voice.id === selectedVoiceId);
    if (selectedVoiceId && !exists) {
      setSelectedVoiceId('');
    }
  }, [selectedVoiceId, voices]);

  useEffect(() => {
    if (!serviceUrl) {
      return;
    }
    listVoices(serviceUrl)
      .then((items) => useAppStore.getState().setVoices(items))
      .catch(console.error);
  }, [serviceUrl]);

  useEffect(() => {
    const selectedExists = currentJobId ? jobs.some((job) => job.id === currentJobId) : false;
    if (currentJobSelectionMode === 'manual' && selectedExists) {
      return;
    }
    const preferred = pickPreferredJobId(jobs);
    if (preferred !== currentJobId) {
      setCurrentJobId(preferred);
    }
  }, [currentJobId, currentJobSelectionMode, jobs, setCurrentJobId]);

  useEffect(() => {
    const activeJobIds = new Set(jobs.filter((job) => isActiveState(job.state)).map((job) => job.id));
    for (const jobId of activeJobIds) {
      attachStream(jobId);
    }
    for (const [jobId, unsubscribe] of eventUnsubscribers.current.entries()) {
      if (!activeJobIds.has(jobId)) {
        unsubscribe();
        eventUnsubscribers.current.delete(jobId);
      }
    }
  }, [attachStream, jobs]);

  useEffect(() => {
    if (!currentJobId) {
      return;
    }
    const selected = jobs.find((job) => job.id === currentJobId);
    if (!selected || (!isActiveState(selected.state) && !isMp4Pending(selected.mp4State))) {
      return;
    }

    const timer = window.setInterval(() => {
      refreshJob(currentJobId).catch(console.error);
    }, 5000);

    return () => window.clearInterval(timer);
  }, [currentJobId, jobs, refreshJob]);

  useEffect(() => {
    return () => {
      for (const unsubscribe of eventUnsubscribers.current.values()) {
        unsubscribe();
      }
      eventUnsubscribers.current.clear();
    };
  }, []);

  return (
    <section className="page">
      <h2 className="page-title">{t('navStudio')}</h2>

      <div className="grid-two">
        <article className="panel">
          <div className="field">
            <label>{t('inputMode')}</label>
            <div className="row">
              <button className={`btn ${mode === 'text' ? 'btn-primary' : ''}`} type="button" onClick={() => setMode('text')}>
                {t('freeText')}
              </button>
              <button className={`btn ${mode === 'pdf' ? 'btn-primary' : ''}`} type="button" onClick={() => setMode('pdf')}>
                {t('pdf')}
              </button>
              <button className="btn" type="button" onClick={pickPdf}>
                {t('selectPdf')}
              </button>
            </div>
          </div>

          {mode === 'text' ? (
            <div className="field">
              <label>{t('freeText')}</label>
              <textarea className="textarea" value={text} onChange={(event) => setText(event.target.value)} placeholder={t('textPlaceholder')} />
            </div>
          ) : (
            <div className="panel panel-muted">
              <p className="kv">{selectedPdfPath ?? t('noPdf')}</p>
            </div>
          )}

          <div className="row">
            <button className="btn btn-primary" type="button" onClick={submit} disabled={isSubmitting}>
              {isSubmitting ? '...' : t('generate')}
            </button>
            {error ? <span className="error">{error}</span> : null}
          </div>
        </article>

        <article className="panel">
          <div className="field">
            <label>{t('voice')}</label>
            <select className="select" value={selectedVoiceId} onChange={(event) => setSelectedVoiceId(event.target.value)}>
              <option value="">{`${t('standardVoice')} (Base)`}</option>
              {voices.map((voice) => (
                <option key={voice.id} value={voice.id}>
                  {voice.name} {voice.type === 'design' ? '(Design)' : '(Clone)'}
                </option>
              ))}
            </select>
          </div>

          <div className="field">
            <label>{t('model')}</label>
            <div className="panel panel-muted">
              <p className="kv">
                {effectiveModel === 'voicedesign' ? 'VoiceDesign' : 'Base'} ({t('auto')})
              </p>
              {selectedVoice?.language ? (
                <p className="kv">
                  {t('language')}: {selectedVoice.language}
                </p>
              ) : null}
            </div>
          </div>

          <div className="field">
            <label>{t('downloadOptions')}</label>
            <div className="row">
              <label className="kv">
                <input type="checkbox" checked={includeMp3} onChange={(event) => setIncludeMp3(event.target.checked)} /> {t('includeMp3')}
              </label>
              <label className="kv">
                <input type="checkbox" checked={includeMp4} onChange={(event) => setIncludeMp4(event.target.checked)} /> {t('includeMp4')}
              </label>
            </div>
          </div>

          <div className="panel panel-muted panel-status">
            <p className="kv">
              {t('queueState')}: {currentJob?.state ?? t('noActiveJob')}
            </p>
            <p className="kv">
              {t('mp4State')}: {currentJob?.mp4State ?? '-'}
            </p>
            <p className="kv">
              {t('alignmentState')}: {currentJob?.alignmentState ?? '-'}
            </p>
            <p className="kv">
              {t('progress')}: {Math.round((currentJob?.progress ?? 0) * 100)}%
            </p>
            <p className="kv">
              {t('phase')}: {currentJob?.phase ?? '-'}
            </p>
            <p className="kv">
              {t('phaseProgress')}: {currentJob?.phaseProgress !== null && currentJob?.phaseProgress !== undefined ? `${Math.round(currentJob.phaseProgress * 100)}%` : '-'}
            </p>
            <p className="kv">
              {t('detectedLanguage')}: {currentJob?.detectedLanguage || t('auto')}
            </p>
            <p className="kv">
              Alignment Coverage:{' '}
              {currentJob?.alignmentCoverage !== undefined && currentJob?.alignmentCoverage !== null
                ? `${Math.round(currentJob.alignmentCoverage * 100)}%`
                : '-'}
            </p>
            <p className="kv">
              {t('statusInfo')}: {currentJob?.statusMessage || '-'}
            </p>
            <div className="status-progress" style={{ marginTop: 8 }}>
              <span style={{ width: currentJob ? `${Math.max(3, currentJob.progress * 100)}%` : '0%' }} />
            </div>
            {currentJob?.mp4Error ? <p className="error">{currentJob.mp4Error}</p> : null}
            {currentJob?.alignmentError ? <p className="error">{currentJob.alignmentError}</p> : null}
            {currentJob?.alignmentWarning ? <p className="warning">{currentJob.alignmentWarning}</p> : null}
            {currentJob?.warning ? <p className="warning">{currentJob.warning}</p> : null}
            {currentJob?.error ? <p className="error">{currentJob.error}</p> : null}
          </div>
        </article>
      </div>

      {languagePrompt ? (
        <div className="language-modal-backdrop">
          <article className="language-modal">
            <h3>{t('languageConfirmTitle')}</h3>
            <p className="kv">
              {t('languageConfirmHint')} ({t('languageConfidence')}: {Math.round(languagePrompt.confidence * 100)}%)
            </p>
            <div className="field">
              <label>{t('language')}</label>
              <select
                className="select"
                value={languagePrompt.selectedLanguage}
                onChange={(event) =>
                  setLanguagePrompt((current) =>
                    current
                      ? {
                          ...current,
                          selectedLanguage: event.target.value
                        }
                      : current
                  )
                }
              >
                {languagePrompt.candidates.map((candidate) => (
                  <option key={candidate.code} value={candidate.code}>
                    {candidate.code} ({Math.round(candidate.confidence * 100)}%)
                  </option>
                ))}
              </select>
            </div>
            <div className="row">
              <button className="btn btn-primary" type="button" onClick={confirmLanguageSelection}>
                {t('confirm')}
              </button>
            </div>
          </article>
        </div>
      ) : null}
    </section>
  );
}

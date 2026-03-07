import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { cancelJob, confirmJobLanguage, createPdfJob, createTextJob, getJob, listVoices, subscribeJobEvents } from '../api/client';
import { useI18n } from '../i18n/I18nProvider';
import { useAppStore } from '../state/appStore';
import type { JobEvent, JobState, ModelId, OutputFormat, WordTiming } from '../types/models';

type InputMode = 'text' | 'pdf';
const STANDARD_PRESET_SPEAKERS = ['serena', 'vivian', 'ryan', 'aiden'] as const;
const DEFAULT_VOICE_SELECTION = `preset:${STANDARD_PRESET_SPEAKERS[0]}`;
const LANGUAGE_OPTIONS = [
  { value: 'auto', labelDe: 'Auto erkennen', labelEn: 'Auto detect' },
  { value: 'de', labelDe: 'Deutsch', labelEn: 'German' },
  { value: 'en', labelDe: 'Englisch', labelEn: 'English' },
  { value: 'fr', labelDe: 'Französisch', labelEn: 'French' },
  { value: 'it', labelDe: 'Italienisch', labelEn: 'Italian' },
  { value: 'es', labelDe: 'Spanisch', labelEn: 'Spanish' },
] as const;

function formatSpeakerLabel(value: string) {
  if (!value) {
    return value;
  }
  return value.charAt(0).toUpperCase() + value.slice(1);
}

type LanguagePromptState = {
  jobId: string;
  candidates: Array<{ code: string; confidence: number }>;
  suggestedLanguage: string | null;
  confidence: number;
  reason: string;
  selectedLanguage: string;
};

function isActiveState(state: JobState) {
  return state === 'queued' || state === 'running' || state === 'waiting_language' || state === 'canceling';
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
  const activeJobs = jobs.filter(
    (job) => job.state === 'queued' || job.state === 'running' || job.state === 'waiting_language' || job.state === 'canceling'
  );
  if (!activeJobs.length) {
    return null;
  }

  const stateRank = (state: JobState) => {
    if (state === 'running' || state === 'waiting_language' || state === 'canceling') {
      return 0;
    }
    if (state === 'queued') {
      return 1;
    }
    return 2;
  };

  const sorted = [...activeJobs].sort((left, right) => {
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
  const { t, locale } = useI18n();
  const localizedDefaultText = t('studioDefaultText');

  const serviceUrl = useAppStore((state) => state.serviceUrl);
  const config = useAppStore((state) => state.config);
  const selectedPdfPath = useAppStore((state) => state.selectedPdfPath);
  const voices = useAppStore((state) => state.voices);
  const jobs = useAppStore((state) => state.jobs);
  const currentJobId = useAppStore((state) => state.currentJobId);
  const currentJobSelectionMode = useAppStore((state) => state.currentJobSelectionMode);
  const activePlaybackJobId = useAppStore((state) => state.activePlaybackJobId);
  const isPlaying = useAppStore((state) => state.isPlaying);

  const setCurrentJobId = useAppStore((state) => state.setCurrentJobId);
  const setCurrentJobIdManual = useAppStore((state) => state.setCurrentJobIdManual);
  const setActivePlaybackJobId = useAppStore((state) => state.setActivePlaybackJobId);
  const setSelectedPdfPath = useAppStore((state) => state.setSelectedPdfPath);
  const setCurrentTimeMs = useAppStore((state) => state.setCurrentTimeMs);
  const clearAudioQueue = useAppStore((state) => state.clearAudioQueue);
  const enqueueAudioChunk = useAppStore((state) => state.enqueueAudioChunk);
  const upsertJob = useAppStore((state) => state.upsertJob);

  const [mode, setMode] = useState<InputMode>('text');
  const [text, setText] = useState(localizedDefaultText);
  const [selectedVoiceSelection, setSelectedVoiceSelection] = useState<string>(DEFAULT_VOICE_SELECTION);
  const [selectedLanguage, setSelectedLanguage] = useState<string>('auto');
  const [includeMp3, setIncludeMp3] = useState(true);
  const [includeMp4, setIncludeMp4] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [playbackHint, setPlaybackHint] = useState<string | null>(null);
  const [languagePrompt, setLanguagePrompt] = useState<LanguagePromptState | null>(null);
  const lastLocalizedDefaultRef = useRef(localizedDefaultText);
  const lastEventAtRef = useRef<Map<string, number>>(new Map());
  const [cancelingJobId, setCancelingJobId] = useState<string | null>(null);

  const eventUnsubscribers = useRef<Map<string, () => void>>(new Map());

  const currentJob = useMemo(() => jobs.find((job) => job.id === currentJobId) ?? null, [jobs, currentJobId]);
  const selectedVoice = useMemo(() => {
    if (!selectedVoiceSelection.startsWith('voice:')) {
      return null;
    }
    const voiceId = selectedVoiceSelection.slice('voice:'.length);
    return voices.find((voice) => voice.id === voiceId) ?? null;
  }, [selectedVoiceSelection, voices]);
  const selectedSpeakerPreset = useMemo(() => {
    if (!selectedVoiceSelection.startsWith('preset:')) {
      return null;
    }
    return selectedVoiceSelection.slice('preset:'.length) || null;
  }, [selectedVoiceSelection]);
  const effectiveModel = useMemo<ModelId>(() => {
    if (selectedVoice?.type === 'design') {
      return 'voicedesign';
    }
    if (selectedVoice?.type === 'clone') {
      return 'base';
    }
    return 'customvoice';
  }, [selectedVoice?.type]);

  const refreshJob = useCallback(
    async (jobId: string) => {
      if (!serviceUrl) {
        return null;
      }
      const job = await getJob(serviceUrl, jobId);
      upsertJob(job);
      lastEventAtRef.current.set(jobId, Date.now());
      return job;
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
      lastEventAtRef.current.set(jobId, Date.now());

      if (event.type === 'progress') {
        const nextState =
          current.state === 'done'
            ? 'done'
            : current.state === 'waiting_language'
              ? 'waiting_language'
              : current.state === 'canceling'
                ? 'canceling'
                : current.state === 'canceled'
                  ? 'canceled'
                  : 'running';
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

      if (event.type === 'mp4_background_progress') {
        useAppStore.getState().upsertJob({
          ...current,
          mp4State: 'running',
          mp4Progress: event.progress,
          statusMessage: event.message
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
        if (useAppStore.getState().activePlaybackJobId === jobId) {
          enqueueAudioChunk({
            jobId,
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
        setCancelingJobId((value) => (value === jobId ? null : value));
        refreshJob(jobId).catch(console.error);
        const unsubscribe = eventUnsubscribers.current.get(jobId);
        if (unsubscribe) {
          unsubscribe();
          eventUnsubscribers.current.delete(jobId);
        }
        setLanguagePrompt((currentPrompt) => (currentPrompt?.jobId === jobId ? null : currentPrompt));
      }

      if (event.type === 'canceled') {
        setCancelingJobId((value) => (value === jobId ? null : value));
        useAppStore.getState().upsertJob({
          ...current,
          state: 'canceled',
          cancelReason: event.message,
          statusMessage: event.message,
          phase: 'canceled',
          phaseProgress: 1
        });
        const unsubscribe = eventUnsubscribers.current.get(jobId);
        if (unsubscribe) {
          unsubscribe();
          eventUnsubscribers.current.delete(jobId);
        }
        setLanguagePrompt((currentPrompt) => (currentPrompt?.jobId === jobId ? null : currentPrompt));
      }

      if (event.type === 'error') {
        setCancelingJobId((value) => (value === jobId ? null : value));
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
          lastEventAtRef.current.set(jobId, 0);
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
      const keepCurrentPlayback = Boolean(isPlaying && activePlaybackJobId);
      setPlaybackHint(null);
      if (!keepCurrentPlayback) {
        clearAudioQueue();
        setCurrentTimeMs(0);
        setActivePlaybackJobId(null);
      }

      const selectedVoiceIdPayload =
        selectedVoice && selectedVoiceSelection.startsWith('voice:') ? selectedVoice.id : undefined;
      const selectedSpeakerPayload =
        !selectedVoice && selectedSpeakerPreset ? selectedSpeakerPreset : undefined;
      const requestedLanguage = selectedLanguage !== 'auto' ? selectedLanguage : selectedVoice?.language || undefined;
      const response =
        mode === 'text'
          ? await createTextJob(serviceUrl, {
              text,
              model: effectiveModel,
              voiceId: selectedVoiceIdPayload || undefined,
              speaker: selectedSpeakerPayload || undefined,
              language: requestedLanguage,
              outputFormats: formats
            })
          : await createPdfJob(serviceUrl, {
              pdfPath: selectedPdfPath!,
              model: effectiveModel,
              voiceId: selectedVoiceIdPayload || undefined,
              speaker: selectedSpeakerPayload || undefined,
              language: requestedLanguage,
              outputFormats: formats
            });

      await refreshJob(response.id);
      setCurrentJobIdManual(response.id);
      if (!keepCurrentPlayback) {
        setActivePlaybackJobId(response.id);
      } else if (activePlaybackJobId !== response.id) {
        setPlaybackHint(t('playbackContinuesHint'));
      }
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
    if (!selectedVoiceSelection.startsWith('voice:')) {
      return;
    }
    const selectedId = selectedVoiceSelection.slice('voice:'.length);
    const exists = voices.some((voice) => voice.id === selectedId);
    if (!exists) {
      setSelectedVoiceSelection(DEFAULT_VOICE_SELECTION);
    }
  }, [selectedVoiceSelection, voices]);

  useEffect(() => {
    setText((current) => {
      const previousLocalizedDefault = lastLocalizedDefaultRef.current;
      if (current !== previousLocalizedDefault && current.trim() !== '') {
        return current;
      }
      return localizedDefaultText;
    });
    lastLocalizedDefaultRef.current = localizedDefaultText;
  }, [localizedDefaultText]);

  useEffect(() => {
    if (!config) {
      return;
    }
    setIncludeMp3(config.defaultIncludeMp3 !== false);
    setIncludeMp4(config.defaultIncludeMp4 === true);
  }, [config?.defaultIncludeMp3, config?.defaultIncludeMp4]);

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
    const selectedJob = selectedExists ? jobs.find((job) => job.id === currentJobId) ?? null : null;
    const selectedIsActive = selectedJob ? isActiveState(selectedJob.state) || isMp4Pending(selectedJob.mp4State) : false;
    if (currentJobSelectionMode === 'manual' && selectedExists && selectedIsActive) {
      return;
    }
    if (isPlaying && selectedExists) {
      return;
    }
    const preferred = pickPreferredJobId(jobs);
    if (preferred !== currentJobId) {
      setCurrentJobId(preferred);
    }
  }, [currentJobId, currentJobSelectionMode, isPlaying, jobs, setCurrentJobId]);

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
      const lastEventAt = lastEventAtRef.current.get(currentJobId) ?? 0;
      if (Date.now() - lastEventAt < 15000) {
        return;
      }
      refreshJob(currentJobId).catch(console.error);
    }, 15000);

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

  const requestCancel = useCallback(async () => {
    if (!serviceUrl || !currentJobId) {
      return;
    }
    setError(null);
    setCancelingJobId(currentJobId);
    try {
      await cancelJob(serviceUrl, currentJobId);
      const refreshed = await refreshJob(currentJobId);
      if (refreshed && !isActiveState(refreshed.state) && !isMp4Pending(refreshed.mp4State)) {
        setCancelingJobId(null);
      }
    } catch (cancelError) {
      console.error(cancelError);
      setError((cancelError as Error).message || t('cancelJobFailed'));
      setCancelingJobId(null);
    }
  }, [currentJobId, refreshJob, serviceUrl, t]);

  const currentJobActive = Boolean(currentJob && (isActiveState(currentJob.state) || isMp4Pending(currentJob.mp4State)));
  const currentJobProgress = currentJob
    ? isMp4Pending(currentJob.mp4State)
      ? (currentJob.mp4Progress ?? 0)
      : currentJob.progress
    : 0;
  const currentJobProgressWidth = currentJobProgress > 0 ? Math.max(3, currentJobProgress * 100) : 0;
  const currentJobLabel = currentJob
    ? isMp4Pending(currentJob.mp4State)
      ? `MP4 · ${currentJob.mp4State}`
      : currentJob.phase
        ? `${currentJob.state} · ${currentJob.phase}`
        : currentJob.state
    : null;

  return (
    <section className="page page-studio">
      <header className="page-header">
        <div className="page-header-copy">
          <span className="page-kicker">{t('navStudio')}</span>
          <h2 className="page-title">{t('navStudio')}</h2>
        </div>
      </header>

      <div className="studio-layout">
        <article className="panel panel-hero panel-studio">
          <div className="panel-heading">
            <div>
              <span className="section-eyebrow">{t('inputMode')}</span>
              <h3 className="section-title">{t('composeSection')}</h3>
            </div>
          </div>

          <div className="field">
            <div className="toggle-group">
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
              <textarea className="textarea textarea-hero" value={text} onChange={(event) => setText(event.target.value)} placeholder={t('textPlaceholder')} />
            </div>
          ) : (
            <div className="source-summary-card">
              <div>
                <span className="section-eyebrow">{t('pdf')}</span>
                <strong>{selectedPdfPath ? selectedPdfPath.split('/').pop() : t('noPdf')}</strong>
              </div>
              <p className="kv">{selectedPdfPath ?? t('noPdf')}</p>
            </div>
          )}

          <div className="studio-config-grid">
            <section className="subsection-card">
              <div className="field">
                <label>{t('voice')}</label>
                <select
                  className="select"
                  value={selectedVoiceSelection}
                  onChange={(event) => setSelectedVoiceSelection(event.target.value)}
                >
                  <optgroup label={`${t('standardVoice')} (${t('auto')})`}>
                    {STANDARD_PRESET_SPEAKERS.map((speaker) => (
                      <option key={speaker} value={`preset:${speaker}`}>
                        {formatSpeakerLabel(speaker)}
                      </option>
                    ))}
                  </optgroup>
                  {voices.length ? (
                    <optgroup label={t('navVoices')}>
                      {voices.map((voice) => (
                        <option key={voice.id} value={`voice:${voice.id}`}>
                          {voice.name} {voice.type === 'design' ? '(Design)' : '(Clone)'}
                        </option>
                      ))}
                    </optgroup>
                  ) : null}
                </select>
              </div>
              <div className="field">
                <label>{t('language')}</label>
                <select className="select" value={selectedLanguage} onChange={(event) => setSelectedLanguage(event.target.value)}>
                  {LANGUAGE_OPTIONS.map((option) => (
                    <option key={option.value} value={option.value}>
                      {locale === 'de' ? option.labelDe : option.labelEn}
                    </option>
                  ))}
                </select>
              </div>
              <div className="detail-grid">
                <div className="info-tile">
                  <span className="tile-label">{t('model')}</span>
                  <strong>{effectiveModel === 'voicedesign' ? 'VoiceDesign' : effectiveModel === 'base' ? 'Base' : 'CustomVoice'}</strong>
                </div>
                <div className="info-tile">
                  <span className="tile-label">{t('voice')}</span>
                  <strong>{selectedSpeakerPreset ? formatSpeakerLabel(selectedSpeakerPreset) : selectedVoice?.name ?? '-'}</strong>
                </div>
              </div>
            </section>

            <section className="subsection-card">
              <div className="field">
                <label>{t('downloadOptions')}</label>
                <div className="choice-stack">
                  <label className="choice-row">
                    <input type="checkbox" checked={includeMp3} onChange={(event) => setIncludeMp3(event.target.checked)} />
                    <span className="choice-row-label">{t('includeMp3')}</span>
                  </label>
                  <label className="choice-row">
                    <input type="checkbox" checked={includeMp4} onChange={(event) => setIncludeMp4(event.target.checked)} />
                    <span className="choice-row-label">{t('includeMp4')}</span>
                  </label>
                </div>
              </div>
            </section>
          </div>

          <div className="composer-footer">
            <button className="btn btn-primary btn-large" type="button" onClick={submit} disabled={isSubmitting}>
              {isSubmitting ? '...' : t('generate')}
            </button>
            <div className="inline-feedback">
              {error ? <span className="error">{error}</span> : null}
              {playbackHint ? <span className="warning">{playbackHint}</span> : null}
            </div>
          </div>

          {currentJob ? (
            <div className="studio-job-strip">
              <div className="studio-job-strip-head">
                <span className="kv">{currentJobLabel}</span>
                <div className="studio-job-strip-actions">
                  <span className="studio-job-strip-progress-value">{Math.round(currentJobProgress * 100)}%</span>
                  {currentJobActive ? (
                    <button
                      className="btn btn-danger btn-compact"
                      type="button"
                      onClick={requestCancel}
                      disabled={cancelingJobId === currentJob.id}
                    >
                      {cancelingJobId === currentJob.id ? t('canceling') : t('cancelJob')}
                    </button>
                  ) : null}
                </div>
              </div>

              <div className="status-progress studio-job-progress">
                <span style={{ width: `${currentJobProgressWidth}%` }} />
              </div>

              {currentJob?.cancelReason ? <p className="kv">{t('cancelReasonLabel')}: {currentJob.cancelReason}</p> : null}
              {currentJob?.mp4Error ? <p className="error">{currentJob.mp4Error}</p> : null}
              {currentJob?.alignmentError ? <p className="error">{currentJob.alignmentError}</p> : null}
              {currentJob?.warning ? <p className="warning">{currentJob.warning}</p> : null}
              {currentJob?.error ? <p className="error">{currentJob.error}</p> : null}
            </div>
          ) : null}
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

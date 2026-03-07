import { useEffect, useMemo, useState } from 'react';
import { cancelJob, deleteJob, listJobs } from '../api/client';
import { Mp4PreviewDock } from '../components/Mp4PreviewDock';
import { useI18n } from '../i18n/I18nProvider';
import { useAppStore } from '../state/appStore';
import type { Artifact } from '../types/models';

function getArtifact(jobArtifacts: Artifact[], type: 'mp3' | 'mp4') {
  return jobArtifacts.find((artifact) => artifact.type === type) ?? null;
}

function toneForState(state: string) {
  if (state === 'done') {
    return 'tone-success';
  }
  if (state === 'failed') {
    return 'tone-danger';
  }
  if (state === 'canceled') {
    return 'tone-muted';
  }
  if (state === 'running' || state === 'waiting_language' || state === 'canceling' || state === 'queued') {
    return 'tone-info';
  }
  return 'tone-neutral';
}

const PAGE_SIZE = 4;

export function ExportsPage() {
  const { t } = useI18n();
  const serviceUrl = useAppStore((state) => state.serviceUrl);
  const jobs = useAppStore((state) => state.jobs);
  const config = useAppStore((state) => state.config);
  const setJobs = useAppStore((state) => state.setJobs);
  const removeJobFromStore = useAppStore((state) => state.removeJob);
  const setCurrentJobIdManual = useAppStore((state) => state.setCurrentJobIdManual);
  const [deletingJobId, setDeletingJobId] = useState<string | null>(null);
  const [cancelingJobId, setCancelingJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isVisible, setIsVisible] = useState(() => document.visibilityState === 'visible');
  const [page, setPage] = useState(1);
  const [preview, setPreview] = useState<{
    assetId: string;
    label: string;
  } | null>(null);

  useEffect(() => {
    if (!serviceUrl) {
      return;
    }
    listJobs(serviceUrl)
      .then((next) => setJobs(next))
      .catch(console.error);
  }, [serviceUrl, setJobs]);

  useEffect(() => {
    const handleVisibilityChange = () => setIsVisible(document.visibilityState === 'visible');
    document.addEventListener('visibilitychange', handleVisibilityChange);
    return () => document.removeEventListener('visibilitychange', handleVisibilityChange);
  }, []);

  useEffect(() => {
    if (!serviceUrl) {
      return;
    }
    const hasMp4Background = jobs.some((job) => job.mp4State === 'queued' || job.mp4State === 'running');
    if (!hasMp4Background || !isVisible) {
      return;
    }
    const timer = window.setInterval(() => {
      listJobs(serviceUrl)
        .then((next) => setJobs(next))
        .catch(console.error);
    }, 8000);
    return () => window.clearInterval(timer);
  }, [isVisible, jobs, serviceUrl, setJobs]);

  const sortedJobs = useMemo(() => {
    const rank = (jobState: string, mp4State: string) => {
      if (jobState === 'running' || jobState === 'waiting_language' || jobState === 'canceling') {
        return 0;
      }
      if (jobState === 'queued') {
        return 1;
      }
      if (mp4State === 'running' || mp4State === 'queued') {
        return 2;
      }
      if (jobState === 'failed') {
        return 4;
      }
      if (jobState === 'canceled') {
        return 5;
      }
      return 3;
    };

    return [...jobs].sort((left, right) => {
      const rankDiff = rank(left.state, left.mp4State) - rank(right.state, right.mp4State);
      if (rankDiff !== 0) {
        return rankDiff;
      }
      return new Date(right.updatedAt).getTime() - new Date(left.updatedAt).getTime();
    });
  }, [jobs]);

  const totalPages = Math.max(1, Math.ceil(sortedJobs.length / PAGE_SIZE));
  const currentPage = Math.min(page, totalPages);
  const pageStart = (currentPage - 1) * PAGE_SIZE;
  const visibleJobs = sortedJobs.slice(pageStart, pageStart + PAGE_SIZE);

  useEffect(() => {
    if (page > totalPages) {
      setPage(totalPages);
    }
  }, [page, totalPages]);

  const openOutputFolder = async () => {
    if (!config?.outputDir) {
      return;
    }
    await window.desktopApi.openPath(config.outputDir);
  };

  const downloadAsset = (assetId: string, defaultName: string) => {
    const link = document.createElement('a');
    link.href = `${serviceUrl}/v1/assets/${assetId}`;
    link.download = defaultName;
    link.click();
  };

  const removeJob = async (jobId: string) => {
    setError(null);
    if (!window.confirm(t('deleteOutputConfirm'))) {
      return;
    }

    const targetJob = jobs.find((job) => job.id === jobId) ?? null;
    if (preview && targetJob?.artifacts.some((artifact) => artifact.id === preview.assetId)) {
      setPreview(null);
    }

    setDeletingJobId(jobId);
    removeJobFromStore(jobId);
    try {
      await deleteJob(serviceUrl, jobId);
    } catch (deleteError) {
      const next = await listJobs(serviceUrl);
      setJobs(next);
      if (next.some((job) => job.id === jobId)) {
        setError((deleteError as Error).message);
      }
    } finally {
      setDeletingJobId(null);
    }
  };

  const requestCancel = async (jobId: string) => {
    if (!serviceUrl) {
      return;
    }
    setError(null);
    setCancelingJobId(jobId);
    try {
      await cancelJob(serviceUrl, jobId);
      const next = await listJobs(serviceUrl);
      setJobs(next);
    } catch (cancelError) {
      setError((cancelError as Error).message || t('cancelJobFailed'));
    } finally {
      setCancelingJobId(null);
    }
  };

  return (
    <section className="page">
      <header className="page-header page-header-compact">
        <div className="page-header-copy">
          <span className="page-kicker">{t('navExports')}</span>
          <h2 className="page-title">{t('navExports')}</h2>
        </div>
        <div className="page-actions">
          <button className="btn" onClick={openOutputFolder} type="button">
            {t('openOutputFolder')}
          </button>
        </div>
      </header>

      {!jobs.length ? (
        <article className="panel">
          <div className="empty-state">
            <strong>{t('noJobs')}</strong>
            <p className="kv">{t('exportsIntro')}</p>
          </div>
        </article>
      ) : (
        <>
          <div className="exports-toolbar">
            <p className="kv">
              {t('showingRange')}: {pageStart + 1}-{Math.min(pageStart + visibleJobs.length, sortedJobs.length)} / {sortedJobs.length}
            </p>
            {totalPages > 1 ? (
              <div className="pagination-controls">
                <button className="btn btn-compact" type="button" disabled={currentPage <= 1} onClick={() => setPage((value) => Math.max(1, value - 1))}>
                  {t('previous')}
                </button>
                <span className="metric-chip">
                  {t('page')} {currentPage}/{totalPages}
                </span>
                <button className="btn btn-compact" type="button" disabled={currentPage >= totalPages} onClick={() => setPage((value) => Math.min(totalPages, value + 1))}>
                  {t('next')}
                </button>
              </div>
            ) : null}
          </div>

          <div className="exports-list exports-list-paged">
            {visibleJobs.map((job) => {
            const mp3 = getArtifact(job.artifacts, 'mp3');
            const mp4 = getArtifact(job.artifacts, 'mp4');
            const isActive =
              job.state === 'queued' ||
              job.state === 'running' ||
              job.state === 'waiting_language' ||
              job.state === 'canceling' ||
              job.mp4State === 'queued' ||
              job.mp4State === 'running';
            const progressLabel = `${Math.round(job.progress * 100)}%`;
            return (
              <article key={job.id} className="status-card export-row">
                <div className="export-row-head">
                  <div className="export-row-title">
                    <strong>{job.sourceLabel}</strong>
                  </div>
                  <div className="export-row-status">
                    <div className="status-cluster export-row-status-cluster">
                      <span className={`status-pill ${toneForState(job.state)}`}>{job.state}</span>
                      {job.mp4State !== 'not_requested' ? (
                        <span className={`status-pill ${toneForState(job.mp4State)}`}>MP4 {job.mp4State}</span>
                      ) : null}
                    </div>
                    <button
                      className="btn btn-ghost-danger btn-compact"
                      type="button"
                      disabled={deletingJobId === job.id}
                      onClick={() => removeJob(job.id)}
                    >
                      {t('deleteOutput')}
                    </button>
                  </div>
                </div>

                {isActive ? (
                  <div className="export-row-summary">
                    <span>{t('progress')}: {progressLabel}</span>
                    <span>{t('phase')}: {job.phase}</span>
                  </div>
                ) : null}

                {isActive ? (
                  <div className="status-progress export-progress">
                    <span style={{ width: `${Math.max(3, job.progress * 100)}%` }} />
                  </div>
                ) : null}

                <div className="row export-row-actions">
                  <button className="btn btn-compact" type="button" onClick={() => setCurrentJobIdManual(job.id)}>
                    {t('navStudio')}
                  </button>
                  <button className="btn btn-compact" type="button" disabled={!mp3} onClick={() => mp3 && downloadAsset(mp3.id, `${job.id}.mp3`)}>
                    {t('downloadMp3')}
                  </button>
                  <button className="btn btn-compact" type="button" disabled={!mp4} onClick={() => mp4 && downloadAsset(mp4.id, `${job.id}.mp4`)}>
                    {t('downloadMp4')}
                  </button>
                  <button
                    className="btn btn-compact"
                    type="button"
                    disabled={!mp4}
                    onClick={() => {
                      if (!mp4) {
                        return;
                      }
                      setPreview({
                        assetId: mp4.id,
                        label: job.sourceLabel,
                      });
                    }}
                  >
                    {t('previewMp4')}
                  </button>
                  {isActive ? (
                    <button
                      className="btn btn-danger btn-compact"
                      type="button"
                      disabled={cancelingJobId === job.id}
                      onClick={() => requestCancel(job.id)}
                    >
                      {cancelingJobId === job.id ? t('canceling') : t('cancelJob')}
                    </button>
                  ) : null}
                </div>

                {job.mp4Error ? <p className="error">{job.mp4Error}</p> : null}
                {job.alignmentError ? <p className="error">{job.alignmentError}</p> : null}
                {job.cancelReason ? <p className="warning">{t('cancelReasonLabel')}: {job.cancelReason}</p> : null}
                {job.error ? <p className="error">{job.error}</p> : null}
              </article>
            );
          })}
          {error ? <p className="error">{error}</p> : null}
          </div>
        </>
      )}
      {preview ? (
        <Mp4PreviewDock
          serviceUrl={serviceUrl}
          assetId={preview.assetId}
          title={`${t('previewMp4')} - ${preview.label}`}
          onClose={() => setPreview(null)}
        />
      ) : null}
    </section>
  );
}

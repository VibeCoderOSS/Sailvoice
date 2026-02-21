import { useEffect, useState } from 'react';
import { deleteJob, listJobs } from '../api/client';
import { Mp4PreviewDock } from '../components/Mp4PreviewDock';
import { useI18n } from '../i18n/I18nProvider';
import { useAppStore } from '../state/appStore';
import type { Artifact } from '../types/models';

function getArtifact(jobArtifacts: Artifact[], type: 'mp3' | 'mp4') {
  return jobArtifacts.find((artifact) => artifact.type === type) ?? null;
}

export function ExportsPage() {
  const { t } = useI18n();
  const serviceUrl = useAppStore((state) => state.serviceUrl);
  const jobs = useAppStore((state) => state.jobs);
  const config = useAppStore((state) => state.config);
  const setJobs = useAppStore((state) => state.setJobs);
  const setCurrentJobIdManual = useAppStore((state) => state.setCurrentJobIdManual);
  const clearAudioQueue = useAppStore((state) => state.clearAudioQueue);
  const setCurrentTimeMs = useAppStore((state) => state.setCurrentTimeMs);
  const currentJobId = useAppStore((state) => state.currentJobId);
  const [deletingJobId, setDeletingJobId] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [preview, setPreview] = useState<{ assetId: string; label: string } | null>(null);

  useEffect(() => {
    if (!serviceUrl) {
      return;
    }
    listJobs(serviceUrl)
      .then((next) => setJobs(next))
      .catch(console.error);
  }, [serviceUrl, setJobs]);

  useEffect(() => {
    if (!serviceUrl) {
      return;
    }
    const hasMp4Background = jobs.some((job) => job.mp4State === 'queued' || job.mp4State === 'running');
    if (!hasMp4Background) {
      return;
    }
    const timer = window.setInterval(() => {
      listJobs(serviceUrl)
        .then((next) => setJobs(next))
        .catch(console.error);
    }, 4000);
    return () => window.clearInterval(timer);
  }, [jobs, serviceUrl, setJobs]);

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

  const removeJob = async (jobId: string, state: string) => {
    setError(null);
    if (state === 'queued' || state === 'running' || state === 'waiting_language') {
      setError(t('deleteActiveJobError'));
      return;
    }
    if (!window.confirm(t('deleteOutputConfirm'))) {
      return;
    }

    setDeletingJobId(jobId);
    try {
      await deleteJob(serviceUrl, jobId);
      const next = await listJobs(serviceUrl);
      setJobs(next);
      if (currentJobId === jobId) {
        setCurrentJobIdManual(null);
        clearAudioQueue();
        setCurrentTimeMs(0);
      }
    } catch (deleteError) {
      setError((deleteError as Error).message);
    } finally {
      setDeletingJobId(null);
    }
  };

  return (
    <section className="page">
      <div className="row-between">
        <h2 className="page-title">{t('navExports')}</h2>
        <button className="btn" onClick={openOutputFolder} type="button">
          {t('openOutputFolder')}
        </button>
      </div>

      <article className="panel">
        {!jobs.length ? (
          <p className="kv">{t('noJobs')}</p>
        ) : (
          jobs.map((job) => {
            const mp3 = getArtifact(job.artifacts, 'mp3');
            const mp4 = getArtifact(job.artifacts, 'mp4');
            const isActive = job.state === 'queued' || job.state === 'running' || job.state === 'waiting_language';
            return (
              <div key={job.id} className="status-card">
                <div className="row-between">
                  <div>
                    <strong>{job.sourceLabel}</strong>
                    <p className="kv">
                      {t('queueState')}: {job.state}
                    </p>
                    <p className="kv">
                      {t('progress')}: {Math.round(job.progress * 100)}%
                    </p>
                    <p className="kv">
                      {t('phase')}: {job.phase}
                    </p>
                    <p className="kv">
                      {t('mp4State')}: {job.mp4State}
                    </p>
                    <p className="kv">
                      {t('alignmentState')}: {job.alignmentState}
                    </p>
                  </div>

                  <div className="row">
                    <button className="btn" type="button" onClick={() => setCurrentJobIdManual(job.id)}>
                      {t('navStudio')}
                    </button>
                    <button className="btn" type="button" disabled={!mp3} onClick={() => mp3 && downloadAsset(mp3.id, `${job.id}.mp3`)}>
                      {t('downloadMp3')}
                    </button>
                    <button className="btn" type="button" disabled={!mp4} onClick={() => mp4 && downloadAsset(mp4.id, `${job.id}.mp4`)}>
                      {t('downloadMp4')}
                    </button>
                    <button
                      className="btn"
                      type="button"
                      disabled={!mp4}
                      onClick={() => {
                        if (!mp4) {
                          return;
                        }
                        setPreview({ assetId: mp4.id, label: job.sourceLabel });
                      }}
                    >
                      {t('previewMp4')}
                    </button>
                    <button
                      className="btn btn-danger"
                      type="button"
                      disabled={deletingJobId === job.id || isActive}
                      onClick={() => removeJob(job.id, job.state)}
                    >
                      {t('deleteOutput')}
                    </button>
                  </div>
                </div>
                <div className="status-progress">
                  <span style={{ width: `${Math.max(3, job.progress * 100)}%` }} />
                </div>
                {job.mp4Error ? <p className="error">{job.mp4Error}</p> : null}
                {job.alignmentError ? <p className="error">{job.alignmentError}</p> : null}
                {job.alignmentWarning ? <p className="warning">{job.alignmentWarning}</p> : null}
                {job.error ? <p className="error">{job.error}</p> : null}
              </div>
            );
          })
        )}
        {error ? <p className="error">{error}</p> : null}
      </article>
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

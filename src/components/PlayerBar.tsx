import { useEffect, useMemo, useRef, useState } from 'react';
import { getJob } from '../api/client';
import { useI18n } from '../i18n/I18nProvider';
import { useAppStore } from '../state/appStore';

function formatMs(ms: number) {
  const totalSec = Math.floor(ms / 1000);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${String(s).padStart(2, '0')}`;
}

type QueueChunk = { assetId: string; startMs: number; endMs: number };

export function PlayerBar({ serviceUrl }: { serviceUrl: string }) {
  const { t } = useI18n();
  const currentJobId = useAppStore((state) => state.currentJobId);
  const jobs = useAppStore((state) => state.jobs);
  const audioQueue = useAppStore((state) => state.audioQueue);
  const currentTimeMs = useAppStore((state) => state.currentTimeMs);
  const isPlaying = useAppStore((state) => state.isPlaying);
  const pendingSeekMs = useAppStore((state) => state.pendingSeekMs);
  const setCurrentTimeMs = useAppStore((state) => state.setCurrentTimeMs);
  const setIsPlaying = useAppStore((state) => state.setIsPlaying);
  const setPendingSeekMs = useAppStore((state) => state.setPendingSeekMs);
  const upsertJob = useAppStore((state) => state.upsertJob);

  const currentChunkRef = useRef<QueueChunk | null>(null);
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const serviceUrlRef = useRef(serviceUrl);
  const [audioError, setAudioError] = useState<string | null>(null);
  const [audioDurationMs, setAudioDurationMs] = useState<number>(0);

  useEffect(() => {
    serviceUrlRef.current = serviceUrl;
  }, [serviceUrl]);

  const currentJob = useMemo(() => jobs.find((job) => job.id === currentJobId) ?? null, [jobs, currentJobId]);

  const durationMs = useMemo(() => {
    const queueEnd = audioQueue.reduce((max, item) => Math.max(max, item.endMs), 0);
    const persisted = Math.max(0, currentJob?.audioDurationMs ?? 0);
    return Math.max(queueEnd, persisted, audioDurationMs);
  }, [audioQueue, audioDurationMs, currentJob?.audioDurationMs]);

  const hasPlayableAsset = useMemo(() => {
    if (!currentJob) {
      return false;
    }
    if (currentJob.state === 'done') {
      return true;
    }
    if (audioQueue.length > 0) {
      return true;
    }
    return currentJob.artifacts.some((artifact) => artifact.type === 'mp3' || artifact.type === 'audio_chunk');
  }, [audioQueue.length, currentJob]);

  const refreshCurrentJob = async () => {
    if (!serviceUrlRef.current || !currentJobId) {
      return null;
    }
    const refreshed = await getJob(serviceUrlRef.current, currentJobId);
    upsertJob(refreshed);
    return refreshed;
  };

  const getPreferredAssetId = (job = currentJob) => {
    if (!job) {
      return null;
    }
    const mp3 = job.artifacts.find((artifact) => artifact.type === 'mp3');
    if (mp3) {
      return mp3.id;
    }
    const firstChunk = job.artifacts.find((artifact) => artifact.type === 'audio_chunk');
    return firstChunk?.id ?? null;
  };

  const loadAndSeek = async (assetId: string, seekMs?: number) => {
    const audio = audioRef.current;
    if (!audio || !serviceUrlRef.current) {
      return false;
    }

    const nextSrc = `${serviceUrlRef.current}/v1/assets/${assetId}`;
    if (audio.src !== nextSrc) {
      audio.src = nextSrc;
    }

    if (typeof seekMs === 'number' && Number.isFinite(seekMs)) {
      const seconds = Math.max(0, seekMs / 1000);
      if (audio.readyState >= 1) {
        audio.currentTime = seconds;
      } else {
        await new Promise<void>((resolve) => {
          audio.addEventListener(
            'loadedmetadata',
            () => {
              audio.currentTime = seconds;
              resolve();
            },
            { once: true }
          );
        });
      }
    }

    return true;
  };

  const playFromFinalAsset = async (seekMs?: number) => {
    const audio = audioRef.current;
    if (!audio) {
      return false;
    }

    let assetId = getPreferredAssetId();
    if (!assetId) {
      const refreshed = await refreshCurrentJob();
      assetId = getPreferredAssetId(refreshed ?? currentJob);
    }
    if (!assetId) {
      return false;
    }

    setAudioError(null);
    currentChunkRef.current = null;
    const loaded = await loadAndSeek(assetId, seekMs);
    if (!loaded) {
      return false;
    }

    try {
      await audio.play();
      setIsPlaying(true);
      return true;
    } catch (error) {
      console.error(error);
      setAudioError((error as Error).message || 'Playback failed');
      setIsPlaying(false);
      return false;
    }
  };

  const playNextChunk = async () => {
    const nextChunk = useAppStore.getState().dequeueAudioChunk();
    const audio = audioRef.current;
    if (!nextChunk || !audio || !serviceUrlRef.current) {
      currentChunkRef.current = null;
      setIsPlaying(false);
      return false;
    }

    setAudioError(null);
    currentChunkRef.current = nextChunk;
    await loadAndSeek(nextChunk.assetId, 0);

    try {
      await audio.play();
      setIsPlaying(true);
      return true;
    } catch (error) {
      console.error(error);
      setAudioError((error as Error).message || 'Playback failed');
      setIsPlaying(false);
      return false;
    }
  };

  const togglePlayPause = async () => {
    const audio = audioRef.current;
    if (!audio) {
      return;
    }

    if (isPlaying) {
      audio.pause();
      return;
    }

    setAudioError(null);

    if (currentJob?.state === 'done') {
      const nearEnd = durationMs > 0 && currentTimeMs >= Math.max(0, durationMs - 80);
      const seekTarget = nearEnd ? 0 : currentTimeMs;
      if (nearEnd) {
        setCurrentTimeMs(0);
      }
      const started = await playFromFinalAsset(seekTarget);
      if (started) {
        return;
      }
    }

    if (!audio.src) {
      const started = await playNextChunk();
      if (started) {
        return;
      }
    }

    if (audio.ended || (audio.duration && audio.currentTime >= Math.max(0, audio.duration - 0.05))) {
      if (currentChunkRef.current) {
        audio.currentTime = 0;
        setCurrentTimeMs(currentChunkRef.current.startMs);
      } else {
        audio.currentTime = 0;
        setCurrentTimeMs(0);
      }
    }

    try {
      await audio.play();
      setIsPlaying(true);
    } catch (error) {
      console.error(error);
      setAudioError((error as Error).message || 'Playback failed');
    }
  };

  useEffect(() => {
    const audio = new Audio();
    audioRef.current = audio;

    const onEnded = () => {
      if (currentChunkRef.current) {
        void playNextChunk();
      } else {
        setIsPlaying(false);
      }
    };

    const onTimeUpdate = () => {
      const currentChunk = currentChunkRef.current;
      const timeMs = Math.round(audio.currentTime * 1000);
      const absoluteMs = currentChunk ? currentChunk.startMs + timeMs : timeMs;
      setCurrentTimeMs(absoluteMs);
    };

    const onLoadedMetadata = () => {
      const currentChunk = currentChunkRef.current;
      const offset = currentChunk ? currentChunk.startMs : 0;
      const duration = Math.round(audio.duration * 1000);
      if (Number.isFinite(duration) && duration > 0) {
        setAudioDurationMs((prev) => Math.max(prev, offset + duration));
      }
    };

    const onPlay = () => setIsPlaying(true);
    const onPause = () => setIsPlaying(false);
    const onError = () => {
      const mediaError = audio.error;
      if (mediaError) {
        setAudioError(`Audio error (${mediaError.code})`);
      } else {
        setAudioError('Audio playback error');
      }
      setIsPlaying(false);
    };

    audio.addEventListener('ended', onEnded);
    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('loadedmetadata', onLoadedMetadata);
    audio.addEventListener('play', onPlay);
    audio.addEventListener('pause', onPause);
    audio.addEventListener('error', onError);

    return () => {
      audio.pause();
      audio.removeEventListener('ended', onEnded);
      audio.removeEventListener('timeupdate', onTimeUpdate);
      audio.removeEventListener('loadedmetadata', onLoadedMetadata);
      audio.removeEventListener('play', onPlay);
      audio.removeEventListener('pause', onPause);
      audio.removeEventListener('error', onError);
    };
  }, [setCurrentTimeMs, setIsPlaying]);

  useEffect(() => {
    if (!audioRef.current) {
      return;
    }

    if (!currentChunkRef.current && audioQueue.length > 0) {
      void playNextChunk();
    }
  }, [audioQueue.length]);

  useEffect(() => {
    if (!audioRef.current) {
      return;
    }
    if (currentJob?.state === 'done') {
      const hasFinalMp3 = currentJob.artifacts.some((artifact) => artifact.type === 'mp3');
      if (hasFinalMp3) {
        currentChunkRef.current = null;
      }
      if (currentJob.audioDurationMs && currentJob.audioDurationMs > 0) {
        setAudioDurationMs(currentJob.audioDurationMs);
      }
    }
  }, [currentJob?.audioDurationMs, currentJob?.artifacts, currentJob?.state]);

  useEffect(() => {
    if (!audioRef.current || pendingSeekMs === null) {
      return;
    }

    const audio = audioRef.current;
    const currentChunk = currentChunkRef.current;
    const targetMs = Math.max(0, pendingSeekMs);

    const seekInCurrentAudio = (offsetMs: number) => {
      audio.currentTime = Math.max(0, offsetMs / 1000);
    };

    if (currentJob?.state === 'done') {
      const finalAssetId = getPreferredAssetId();
      if (finalAssetId) {
        void loadAndSeek(finalAssetId, targetMs);
        currentChunkRef.current = null;
      } else if (currentChunk && targetMs >= currentChunk.startMs && targetMs <= currentChunk.endMs) {
        seekInCurrentAudio(targetMs - currentChunk.startMs);
      } else {
        seekInCurrentAudio(targetMs);
      }
    } else if (currentChunk && targetMs >= currentChunk.startMs && targetMs <= currentChunk.endMs) {
      seekInCurrentAudio(targetMs - currentChunk.startMs);
    } else if (!currentChunk) {
      seekInCurrentAudio(targetMs);
    }

    setCurrentTimeMs(targetMs);
    setPendingSeekMs(null);
  }, [currentJob?.state, pendingSeekMs, setCurrentTimeMs, setPendingSeekMs]);

  useEffect(() => {
    if (!audioRef.current) {
      return;
    }
    audioRef.current.pause();
    audioRef.current.removeAttribute('src');
    audioRef.current.load();
    currentChunkRef.current = null;
    setAudioDurationMs(Math.max(0, currentJob?.audioDurationMs ?? 0));
    setCurrentTimeMs(0);
    setIsPlaying(false);
    setAudioError(null);
  }, [currentJobId, setCurrentTimeMs, setIsPlaying]);

  return (
    <section className="player-bar">
      <div className="row player-leading">
        <button className="btn" type="button" onClick={togglePlayPause} disabled={!hasPlayableAsset}>
          {isPlaying ? t('pause') : t('play')}
        </button>
        <span className="player-time">
          {formatMs(currentTimeMs)} / {formatMs(durationMs)}
        </span>
      </div>

      <input
        className="timeline"
        type="range"
        min={0}
        max={Math.max(durationMs, 1)}
        value={Math.min(currentTimeMs, durationMs || 0)}
        onChange={(event) => setPendingSeekMs(Number(event.target.value))}
      />

      <div className="row player-trailing">
        <span className="kv">{isPlaying ? 'LIVE' : t('notStarted')}</span>
        <span className="kv">{currentJob?.statusMessage || '-'}</span>
        {audioError ? <span className="error">{audioError}</span> : null}
      </div>
    </section>
  );
}

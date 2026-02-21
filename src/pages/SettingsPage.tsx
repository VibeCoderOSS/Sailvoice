import { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { downloadModel, getRuntimeStatus, listModels } from '../api/client';
import type { Locale } from '../i18n/translations';
import { useI18n } from '../i18n/I18nProvider';
import { MODELS } from '../constants';
import { useAppStore } from '../state/appStore';
import type { ModelCatalogId, ModelId, ModelInfo, ModelSource, RuntimeStatus } from '../types/models';

type SourceSelection = Record<ModelId, ModelSource>;

const defaultSourceSelection: SourceSelection = {
  base: 'mlx',
  customvoice: 'mlx',
  voicedesign: 'mlx'
};

const qualitySteps: Record<'speed' | 'balanced' | 'quality', number> = {
  speed: 16,
  balanced: 28,
  quality: 40
};

export function SettingsPage() {
  const { t, setLocale } = useI18n();
  const serviceUrl = useAppStore((state) => state.serviceUrl);
  const config = useAppStore((state) => state.config);
  const setConfig = useAppStore((state) => state.setConfig);

  const [busy, setBusy] = useState(false);
  const [models, setModels] = useState<ModelInfo[]>([]);
  const [runtime, setRuntime] = useState<RuntimeStatus | null>(null);
  const [downloading, setDownloading] = useState<Record<string, boolean>>({});
  const [downloadMessage, setDownloadMessage] = useState<string>('');
  const [runtimeMessage, setRuntimeMessage] = useState<string>('');
  const [sourceSelection, setSourceSelection] = useState<SourceSelection>(defaultSourceSelection);

  const groupedModels = useMemo(() => {
    const map = new Map<ModelCatalogId, ModelInfo[]>();
    for (const info of models) {
      const current = map.get(info.model) ?? [];
      current.push(info);
      map.set(info.model, current);
    }
    return map;
  }, [models]);
  const whisperxInfo = useMemo(
    () => groupedModels.get('whisperx')?.find((entry) => entry.source === 'mlx') ?? null,
    [groupedModels]
  );

  useEffect(() => {
    if (!serviceUrl) {
      return;
    }

    listModels(serviceUrl)
      .then((items) => setModels(items))
      .catch((error) => {
        console.error(error);
      });
    getRuntimeStatus(serviceUrl)
      .then((status) => setRuntime(status))
      .catch((error) => {
        console.error(error);
      });
  }, [serviceUrl]);

  if (!config) {
    return (
      <section className="page">
        <h2 className="page-title">{t('navSettings')}</h2>
      </section>
    );
  }

  const updateConfig = async (patch: Partial<typeof config>) => {
    setBusy(true);
    try {
      const next = await window.desktopApi.setConfig(patch);
      setConfig(next);
      setLocale(next.locale as Locale);
    } finally {
      setBusy(false);
    }
  };

  const openModelCacheFolder = async () => {
    await window.desktopApi.openPath(config.modelCacheDir);
  };

  const openOutputFolder = async () => {
    await window.desktopApi.openPath(config.outputDir);
  };

  const refreshModels = async () => {
    const next = await listModels(serviceUrl);
    setModels(next);
  };

  const triggerDownload = async (model: ModelCatalogId) => {
    const source: ModelSource = model === 'whisperx' ? 'mlx' : (sourceSelection[model] ?? 'mlx');
    const key = `${model}:${source}`;

    setDownloading((state) => ({ ...state, [key]: true }));
    setDownloadMessage('');
    try {
      const result = await downloadModel(serviceUrl, { model, source });
      await refreshModels();
      if (result.downloaded) {
        const suffix = result.warning ? ` (${result.warning})` : '';
        setDownloadMessage(`${t('downloadDone')}: ${result.repo}${suffix}`);
      } else {
        setDownloadMessage(result.warning || t('downloadFailed'));
      }
    } catch (error) {
      if (axios.isAxiosError(error)) {
        const detail = (error.response?.data as { detail?: string; warning?: string } | undefined)?.detail
          || (error.response?.data as { detail?: string; warning?: string } | undefined)?.warning;
        setDownloadMessage(`${t('downloadFailed')}: ${detail || error.message}`);
      } else {
        setDownloadMessage(`${t('downloadFailed')}: ${(error as Error).message}`);
      }
    } finally {
      setDownloading((state) => ({ ...state, [key]: false }));
    }
  };

  const verifyRuntime = async () => {
    if (!serviceUrl) {
      return;
    }
    setRuntimeMessage('');
    try {
      const status = await getRuntimeStatus(serviceUrl);
      setRuntime(status);
      setRuntimeMessage(status.reason);
    } catch (error) {
      setRuntimeMessage((error as Error).message);
    }
  };

  const whisperxLocalAvailable = Boolean(whisperxInfo?.downloaded);
  const whisperxButtonClass = whisperxLocalAvailable ? 'btn btn-available' : 'btn btn-primary';
  const whisperxButtonLabel = downloading['whisperx:mlx']
    ? t('downloading')
    : whisperxLocalAvailable
      ? t('localAvailable')
      : t('downloadNow');

  return (
    <section className="page">
      <h2 className="page-title">{t('navSettings')}</h2>

      <article className="panel">
        <div className="field">
          <label>{t('language')}</label>
          <select
            className="select"
            value={config.locale}
            onChange={(event) => updateConfig({ locale: event.target.value as 'de' | 'en' })}
            disabled={busy}
          >
            <option value="de">{t('german')}</option>
            <option value="en">{t('english')}</option>
          </select>
        </div>

        <div className="field">
          <label>{t('profile')}</label>
          <select
            className="select"
            value={config.performanceProfile}
            onChange={(event) => updateConfig({ performanceProfile: event.target.value as 'standard' | 'memory' })}
            disabled={busy}
          >
            <option value="standard">{t('standard')}</option>
            <option value="memory">{t('memory')}</option>
          </select>
        </div>

        <div className="field">
          <label>{t('qualityPreset')}</label>
          <select
            className="select"
            value={config.qualityPreset}
            onChange={(event) => updateConfig({ qualityPreset: event.target.value as 'speed' | 'balanced' | 'quality' })}
            disabled={busy}
          >
            <option value="speed">
              {t('qualitySpeed')} ({t('qualityStepInfo')}: {qualitySteps.speed})
            </option>
            <option value="balanced">
              {t('qualityBalanced')} ({t('qualityStepInfo')}: {qualitySteps.balanced})
            </option>
            <option value="quality">
              {t('qualityHigh')} ({t('qualityStepInfo')}: {qualitySteps.quality})
            </option>
          </select>
        </div>

        <div className="field">
          <label>{t('fallbackSetting')}</label>
          <label className="kv">
            <input
              type="checkbox"
              checked={config.allowFallback}
              onChange={(event) => updateConfig({ allowFallback: event.target.checked })}
              disabled={busy}
            />{' '}
            {t('fallbackEnableLabel')}
          </label>
        </div>

        <div className="panel panel-muted">
          <p className="kv">
            Model cache: <strong>{config.modelCacheDir}</strong>
          </p>
          <p className="kv">
            Output dir: <strong>{config.outputDir}</strong>
          </p>
          <p className="kv">{t('selfContainedInfo')}</p>
          <div className="row">
            <button className="btn" type="button" onClick={openModelCacheFolder}>
              {t('openModelFolder')}
            </button>
            <button className="btn" type="button" onClick={openOutputFolder}>
              {t('openOutputFolder')}
            </button>
          </div>
          <p className="kv">{t('outputDefaults')}</p>
          <p className="kv">
            {t('mp3Default')} | {t('mp4Default')}
          </p>
        </div>
      </article>

      <article className="panel">
        <h3 className="section-title">{t('runtimeStatus')}</h3>
        <div className="status-card">
          <p className="kv">
            {t('runtimeVersion')}: <strong>{runtime?.runtimeVersion || '-'}</strong>
          </p>
          <p className="kv">
            qwen3_tts: <strong>{runtime?.supportsQwen3Tts ? t('runtimeOk') : t('runtimeMissing')}</strong>
          </p>
          <p className="kv">
            {t('runtimeCompatible')}: <strong>{runtime?.mlxCompatible ? t('yes') : t('no')}</strong>
          </p>
          <p className="kv">{runtime?.reason || runtimeMessage}</p>
          <p className="kv">
            {t('alignmentRuntimeLabel')}: <strong>{runtime?.alignmentRuntimeReady ? t('yes') : t('no')}</strong>
          </p>
          <p className="kv">
            {t('alignmentModelLabel')}: <strong>{runtime?.alignmentModelsReady ? t('yes') : t('no')}</strong>
          </p>
          <p className="kv">{runtime?.alignmentReason || '-'}</p>
          <p className="kv">
            {t('alignmentProbeAtLabel')}: <strong>{runtime?.alignmentProbeAt || '-'}</strong>
          </p>
          <p className="kv">
            {t('alignmentProbeErrorLabel')}: <strong>{runtime?.alignmentProbeError || '-'}</strong>
          </p>
          <div className="row">
            <button className="btn" type="button" onClick={verifyRuntime}>
              {t('verifyRuntime')}
            </button>
          </div>
        </div>
      </article>

      <article className="panel">
        <h3 className="section-title">{t('modelDownloads')}</h3>

        {MODELS.map((item) => {
          const selectedSource = sourceSelection[item.id];
          const info = groupedModels.get(item.id)?.find((entry) => entry.source === selectedSource) ?? null;
          const key = `${item.id}:${selectedSource}`;
          const localAvailable = Boolean(info?.downloaded);
          const buttonClass = localAvailable ? 'btn btn-available' : 'btn btn-primary';
          const buttonLabel = downloading[key]
            ? t('downloading')
            : localAvailable
              ? t('localAvailable')
              : t('downloadNow');

          return (
            <div className="status-card" key={item.id}>
              <div className="row-between">
                <div>
                  <strong>{item.label}</strong>
                  <p className="kv">{info?.repo ?? item.hfRepo}</p>
                  <p className="kv">{info?.localPath ?? config.modelCacheDir}</p>
                  {!localAvailable ? <p className="kv">{t('modelMissing')}</p> : null}
                </div>

                <div className="row">
                  <select
                    value={selectedSource}
                    onChange={(event) =>
                      setSourceSelection((state) => ({
                        ...state,
                        [item.id]: event.target.value as ModelSource
                      }))
                    }
                    className="select source-select"
                    disabled={localAvailable}
                  >
                    <option value="mlx">MLX 8bit</option>
                    <option value="official">Official</option>
                  </select>
                  <button className={buttonClass} type="button" onClick={() => triggerDownload(item.id)} disabled={!!downloading[key] || localAvailable}>
                    {buttonLabel}
                  </button>
                </div>
              </div>
            </div>
          );
        })}

        <div className="status-card">
          <div className="row-between">
            <div>
              <strong>{t('alignmentModelTitle')}</strong>
              <p className="kv">{whisperxInfo?.repo ?? 'whisperx/forced-alignment'}</p>
              <p className="kv">{whisperxInfo?.localPath ?? config.modelCacheDir}</p>
              {!whisperxLocalAvailable ? <p className="kv">{t('modelMissing')}</p> : null}
            </div>

            <div className="row">
              <button
                className={whisperxButtonClass}
                type="button"
                onClick={() => triggerDownload('whisperx')}
                disabled={!!downloading['whisperx:mlx'] || whisperxLocalAvailable}
              >
                {whisperxButtonLabel}
              </button>
            </div>
          </div>
          <p className="kv">{t('alignmentModelHint')}</p>
        </div>

        {downloadMessage ? <p className="kv">{downloadMessage}</p> : null}
      </article>
    </section>
  );
}

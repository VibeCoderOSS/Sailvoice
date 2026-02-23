import { useEffect, useMemo, useState } from 'react';
import { Navigate, Route, Routes, useLocation, useNavigate } from 'react-router-dom';
import { useI18n } from './i18n/I18nProvider';
import { useAppStore } from './state/appStore';
import { downloadModel, getRuntimeStatus, listJobs, listModels, listVoices } from './api/client';
import { MODELS } from './constants';
import { StudioPage } from './pages/StudioPage';
import { PdfReaderPage } from './pages/PdfReaderPage';
import { VoiceClonePage } from './pages/VoiceClonePage';
import { VoiceDesignPage } from './pages/VoiceDesignPage';
import { ExportsPage } from './pages/ExportsPage';
import { SettingsPage } from './pages/SettingsPage';
import { PlayerBar } from './components/PlayerBar';
import { ModelBootstrapModal } from './components/ModelBootstrapModal';
import type { ModelCatalogId, ModelInfo, RuntimeStatus } from './types/models';
import ship42Logo from '../Logo.jpeg';

type NavItem = {
  path: string;
  label: string;
};

function pickPreferredJobId(
  jobs: Array<{
    id: string;
    state: 'queued' | 'running' | 'waiting_language' | 'done' | 'failed';
    updatedAt: string;
  }>
) {
  if (!jobs.length) {
    return null;
  }
  const stateRank = (state: string) => {
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

async function waitForBackendReady(baseUrl: string, timeoutMs = 30000): Promise<void> {
  const startedAt = Date.now();
  while (Date.now() - startedAt < timeoutMs) {
    try {
      const response = await fetch(`${baseUrl}/health`, { cache: 'no-store' });
      if (response.ok) {
        return;
      }
    } catch {
      // retry until timeout
    }
    await new Promise((resolve) => window.setTimeout(resolve, 350));
  }
  throw new Error('Backend did not become healthy in time.');
}

export function App() {
  const { t, locale, setLocale } = useI18n();
  const location = useLocation();
  const navigate = useNavigate();

  const serviceUrl = useAppStore((state) => state.serviceUrl);
  const config = useAppStore((state) => state.config);
  const setServiceUrl = useAppStore((state) => state.setServiceUrl);
  const setConfig = useAppStore((state) => state.setConfig);
  const setJobs = useAppStore((state) => state.setJobs);
  const setVoices = useAppStore((state) => state.setVoices);
  const setCurrentJobId = useAppStore((state) => state.setCurrentJobId);
  const [bootstrapDismissed, setBootstrapDismissed] = useState(false);
  const [missingModels, setMissingModels] = useState<ModelCatalogId[]>([]);
  const [bootstrapStatus, setBootstrapStatus] = useState<Record<string, string>>({});
  const [bootstrapDownloading, setBootstrapDownloading] = useState(false);

  const resolveMissingModels = (models: ModelInfo[], runtime: RuntimeStatus): ModelCatalogId[] => {
    const required: ModelCatalogId[] = ['base', 'customvoice', 'voicedesign', 'whisperx'];
    return required.filter((model) => {
      if (model === 'whisperx') {
        const hasCatalog = models.some((entry) => entry.model === 'whisperx' && entry.downloaded);
        return !hasCatalog || !runtime.alignmentModelsReady;
      }
      const hasLocal = models.some((entry) => entry.model === model && entry.downloaded);
      return !hasLocal;
    });
  };

  const refreshMissingModels = async (url: string) => {
    const [models, runtime] = await Promise.all([listModels(url), getRuntimeStatus(url)]);
    const missing = resolveMissingModels(models, runtime);
    setMissingModels(missing);
    return missing;
  };

  useEffect(() => {
    let mounted = true;
    let retryTimer: number | null = null;

    const bootstrap = async () => {
      const [url, cfg] = await Promise.all([window.desktopApi.getServiceUrl(), window.desktopApi.getConfig()]);
      if (!mounted) {
        return;
      }
      setServiceUrl(url);
      setConfig(cfg);
      setLocale(cfg.locale);
      await waitForBackendReady(url);

      const [jobs, voices] = await Promise.all([listJobs(url), listVoices(url)]);
      if (!mounted) {
        return;
      }
      setJobs(jobs);
      setVoices(voices);

      const preferredJobId = pickPreferredJobId(jobs);
      const state = useAppStore.getState();
      const selectedExists = state.currentJobId ? jobs.some((job) => job.id === state.currentJobId) : false;
      if (state.currentJobSelectionMode !== 'manual' || !selectedExists) {
        setCurrentJobId(preferredJobId);
      }

      void refreshMissingModels(url)
        .then((missing) => {
          if (!mounted) {
            return;
          }
          if (missing.length > 0 && !bootstrapDismissed) {
            setBootstrapStatus({});
          }
        })
        .catch((error) => {
          console.error(error);
        });
    };

    const scheduleRetry = () => {
      if (retryTimer !== null) {
        window.clearTimeout(retryTimer);
      }
      retryTimer = window.setTimeout(() => {
        void runBootstrap();
      }, 1500);
    };

    const runBootstrap = async () => {
      try {
        await bootstrap();
      } catch (error) {
        console.error(error);
        if (!mounted) {
          return;
        }
        scheduleRetry();
      }
    };

    void runBootstrap();

    return () => {
      mounted = false;
      if (retryTimer !== null) {
        window.clearTimeout(retryTimer);
      }
    };
  }, [bootstrapDismissed, setConfig, setCurrentJobId, setJobs, setLocale, setServiceUrl, setVoices]);

  useEffect(() => {
    if (config?.locale && config.locale !== locale) {
      setLocale(config.locale);
    }
  }, [config?.locale, locale, setLocale]);

  const navItems = useMemo<NavItem[]>(
    () => [
      { path: '/studio', label: t('navStudio') },
      { path: '/reader', label: t('navReader') },
      { path: '/voice-clone', label: t('navVoiceClone') },
      { path: '/voice-design', label: t('navVoiceDesign') },
      { path: '/exports', label: t('navExports') },
      { path: '/settings', label: t('navSettings') }
    ],
    [t]
  );

  const downloadAllMissingModels = async () => {
    if (!serviceUrl || !missingModels.length) {
      return;
    }
    setBootstrapDownloading(true);
    const nextStatus: Record<string, string> = {};
    for (const model of missingModels) {
      nextStatus[model] = 'Lädt...';
      setBootstrapStatus({ ...nextStatus });
      try {
        const result = await downloadModel(serviceUrl, { model, source: 'mlx' });
        if (result.downloaded) {
          nextStatus[model] = 'Lokal verfügbar';
        } else {
          nextStatus[model] = result.warning || 'Fehlgeschlagen';
        }
      } catch (error) {
        nextStatus[model] = (error as Error).message || 'Fehlgeschlagen';
      }
      setBootstrapStatus({ ...nextStatus });
    }
    try {
      const missing = await refreshMissingModels(serviceUrl);
      if (!missing.length) {
        setBootstrapDismissed(true);
      }
    } catch (error) {
      console.error(error);
    } finally {
      setBootstrapDownloading(false);
    }
  };

  return (
    <div className="app-shell">
      <aside className="app-sidebar">
        <div className="logo-block">
          <div className="brand-mark" aria-hidden="true">
            <img src={ship42Logo} alt="Ship-42 Logo" className="brand-mark-image" />
            <div className="brand-mark-copy">
              <p className="brand-mark-name">Ship-42</p>
              <p className="brand-mark-motto">Open local AI. Set sail offline.</p>
            </div>
          </div>
          <p className="logo-subtitle">Qwen3-TTS</p>
          <h1 className="logo-title">Voice Studio</h1>
          <p className="logo-footnote">localship line</p>
        </div>

        <nav className="nav-list" aria-label="Primary">
          {navItems.map((item) => {
            const active = location.pathname === item.path;
            return (
              <button
                key={item.path}
                className={`nav-item ${active ? 'active' : ''}`}
                onClick={() => navigate(item.path)}
                type="button"
              >
                {item.label}
              </button>
            );
          })}
        </nav>

        <div className="model-chips">
          {MODELS.map((model) => (
            <span key={model.id} className="chip">
              {model.id.toUpperCase()}
            </span>
          ))}
        </div>
      </aside>

      <main className="app-main">
        <Routes>
          <Route path="/" element={<Navigate to="/studio" replace />} />
          <Route path="/studio" element={<StudioPage />} />
          <Route path="/reader" element={<PdfReaderPage />} />
          <Route path="/voice-clone" element={<VoiceClonePage />} />
          <Route path="/voice-design" element={<VoiceDesignPage />} />
          <Route path="/voices" element={<Navigate to="/voice-clone" replace />} />
          <Route path="/exports" element={<ExportsPage />} />
          <Route path="/settings" element={<SettingsPage />} />
          <Route path="*" element={<Navigate to="/studio" replace />} />
        </Routes>
      </main>

      <ModelBootstrapModal
        open={!bootstrapDismissed && missingModels.length > 0}
        missingModels={missingModels}
        statusByModel={bootstrapStatus}
        isDownloading={bootstrapDownloading}
        onDownloadAll={downloadAllMissingModels}
        onLater={() => setBootstrapDismissed(true)}
      />

      <PlayerBar serviceUrl={serviceUrl} />
    </div>
  );
}

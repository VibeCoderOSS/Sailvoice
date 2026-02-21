import { useState } from 'react';
import { createDesignVoice, createVoice, deleteVoice, listVoices, renameVoice, requestVoicePreview } from '../api/client';
import { useI18n } from '../i18n/I18nProvider';
import { useAppStore } from '../state/appStore';
import type { VoiceItem } from '../types/models';

const LANGUAGE_CODES = ['auto', 'de', 'en', 'fr', 'it', 'es', 'pt', 'zh', 'ja', 'ko', 'ru'] as const;

function normalizeLanguage(value: string) {
  if (value === 'auto') {
    return undefined;
  }
  return value;
}

export function VoicesPage() {
  const { t } = useI18n();
  const serviceUrl = useAppStore((state) => state.serviceUrl);
  const voices = useAppStore((state) => state.voices);
  const setVoices = useAppStore((state) => state.setVoices);

  const [cloneName, setCloneName] = useState('');
  const [cloneLanguage, setCloneLanguage] = useState<string>('auto');
  const [audioPath, setAudioPath] = useState('');
  const [referenceText, setReferenceText] = useState('');

  const [designName, setDesignName] = useState('');
  const [designLanguage, setDesignLanguage] = useState<string>('auto');
  const [designDescription, setDesignDescription] = useState('');

  const [testText, setTestText] = useState('This is a local voice preview.');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const refresh = async () => {
    const next = await listVoices(serviceUrl);
    setVoices(next);
  };

  const pickAudio = async () => {
    const path = await window.desktopApi.pickAudio();
    if (path) {
      setAudioPath(path);
    }
  };

  const addCloneVoice = async () => {
    if (!cloneName.trim() || !audioPath) {
      return;
    }

    setBusy(true);
    try {
      await createVoice(serviceUrl, {
        name: cloneName.trim(),
        language: normalizeLanguage(cloneLanguage),
        audioPath,
        referenceText: referenceText.trim() || undefined
      });
      setCloneName('');
      setCloneLanguage('auto');
      setAudioPath('');
      setReferenceText('');
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const addDesignVoice = async () => {
    if (!designName.trim() || !designDescription.trim()) {
      return;
    }

    setBusy(true);
    try {
      await createDesignVoice(serviceUrl, {
        name: designName.trim(),
        language: normalizeLanguage(designLanguage),
        description: designDescription.trim()
      });
      setDesignName('');
      setDesignLanguage('auto');
      setDesignDescription('');
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const removeVoice = async (voiceId: string) => {
    setBusy(true);
    try {
      await deleteVoice(serviceUrl, voiceId);
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const renameExistingVoice = async (voiceId: string, currentName: string) => {
    const next = window.prompt('New name', currentName);
    if (!next?.trim()) {
      return;
    }

    setBusy(true);
    try {
      await renameVoice(serviceUrl, voiceId, next.trim());
      await refresh();
    } finally {
      setBusy(false);
    }
  };

  const previewVoice = async (voice?: VoiceItem) => {
    setError(null);
    try {
      const model = voice?.type === 'design' ? 'voicedesign' : 'base';
      const response = await requestVoicePreview(serviceUrl, {
        voiceId: voice?.id,
        text: testText,
        model
      });
      const audio = new Audio(`${serviceUrl}/v1/assets/${response.assetId}`);
      await audio.play();
    } catch (previewError) {
      setError((previewError as Error).message);
    }
  };

  return (
    <section className="page">
      <h2 className="page-title">{t('navVoices')}</h2>

      <div className="grid-two">
        <article className="panel">
          <h3>{t('createVoice')}</h3>

          <div className="subsection-card">
            <div className="field">
              <label>{t('cloneVoiceSection')}</label>
            </div>

            <div className="field">
              <label>{t('voiceName')}</label>
              <input className="input" value={cloneName} onChange={(event) => setCloneName(event.target.value)} />
            </div>

            <div className="field">
              <label>{t('optionalLanguage')}</label>
              <select className="select" value={cloneLanguage} onChange={(event) => setCloneLanguage(event.target.value)}>
                {LANGUAGE_CODES.map((code) => (
                  <option key={`clone-${code}`} value={code}>
                    {code === 'auto' ? t('auto') : code.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label>{t('sampleAudio')}</label>
              <div className="row">
                <button className="btn" type="button" onClick={pickAudio}>
                  {t('chooseAudio')}
                </button>
                <span className="kv">{audioPath || '-'}</span>
              </div>
            </div>

            <div className="field">
              <label>{t('referenceTextOptional')}</label>
              <textarea
                className="textarea"
                value={referenceText}
                onChange={(event) => setReferenceText(event.target.value)}
                placeholder={t('referenceTextHint')}
              />
            </div>

            <button className="btn btn-primary" type="button" onClick={addCloneVoice} disabled={busy}>
              {t('add')} ({t('cloneType')})
            </button>
          </div>

          <hr className="section-divider" />

          <div className="subsection-card">
            <div className="field">
              <label>{t('designVoiceSection')}</label>
            </div>

            <div className="field">
              <label>{t('voiceName')}</label>
              <input className="input" value={designName} onChange={(event) => setDesignName(event.target.value)} />
            </div>

            <div className="field">
              <label>{t('optionalLanguage')}</label>
              <select className="select" value={designLanguage} onChange={(event) => setDesignLanguage(event.target.value)}>
                {LANGUAGE_CODES.map((code) => (
                  <option key={`design-${code}`} value={code}>
                    {code === 'auto' ? t('auto') : code.toUpperCase()}
                  </option>
                ))}
              </select>
            </div>

            <div className="field">
              <label>{t('voiceDescription')}</label>
              <textarea
                className="textarea"
                value={designDescription}
                onChange={(event) => setDesignDescription(event.target.value)}
                placeholder={t('voiceDescriptionHint')}
              />
            </div>

            <button className="btn btn-primary" type="button" onClick={addDesignVoice} disabled={busy}>
              {t('add')} ({t('designType')})
            </button>
          </div>
        </article>

        <article className="panel">
          <div className="field">
            <label>{t('testText')}</label>
            <textarea className="textarea" value={testText} onChange={(event) => setTestText(event.target.value)} />
          </div>

          <button className="btn" type="button" onClick={() => previewVoice(undefined)}>
            {t('preview')} ({t('auto')})
          </button>

          <hr className="section-divider" />

          {voices.map((voice) => (
            <article key={voice.id} className="voice-item">
              <div className="row-between">
                <div>
                  <strong>{voice.name}</strong>
                  <p className="kv">
                    {t('voiceType')}:{' '}
                    <span className={`pill ${voice.type === 'design' ? 'pill-design' : 'pill-clone'}`}>
                      {voice.type === 'design' ? t('designType') : t('cloneType')}
                    </span>
                  </p>
                  <p className="kv">{voice.language || t('auto')}</p>
                  {voice.type === 'design' && voice.description ? <p className="kv">{voice.description}</p> : null}
                </div>

                <div className="row">
                  <button className="btn" type="button" onClick={() => previewVoice(voice)}>
                    {t('preview')}
                  </button>
                  <button className="btn" type="button" onClick={() => renameExistingVoice(voice.id, voice.name)}>
                    {t('rename')}
                  </button>
                  <button className="btn btn-danger" type="button" onClick={() => removeVoice(voice.id)}>
                    {t('delete')}
                  </button>
                </div>
              </div>
            </article>
          ))}
          {error ? <p className="error">{error}</p> : null}
        </article>
      </div>
    </section>
  );
}

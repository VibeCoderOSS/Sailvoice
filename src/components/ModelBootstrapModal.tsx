import type { ModelCatalogId } from '../types/models';
import { useI18n } from '../i18n/I18nProvider';

type ModelBootstrapModalProps = {
  open: boolean;
  missingModels: ModelCatalogId[];
  statusByModel: Record<string, string>;
  isDownloading: boolean;
  onDownloadAll: () => void;
  onLater: () => void;
};

export function ModelBootstrapModal({
  open,
  missingModels,
  statusByModel,
  isDownloading,
  onDownloadAll,
  onLater,
}: ModelBootstrapModalProps) {
  const { t } = useI18n();
  if (!open) {
    return null;
  }

  return (
    <div className="language-modal-backdrop">
      <article className="model-bootstrap-modal">
        <div className="modal-header">
          <span className="page-kicker">{t('modelDownloads')}</span>
          <h3>{t('modelBootstrapTitle')}</h3>
          <p className="kv modal-copy">{t('modelBootstrapBody')}</p>
        </div>
        <div className="modal-list">
          {missingModels.map((model) => (
            <div key={model} className="row-between model-bootstrap-item">
              <div>
                <strong>{model}</strong>
              </div>
              <span className="status-pill tone-neutral">{statusByModel[model] || t('notStarted')}</span>
            </div>
          ))}
        </div>
        <div className="row modal-actions">
          <button className="btn btn-primary" type="button" onClick={onDownloadAll} disabled={isDownloading}>
            {isDownloading ? t('downloading') : t('downloadAllModels')}
          </button>
          <button className="btn" type="button" onClick={onLater} disabled={isDownloading}>
            {t('later')}
          </button>
        </div>
      </article>
    </div>
  );
}

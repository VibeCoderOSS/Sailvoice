import type { ModelCatalogId } from '../types/models';

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
  if (!open) {
    return null;
  }

  return (
    <div className="language-modal-backdrop">
      <article className="model-bootstrap-modal">
        <h3>Lokale Modelle laden</h3>
        <p className="kv">
          Einige Modelle sind noch nicht lokal vorhanden. Du kannst sie jetzt in einem Schritt laden.
        </p>
        <div className="field">
          {missingModels.map((model) => (
            <div key={model} className="row-between model-bootstrap-item">
              <span className="kv">{model}</span>
              <span className="kv">{statusByModel[model] || 'Ausstehend'}</span>
            </div>
          ))}
        </div>
        <div className="row">
          <button className="btn btn-primary" type="button" onClick={onDownloadAll} disabled={isDownloading}>
            {isDownloading ? 'Lädt...' : 'Alle Modelle laden'}
          </button>
          <button className="btn" type="button" onClick={onLater} disabled={isDownloading}>
            Später
          </button>
        </div>
      </article>
    </div>
  );
}

type Mp4PreviewDockProps = {
  serviceUrl: string;
  assetId: string;
  title: string;
  onClose: () => void;
};

export function Mp4PreviewDock({ serviceUrl, assetId, title, onClose }: Mp4PreviewDockProps) {
  return (
    <aside className="mp4-preview-dock" aria-label="MP4 preview dock">
      <div className="row-between mp4-preview-head">
        <strong>{title}</strong>
        <button className="btn" type="button" onClick={onClose}>
          X
        </button>
      </div>
      <video
        className="mp4-preview-video"
        src={`${serviceUrl}/v1/assets/${assetId}`}
        controls
        preload="metadata"
      />
    </aside>
  );
}

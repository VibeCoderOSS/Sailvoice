import { useEffect, useMemo, useState } from 'react';
import { Document, Page, pdfjs } from 'react-pdf';
import { useI18n } from '../i18n/I18nProvider';
import { useAppStore } from '../state/appStore';

pdfjs.GlobalWorkerOptions.workerSrc = new URL('pdfjs-dist/build/pdf.worker.min.mjs', import.meta.url).toString();

export function PdfReaderPage() {
  const { t } = useI18n();
  const selectedPdfPath = useAppStore((state) => state.selectedPdfPath);
  const selectedPdfPage = useAppStore((state) => state.selectedPdfPage);
  const setSelectedPdfPage = useAppStore((state) => state.setSelectedPdfPage);
  const currentJobId = useAppStore((state) => state.currentJobId);
  const jobs = useAppStore((state) => state.jobs);
  const currentTimeMs = useAppStore((state) => state.currentTimeMs);

  const [numPages, setNumPages] = useState(0);
  const [pageWidth, setPageWidth] = useState(860);
  const [pageHeight, setPageHeight] = useState(1110);
  const [pdfSourceWidth, setPdfSourceWidth] = useState(595);
  const [pdfSourceHeight, setPdfSourceHeight] = useState(842);

  const currentJob = useMemo(() => jobs.find((job) => job.id === currentJobId) ?? null, [jobs, currentJobId]);
  const activeWord = useMemo(
    () => currentJob?.words.find((word) => currentTimeMs >= word.startMs && currentTimeMs <= word.endMs) ?? null,
    [currentJob?.words, currentTimeMs]
  );

  useEffect(() => {
    const onResize = () => {
      const maxWidth = Math.max(320, Math.min(window.innerWidth - 420, 920));
      const availableHeight = Math.max(420, window.innerHeight - 260);
      const heightBoundWidth = (availableHeight * pdfSourceWidth) / pdfSourceHeight;
      const width = Math.max(320, Math.min(maxWidth, heightBoundWidth));
      setPageWidth(Math.round(width));
    };

    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, [pdfSourceHeight, pdfSourceWidth]);

  const highlight = useMemo(() => {
    if (!activeWord?.bbox || activeWord.page === null || activeWord.page + 1 !== selectedPdfPage) {
      return null;
    }

    const bbox = activeWord.bbox;
    return {
      left: `${(bbox.x0 / pdfSourceWidth) * pageWidth}px`,
      top: `${(bbox.y0 / pdfSourceHeight) * pageHeight}px`,
      width: `${((bbox.x1 - bbox.x0) / pdfSourceWidth) * pageWidth}px`,
      height: `${((bbox.y1 - bbox.y0) / pdfSourceHeight) * pageHeight}px`
    };
  }, [activeWord, selectedPdfPage, pageWidth, pageHeight, pdfSourceHeight, pdfSourceWidth]);

  const nextPage = () => {
    setSelectedPdfPage(Math.min(selectedPdfPage + 1, numPages || 1));
  };

  const prevPage = () => {
    setSelectedPdfPage(Math.max(selectedPdfPage - 1, 1));
  };

  return (
    <section className="page page-reader">
      <header className="page-header page-header-compact">
        <div className="page-header-copy">
          <span className="page-kicker">{t('navReader')}</span>
          <h2 className="page-title">{t('navReader')}</h2>
        </div>
      </header>

      {!selectedPdfPath ? (
        <article className="panel panel-muted">
          <p className="kv">{t('noPdf')}</p>
        </article>
      ) : (
        <div className="reader-layout">
          <article className="panel reader-stage">
            <div className="reader-toolbar-card">
              <div>
                <span className="section-eyebrow">{t('pdf')}</span>
                <strong>{selectedPdfPath.split('/').pop()}</strong>
              </div>
              <div className="row reader-toolbar-nav">
                <button className="btn reader-nav-btn" type="button" onClick={prevPage}>
                  ←
                </button>
                <span className="status-pill tone-neutral">
                  {t('page')} {selectedPdfPage}/{numPages || 1}
                </span>
                <button className="btn reader-nav-btn" type="button" onClick={nextPage}>
                  →
                </button>
              </div>
            </div>

            <div className="pdf-stage-shell">
              <div className="pdf-canvas-wrap">
                <Document file={selectedPdfPath} onLoadSuccess={({ numPages: pages }) => setNumPages(pages)}>
                  <Page
                    pageNumber={selectedPdfPage}
                    width={pageWidth}
                    onRenderSuccess={(page) => {
                      const viewport = page.getViewport({ scale: 1 });
                      setPdfSourceWidth(viewport.width);
                      setPdfSourceHeight(viewport.height);
                      setPageHeight((viewport.height / viewport.width) * pageWidth);
                    }}
                  />
                </Document>
                {highlight ? <div className="highlight-word" style={highlight} /> : null}
              </div>
            </div>
          </article>

          <aside className="reader-side-column">
            <article className="panel panel-muted reader-summary-card">
              <div className="status-inline-grid">
                <div className="status-inline-item">
                  <span className="tile-label">{t('currentWord')}</span>
                  <strong>{activeWord?.word ?? '-'}</strong>
                </div>
                <div className="status-inline-item">
                  <span className="tile-label">{t('queueState')}</span>
                  <strong>{currentJob?.state ?? '-'}</strong>
                </div>
                <div className="status-inline-item">
                  <span className="tile-label">{t('page')}</span>
                  <strong>{selectedPdfPage}</strong>
                </div>
              </div>
              <div className="status-note-list">
                <p className="kv">{t('elapsedTime')}: {Math.max(0, Math.round(currentTimeMs / 1000))}s</p>
              </div>
            </article>
          </aside>
        </div>
      )}
    </section>
  );
}

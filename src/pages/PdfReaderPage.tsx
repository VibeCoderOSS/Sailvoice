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
      const width = Math.max(320, Math.min(window.innerWidth - 420, 940));
      setPageWidth(width);
    };

    onResize();
    window.addEventListener('resize', onResize);
    return () => window.removeEventListener('resize', onResize);
  }, []);

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
    <section className="page">
      <h2 className="page-title">{t('navReader')}</h2>

      {!selectedPdfPath ? (
        <article className="panel panel-muted">
          <p className="kv">{t('noPdf')}</p>
        </article>
      ) : (
        <article className="panel">
          <div className="row-between reader-toolbar">
            <div className="row reader-toolbar-nav">
              <button className="btn" type="button" onClick={prevPage}>
                ←
              </button>
              <span className="kv">
                {t('page')} {selectedPdfPage}/{numPages || 1}
              </span>
              <button className="btn" type="button" onClick={nextPage}>
                →
              </button>
            </div>
          </div>

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
        </article>
      )}
    </section>
  );
}

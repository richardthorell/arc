import { useEffect, useMemo, useRef, useState } from 'react';
import { Download, Grid2X2, List, Search, X } from 'lucide-react';

import type {
  ArcAssetDownloadManifest,
  ArcAssetImportProgress,
  ArcAssetSearchResult,
  ArcAssetSourceDescriptor,
  ArcRemoteAsset,
  ArcRemoteAssetKind,
} from '../../../common/assetSourceTypes';
import { remoteAssetFailureMessage } from './remoteAssetErrors';
import {
  beginRemoteAssetImport,
  cancelRemoteAssetImport,
  completeRemoteAssetImport,
  failRemoteAssetImport,
  initialRemoteAssetImportLifecycle,
  requestRemoteAssetImportCancellation,
  updateRemoteAssetImportProgress,
} from './remoteAssetImportLifecycle';
import {
  manifestFormats,
  manifestResolutions,
  manifestSelectionBytes,
  preferredFormat,
  preferredResolution,
  selectManifestFiles,
} from './remoteAssetVariants';

import './remoteAssetBrowser.css';

type Props = {
  source: ArcAssetSourceDescriptor;
  onImportedFiles?: (paths: readonly string[]) => void;
};

const formatBytes = (bytes: number | undefined): string => {
  if (bytes === undefined) return 'Size unavailable';
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KiB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MiB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GiB`;
};

const progressLabel = (progress: ArcAssetImportProgress | null): string => {
  if (!progress) return '';
  if (progress.phase === 'complete') return `Imported ${progress.completedFiles} files`;
  const file = progress.currentFile ? ` · ${progress.currentFile}` : '';
  return `${progress.phase} ${progress.completedFiles}/${progress.totalFiles}${file}`;
};

const isCancellationReason = (reason: unknown): boolean => {
  const message = reason instanceof Error ? reason.message : String(reason);
  return (reason instanceof Error && reason.name === 'AbortError') || /cancelled|canceled|aborted/i.test(message);
};

export function RemoteAssetBrowser({ source, onImportedFiles }: Props) {
  const [search, setSearch] = useState('');
  const [kind, setKind] = useState<ArcRemoteAssetKind | 'all'>('all');
  const [view, setView] = useState<'grid' | 'list'>('grid');
  const [result, setResult] = useState<ArcAssetSearchResult | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [selected, setSelected] = useState<ArcRemoteAsset | null>(null);
  const [manifest, setManifest] = useState<ArcAssetDownloadManifest | null>(null);
  const [manifestLoading, setManifestLoading] = useState(false);
  const [resolution, setResolution] = useState('');
  const [format, setFormat] = useState('');
  const [importProgress, setImportProgress] = useState<ArcAssetImportProgress | null>(null);
  const [importMessage, setImportMessage] = useState('');
  const [importLifecycle, setImportLifecycle] = useState(initialRemoteAssetImportLifecycle);
  const searchRevision = useRef(0);
  const manifestRevision = useRef(0);
  const activeOperation = useRef<number | null>(null);
  const importing = importLifecycle.phase === 'running' || importLifecycle.phase === 'canceling';
  const canceling = importLifecycle.phase === 'canceling';

  useEffect(() => {
    const revision = ++searchRevision.current;
    const timeout = window.setTimeout(() => {
      setLoading(true);
      setError('');
      void window.arc.assetSources
        .search(source.id, {
          text: search.trim() || undefined,
          kinds: kind === 'all' ? undefined : [kind],
          limit: 160,
        })
        .then((next) => {
          if (revision === searchRevision.current) setResult(next);
        })
        .catch((reason) => {
          if (revision === searchRevision.current) setError(remoteAssetFailureMessage('search', reason));
        })
        .finally(() => {
          if (revision === searchRevision.current) setLoading(false);
        });
    }, 250);
    return () => window.clearTimeout(timeout);
  }, [kind, search, source.id]);

  useEffect(() => {
    ++manifestRevision.current;
    const operationId = activeOperation.current;
    if (operationId !== null) {
      window.arc.assetSources.cancelImport(operationId);
      activeOperation.current = null;
    }
    setSelected(null);
    setManifest(null);
    setManifestLoading(false);
    setImportProgress(null);
    setImportMessage('');
    setImportLifecycle(initialRemoteAssetImportLifecycle());
  }, [source.id]);

  useEffect(
    () => () => {
      const operationId = activeOperation.current;
      if (operationId !== null) window.arc.assetSources.cancelImport(operationId);
    },
    [],
  );

  const selectAsset = (asset: ArcRemoteAsset) => {
    const revision = ++manifestRevision.current;
    setSelected(asset);
    setManifest(null);
    setResolution('');
    setFormat('');
    setImportProgress(null);
    setImportMessage('');
    setManifestLoading(true);
    void window.arc.assetSources
      .manifest(source.id, asset.id)
      .then((next) => {
        if (revision !== manifestRevision.current) return;
        setManifest(next);
        const resolutions = manifestResolutions(next);
        const formats = manifestFormats(next, asset.kind);
        setResolution(preferredResolution(resolutions));
        setFormat(preferredFormat(formats, asset.kind));
      })
      .catch((reason) => {
        if (revision === manifestRevision.current) setImportMessage(remoteAssetFailureMessage('manifest', reason));
      })
      .finally(() => {
        if (revision === manifestRevision.current) setManifestLoading(false);
      });
  };

  const resolutions = useMemo(() => (manifest ? manifestResolutions(manifest) : []), [manifest]);
  const formats = useMemo(
    () => (manifest ? manifestFormats(manifest, selected?.kind) : []),
    [manifest, selected?.kind],
  );
  const selectedFiles = useMemo(
    () => (manifest ? selectManifestFiles(manifest, resolution, format) : []),
    [format, manifest, resolution],
  );
  const selectionBytes = useMemo(() => manifestSelectionBytes(selectedFiles), [selectedFiles]);

  const importSelected = () => {
    if (!selected || selectedFiles.length === 0 || importing) return;
    const operationId = window.arc.assetSources.createImportOperation();
    activeOperation.current = operationId;
    setImportLifecycle((current) => beginRemoteAssetImport(current, operationId));
    setImportMessage('');
    setImportProgress(null);
    void window.arc.assetSources
      .importToProject(
        {
          sourceId: source.id,
          assetId: selected.id,
          logicalPaths: selectedFiles.map((file) => file.logicalPath),
          destinationScope: 'project',
          operationId,
        },
        (progress) => {
          if (activeOperation.current !== operationId) return;
          setImportProgress(progress);
          setImportLifecycle((current) =>
            updateRemoteAssetImportProgress(current, operationId, {
              bytesReceived: progress.completedBytes,
              totalBytes: progress.totalBytes,
            }),
          );
        },
      )
      .then((imported) => {
        if (activeOperation.current !== operationId) return;
        setImportLifecycle((current) => completeRemoteAssetImport(current, operationId));
        if (imported.succeeded) onImportedFiles?.(imported.importedFiles);
        setImportMessage(
          `Imported ${imported.importedFiles.length} files · ${imported.cacheHits} cache hits · ${imported.downloadedFiles} downloaded`,
        );
      })
      .catch((reason) => {
        if (activeOperation.current !== operationId) return;
        if (isCancellationReason(reason)) {
          setImportLifecycle((current) => cancelRemoteAssetImport(current, operationId));
          setImportMessage('Import canceled. No project files were kept.');
          return;
        }
        const message = remoteAssetFailureMessage('import', reason);
        setImportLifecycle((current) => failRemoteAssetImport(current, operationId, message));
        setImportMessage(message);
      })
      .finally(() => {
        if (activeOperation.current === operationId) activeOperation.current = null;
      });
  };

  const cancelSelectedImport = () => {
    const operationId = activeOperation.current;
    if (operationId === null || !importing) return;
    setImportLifecycle((current) => requestRemoteAssetImportCancellation(current));
    setImportMessage('Canceling import…');
    window.arc.assetSources.cancelImport(operationId);
  };

  return (
    <>
      <div className="content-browser-main remote-asset-browser">
        <header className="content-browser-v2-toolbar remote-asset-toolbar">
          <strong>{source.displayName}</strong>
          {source.attribution && <span className="remote-source-attribution">{source.attribution}</span>}
          <label>
            <Search size={13} />
            <input
              aria-label="Search online assets"
              placeholder={`Search ${source.displayName}`}
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </label>
          <select
            aria-label="Online asset type"
            value={kind}
            onChange={(event) => setKind(event.target.value as typeof kind)}
          >
            <option value="all">All types</option>
            <option value="hdri">HDRIs</option>
            <option value="texture">Textures</option>
            <option value="model">Models</option>
          </select>
          <button className={view === 'grid' ? 'active' : ''} aria-label="Grid view" onClick={() => setView('grid')}>
            <Grid2X2 size={14} />
          </button>
          <button className={view === 'list' ? 'active' : ''} aria-label="List view" onClick={() => setView('list')}>
            <List size={14} />
          </button>
        </header>
        <div className="remote-result-summary">
          {loading
            ? 'Loading assets…'
            : error
              ? error
              : `${result?.total ?? 0} assets · ${source.licenseSummary ?? 'License varies'}`}
        </div>
        <div className={`content-assets remote-assets ${view}`} role="listbox">
          {(result?.assets ?? []).map((asset) => (
            <button
              key={asset.id}
              className={`content-asset remote-asset ${selected?.id === asset.id ? 'selected' : ''}`}
              onClick={() => selectAsset(asset)}
            >
              <span className="content-asset-preview remote-asset-preview">
                {asset.thumbnailUrl ? (
                  <img src={asset.thumbnailUrl} alt="" loading="lazy" />
                ) : (
                  <span className="remote-no-preview">No preview</span>
                )}
              </span>
              <span className="content-asset-name">{asset.name}</span>
              <small>
                {asset.kind} · {asset.category || 'uncategorized'}
              </small>
            </button>
          ))}
          {!loading && !error && (result?.assets.length ?? 0) === 0 && (
            <div className="content-empty">No online assets match this search.</div>
          )}
        </div>
      </div>
      {selected && (
        <aside className="content-asset-details remote-asset-details">
          {selected.thumbnailUrl && <img className="remote-detail-image" src={selected.thumbnailUrl} alt="" />}
          <strong>{selected.name}</strong>
          <span>
            {selected.kind} · {selected.category}
          </span>
          {selected.description && <p>{selected.description}</p>}
          <span>{selected.license}</span>
          {selected.attribution && <small>{selected.attribution}</small>}
          {manifestLoading ? (
            <span>Loading download variants…</span>
          ) : manifest ? (
            <>
              {resolutions.length > 0 && (
                <label>
                  Resolution
                  <select
                    aria-label="Remote asset resolution"
                    value={resolution}
                    onChange={(event) => setResolution(event.target.value)}
                  >
                    {resolutions.map((value) => (
                      <option key={value}>{value}</option>
                    ))}
                  </select>
                </label>
              )}
              {formats.length > 0 && (
                <label>
                  Format
                  <select
                    aria-label="Remote asset format"
                    value={format}
                    onChange={(event) => setFormat(event.target.value)}
                  >
                    {formats.map((value) => (
                      <option key={value}>{value}</option>
                    ))}
                  </select>
                </label>
              )}
              <div className="remote-import-summary">
                <span>{selectedFiles.length} files</span>
                <span>{formatBytes(selectionBytes)}</span>
              </div>
              <div className="remote-import-actions">
                <button
                  className="remote-import-button"
                  disabled={selectedFiles.length === 0 || importing}
                  onClick={importSelected}
                >
                  <Download size={14} /> {importing ? (canceling ? 'Canceling…' : 'Importing…') : 'Import to Project'}
                </button>
                {importing && (
                  <button className="remote-import-cancel-button" disabled={canceling} onClick={cancelSelectedImport}>
                    <X size={14} /> {canceling ? 'Canceling…' : 'Cancel'}
                  </button>
                )}
              </div>
              {importProgress && (
                <progress
                  max={importProgress.totalBytes ?? Math.max(importProgress.totalFiles, 1)}
                  value={importProgress.totalBytes ? importProgress.completedBytes : importProgress.completedFiles}
                />
              )}
              {importProgress && <small>{progressLabel(importProgress)}</small>}
            </>
          ) : null}
          {importMessage && <span className="remote-import-message">{importMessage}</span>}
        </aside>
      )}
    </>
  );
}

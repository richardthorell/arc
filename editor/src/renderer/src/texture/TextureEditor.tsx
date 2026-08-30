import { useEffect, useMemo, useState } from 'react';
import { Image, Maximize2 } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import type { AssetItem } from '../services/editorHostTypes';

import '../inspector/inspector.css';
import './textureEditor.css';

type HostResponse<T = unknown> = {
  succeeded: boolean;
  payload?: T;
};

type HostAssetThumbnailSnapshot = {
  path: string;
  width: number;
  height: number;
  dataUrl: string;
};

const extensionOf = (path: string) => {
  const fileName = path.replaceAll('\\', '/').split('/').at(-1) ?? path;
  const dot = fileName.lastIndexOf('.');
  return dot > 0 && dot < fileName.length - 1 ? fileName.slice(dot + 1).toLocaleUpperCase() : 'Unknown';
};

const formatBytes = (bytes: number | undefined) => {
  if (bytes === undefined || !Number.isFinite(bytes) || bytes < 0) return 'Not reported';
  if (bytes < 1024) return `${bytes} B`;
  const units = ['KiB', 'MiB', 'GiB', 'TiB'];
  let value = bytes / 1024;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value >= 10 ? value.toFixed(1) : value.toFixed(2)} ${units[unit]}`;
};

const dimensionsOf = (asset: AssetItem) => {
  if (asset.width === undefined || asset.height === undefined) return 'Not reported';
  return `${asset.width} × ${asset.height}${asset.depth && asset.depth > 1 ? ` × ${asset.depth}` : ''}`;
};

function TextureProperty({ label, value }: { label: string; value: string }) {
  return (
    <div className="inspector-property texture-inspector-property">
      <span className="inspector-property-label">{label}</span>
      <span className="texture-inspector-value" title={value}>
        {value}
      </span>
    </div>
  );
}

function TextureInspector({ asset }: { asset: AssetItem }) {
  return (
    <aside aria-label="Texture details" className="data-inspector texture-inspector">
      <header className="texture-inspector-header">
        <Image aria-hidden="true" size={18} />
        <div>
          <strong>{asset.name}</strong>
          <small>Texture</small>
        </div>
      </header>

      <section className="texture-inspector-section">
        <h3>Texture</h3>
        <TextureProperty label="Dimensions" value={dimensionsOf(asset)} />
        <TextureProperty label="Format" value={extensionOf(asset.path)} />
        <TextureProperty
          label="Mip Levels"
          value={asset.mipLevels === undefined ? 'Not reported' : String(asset.mipLevels)}
        />
        <TextureProperty label="Source Size" value={formatBytes(asset.sourceBytes)} />
      </section>

      <section className="texture-inspector-section">
        <h3>Asset</h3>
        <TextureProperty label="Status" value={asset.status} />
        <TextureProperty label="Residency" value={asset.residency ?? 'Not reported'} />
        <TextureProperty label="Importer" value={asset.importerId ?? 'Not reported'} />
        <TextureProperty label="Scope" value={asset.scope ?? 'project'} />
        <TextureProperty label="Path" value={asset.path} />
        {asset.guid && <TextureProperty label="GUID" value={asset.guid} />}
      </section>
    </aside>
  );
}

export function TextureEditor({ document }: { document: EditorDocument }) {
  const asset = useMemo<AssetItem>(
    () =>
      document.assetSnapshot ?? {
        id: document.assetId ?? document.id,
        guid: document.assetGuid,
        name: document.title,
        path: document.path ?? '',
        scope: document.assetScope,
        kind: 'texture',
        status: 'unknown',
        readOnly: document.readOnly,
      },
    [document],
  );
  const [preview, setPreview] = useState<HostAssetThumbnailSnapshot | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);

  useEffect(() => {
    let active = true;
    setPreview(null);
    setPreviewFailed(false);
    if (!asset.path || typeof window === 'undefined' || !window.arc?.host?.query) {
      setPreviewFailed(true);
      return;
    }

    void (async () => {
      try {
        const response = (await window.arc.host.query('asset.thumbnail', {
          path: asset.path,
          maxSize: 2048,
        })) as HostResponse<HostAssetThumbnailSnapshot>;
        if (!active) return;
        if (response.succeeded && response.payload?.dataUrl) setPreview(response.payload);
        else setPreviewFailed(true);
      } catch {
        if (active) setPreviewFailed(true);
      }
    })();

    return () => {
      active = false;
    };
  }, [asset.path, asset.generation]);

  const previewDimensions =
    preview?.width && preview?.height
      ? `${preview.width} × ${preview.height}`
      : asset.width !== undefined && asset.height !== undefined
        ? `${asset.width} × ${asset.height}`
        : null;

  return (
    <section className="texture-editor">
      <main className="texture-preview-pane">
        <div className="texture-preview-meta">
          <span>{extensionOf(asset.path)}</span>
          {previewDimensions && <span>{previewDimensions}</span>}
          {asset.mipLevels !== undefined && <span>{asset.mipLevels} mips</span>}
          <span className={`texture-status texture-status-${asset.status}`}>{asset.status}</span>
        </div>
        <div className="texture-preview-stage">
          {preview?.dataUrl && !previewFailed ? (
            <img alt={`${asset.name} texture preview`} draggable={false} src={preview.dataUrl} />
          ) : previewFailed ? (
            <div className="texture-preview-empty">
              <Image aria-hidden="true" size={34} />
              <strong>Preview unavailable</strong>
              <span>The texture metadata is still available in the details panel.</span>
            </div>
          ) : (
            <div className="texture-preview-empty">
              <Maximize2 aria-hidden="true" size={30} />
              <strong>Loading texture…</strong>
            </div>
          )}
        </div>
      </main>
      <TextureInspector asset={asset} />
    </section>
  );
}

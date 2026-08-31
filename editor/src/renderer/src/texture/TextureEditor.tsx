import { useEffect, useMemo, useRef, useState } from 'react';
import type { WheelEvent } from 'react';
import { Image, Maximize2 } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { UiPanel } from '../ui';

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

type RulerMark = {
  value: number;
  position: number;
  major: boolean;
};

const minZoom = 0.05;
const maxZoom = 16;

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

const clampZoom = (value: number) => Math.min(maxZoom, Math.max(minZoom, value));

const rulerInterval = (zoom: number) => {
  const candidates = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000];
  return candidates.find((candidate) => candidate * zoom >= 42) ?? candidates.at(-1)!;
};

const rulerMarks = (size: number, zoom: number): RulerMark[] => {
  const majorInterval = rulerInterval(zoom);
  const minorInterval = majorInterval / 5;
  const count = Math.ceil(size / minorInterval);
  return Array.from({ length: count + 1 }, (_, index) => {
    const value = Math.min(size, index * minorInterval);
    return {
      value,
      position: value * zoom,
      major: index % 5 === 0,
    };
  });
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
    <UiPanel aria-label="Texture details" className="texture-inspector" role="complementary" variant="inspector">
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
    </UiPanel>
  );
}

function HorizontalRuler({ width, zoom }: { width: number; zoom: number }) {
  const marks = rulerMarks(width, zoom);
  return (
    <div aria-hidden="true" className="texture-ruler texture-ruler-horizontal" style={{ width: width * zoom }}>
      {marks.map((mark) => (
        <span
          className={mark.major ? 'texture-ruler-mark major' : 'texture-ruler-mark'}
          key={`${mark.value}-${mark.position}`}
          style={{ left: mark.position }}
        >
          {mark.major && <em>{Math.round(mark.value)}</em>}
        </span>
      ))}
    </div>
  );
}

function VerticalRuler({ height, zoom }: { height: number; zoom: number }) {
  const marks = rulerMarks(height, zoom);
  return (
    <div aria-hidden="true" className="texture-ruler texture-ruler-vertical" style={{ height: height * zoom }}>
      {marks.map((mark) => (
        <span
          className={mark.major ? 'texture-ruler-mark major' : 'texture-ruler-mark'}
          key={`${mark.value}-${mark.position}`}
          style={{ top: mark.position }}
        >
          {mark.major && <em>{Math.round(mark.value)}</em>}
        </span>
      ))}
    </div>
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
  const stageRef = useRef<HTMLDivElement | null>(null);
  const [preview, setPreview] = useState<HostAssetThumbnailSnapshot | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [zoom, setZoom] = useState(1);

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

  useEffect(() => {
    const stage = stageRef.current;
    if (!stage || !preview) return;

    const fitPreview = () => {
      const availableWidth = Math.max(1, stage.clientWidth - 80);
      const availableHeight = Math.max(1, stage.clientHeight - 80);
      setZoom(clampZoom(Math.min(1, availableWidth / preview.width, availableHeight / preview.height)));
    };

    fitPreview();
    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(fitPreview);
    observer.observe(stage);
    return () => observer.disconnect();
  }, [preview?.path, preview?.width, preview?.height]);

  const previewDimensions =
    preview?.width && preview?.height
      ? `${preview.width} × ${preview.height}`
      : asset.width !== undefined && asset.height !== undefined
        ? `${asset.width} × ${asset.height}`
        : null;

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!preview) return;
    event.preventDefault();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    setZoom((current) => clampZoom(current * factor));
  };

  return (
    <section className="texture-editor">
      <main className="texture-preview-pane">
        <div className="texture-preview-meta">
          <span>{extensionOf(asset.path)}</span>
          {previewDimensions && <span>{previewDimensions}</span>}
          {asset.mipLevels !== undefined && <span>{asset.mipLevels} mips</span>}
          <span className="texture-zoom">{Math.round(zoom * 100)}%</span>
          <span className={`texture-status texture-status-${asset.status}`}>{asset.status}</span>
        </div>
        <div className="texture-preview-stage" onWheel={onWheel} ref={stageRef}>
          {preview?.dataUrl && !previewFailed ? (
            <div
              className="texture-preview-canvas"
              style={{
                gridTemplateColumns: `28px ${preview.width * zoom}px`,
                gridTemplateRows: `22px ${preview.height * zoom}px`,
              }}
            >
              <div aria-hidden="true" className="texture-ruler-corner" />
              <HorizontalRuler width={preview.width} zoom={zoom} />
              <VerticalRuler height={preview.height} zoom={zoom} />
              <div
                className="texture-preview-image-frame"
                style={{ width: preview.width * zoom, height: preview.height * zoom }}
              >
                <img
                  alt={`${asset.name} texture preview`}
                  draggable={false}
                  height={preview.height * zoom}
                  src={preview.dataUrl}
                  width={preview.width * zoom}
                />
              </div>
            </div>
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

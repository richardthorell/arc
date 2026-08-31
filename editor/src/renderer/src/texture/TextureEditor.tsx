import { useEffect, useMemo, useRef, useState } from 'react';
import type { PointerEvent as ReactPointerEvent, UIEvent, WheelEvent } from 'react';
import { Image, Maximize2 } from 'lucide-react';

import type { EditorDocument } from '../editors/editorTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { UiPanel, UiPanelSection } from '../ui';
import { setTextureEditorViewState, useTextureEditorViewState } from './textureEditorViewState';

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

type ViewportMetrics = {
  scrollLeft: number;
  scrollTop: number;
  width: number;
  height: number;
};

type PanState = {
  pointerId: number;
  startX: number;
  startY: number;
  scrollLeft: number;
  scrollTop: number;
};

const minZoom = 0.25;
const maxZoom = 16;
const previewPadding = 28;
const defaultInspectorWidth = 400;
const minInspectorWidth = 320;
const maxInspectorWidth = 680;

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

const textureTypeOf = (asset: AssetItem) => (asset.depth && asset.depth > 1 ? '3D Texture' : '2D Texture');
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
  const [collapsedSections, setCollapsedSections] = useState<Record<string, boolean>>({
    texture: false,
    sampling: false,
    mipmaps: false,
    compression: true,
    streaming: true,
    import: true,
    asset: true,
  });
  const toggleSection = (section: string) =>
    setCollapsedSections((current) => ({ ...current, [section]: !current[section] }));

  return (
    <UiPanel aria-label="Texture details" className="texture-inspector" role="complementary" variant="inspector">
      <div className="texture-inspector-sections">
        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.texture}
          onToggle={() => toggleSection('texture')}
          title="Texture"
        >
          <TextureProperty label="Type" value={textureTypeOf(asset)} />
          <TextureProperty label="Dimensions" value={dimensionsOf(asset)} />
          <TextureProperty label="Depth / Layers" value={asset.depth === undefined ? '1' : String(asset.depth)} />
          <TextureProperty label="Format" value={asset.textureFormat ?? extensionOf(asset.path)} />
          <TextureProperty label="Color Space / sRGB" value="Not reported" />
          <TextureProperty label="Alpha" value="Not reported" />
          <TextureProperty label="Source Size" value={formatBytes(asset.sourceBytes)} />
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.sampling}
          onToggle={() => toggleSection('sampling')}
          title="Sampling"
        >
          <TextureProperty label="Wrap U" value="Not configured" />
          <TextureProperty label="Wrap V" value="Not configured" />
          <TextureProperty label="Filter Mode" value="Not configured" />
          <TextureProperty label="Anisotropy" value="Not configured" />
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.mipmaps}
          onToggle={() => toggleSection('mipmaps')}
          title="Mipmaps"
        >
          <TextureProperty label="Generate Mips" value={(asset.mipLevels ?? 1) > 1 ? 'Yes' : 'No'} />
          <TextureProperty label="Mip Count" value={asset.mipLevels === undefined ? 'Not reported' : String(asset.mipLevels)} />
          <TextureProperty label="Generation Filter" value="Not configured" />
          <TextureProperty label="LOD Bias" value="Not configured" />
          <TextureProperty label="Min / Max LOD" value="Not configured" />
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.compression}
          onToggle={() => toggleSection('compression')}
          title="Compression"
        >
          <TextureProperty label="Compression Preset" value="Not configured" />
          <TextureProperty label="GPU Format" value={asset.textureFormat ?? 'Not reported'} />
          <TextureProperty label="Quality" value="Not configured" />
          <TextureProperty label="Alpha Policy" value="Not configured" />
          <TextureProperty label="Artifact Size" value={formatBytes(asset.artifactSize)} />
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.streaming}
          onToggle={() => toggleSection('streaming')}
          title="Streaming"
        >
          <TextureProperty label="Mode" value={asset.streamingMode ?? 'Not reported'} />
          <TextureProperty label="Residency" value={asset.residency ?? 'Not reported'} />
          <TextureProperty label="Tile Count" value={asset.tileCount === undefined ? 'Not reported' : String(asset.tileCount)} />
          <TextureProperty label="Priority" value="Not configured" />
          {asset.streamingEligibilityError && (
            <TextureProperty label="Eligibility" value={asset.streamingEligibilityError} />
          )}
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.import}
          onToggle={() => toggleSection('import')}
          title="Import"
        >
          <TextureProperty label="Importer" value={asset.importerId ?? 'Not reported'} />
          <TextureProperty label="Source Path" value={asset.path} />
          <TextureProperty
            label="Settings Version"
            value={asset.settingsVersion === undefined ? 'Not reported' : String(asset.settingsVersion)}
          />
          <TextureProperty label="Power-of-Two Policy" value="Not configured" />
        </UiPanelSection>

        <UiPanelSection
          className="texture-inspector-section"
          collapsed={collapsedSections.asset}
          onToggle={() => toggleSection('asset')}
          title="Asset"
        >
          <TextureProperty label="Status" value={asset.status} />
          <TextureProperty label="Scope" value={asset.scope ?? 'project'} />
          <TextureProperty label="Path" value={asset.path} />
          {asset.guid && <TextureProperty label="GUID" value={asset.guid} />}
        </UiPanelSection>
      </div>
    </UiPanel>
  );
}

function HorizontalRuler({ width, zoom, offset }: { width: number; zoom: number; offset: number }) {
  const marks = rulerMarks(width, zoom);
  return (
    <div
      aria-hidden="true"
      className="texture-ruler texture-ruler-horizontal"
      style={{ width: width * zoom, transform: `translateX(${offset}px)` }}
    >
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

function VerticalRuler({ height, zoom, offset }: { height: number; zoom: number; offset: number }) {
  const marks = rulerMarks(height, zoom);
  return (
    <div
      aria-hidden="true"
      className="texture-ruler texture-ruler-vertical"
      style={{ height: height * zoom, transform: `translateY(${offset}px)` }}
    >
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
  const scrollRef = useRef<HTMLDivElement | null>(null);
  const panRef = useRef<PanState | null>(null);
  const resizeRef = useRef<{ startX: number; width: number } | null>(null);
  const [preview, setPreview] = useState<HostAssetThumbnailSnapshot | null>(null);
  const [previewFailed, setPreviewFailed] = useState(false);
  const [viewport, setViewport] = useState<ViewportMetrics>({ scrollLeft: 0, scrollTop: 0, width: 0, height: 0 });
  const [spaceHeld, setSpaceHeld] = useState(false);
  const [panning, setPanning] = useState(false);
  const [inspectorWidth, setInspectorWidth] = useState(defaultInspectorWidth);
  const viewState = useTextureEditorViewState(document.id);
  const zoom = viewState.zoom;
  const mipScale = 1 / 2 ** viewState.mipLevel;
  const displayWidth = Math.max(1, Math.round((preview?.width ?? 1) * mipScale));
  const displayHeight = Math.max(1, Math.round((preview?.height ?? 1) * mipScale));
  const renderedWidth = displayWidth * zoom;
  const renderedHeight = displayHeight * zoom;
  const canvasWidth = Math.max(viewport.width, renderedWidth + previewPadding * 2);
  const canvasHeight = Math.max(viewport.height, renderedHeight + previewPadding * 2);
  const imageLeft = Math.max(previewPadding, (canvasWidth - renderedWidth) / 2);
  const imageTop = Math.max(previewPadding, (canvasHeight - renderedHeight) / 2);

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
    const scroll = scrollRef.current;
    if (!scroll || !preview) return;

    const updateViewport = () => {
      setViewport({
        scrollLeft: scroll.scrollLeft,
        scrollTop: scroll.scrollTop,
        width: scroll.clientWidth,
        height: scroll.clientHeight,
      });
    };

    updateViewport();
    const availableWidth = Math.max(1, scroll.clientWidth - previewPadding * 2);
    const availableHeight = Math.max(1, scroll.clientHeight - previewPadding * 2);
    setTextureEditorViewState(document.id, {
      zoom: clampZoom(Math.min(1, availableWidth / displayWidth, availableHeight / displayHeight)),
    });

    if (typeof ResizeObserver === 'undefined') return;
    const observer = new ResizeObserver(updateViewport);
    observer.observe(scroll);
    return () => observer.disconnect();
  }, [displayHeight, displayWidth, document.id, preview?.path]);

  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.code === 'Space' && !event.repeat) {
        setSpaceHeld(true);
        if (event.target instanceof HTMLElement && !['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target.tagName)) {
          event.preventDefault();
        }
      }
    };
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.code === 'Space') setSpaceHeld(false);
    };
    window.addEventListener('keydown', onKeyDown);
    window.addEventListener('keyup', onKeyUp);
    window.addEventListener('blur', () => setSpaceHeld(false), { once: true });
    return () => {
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    };
  }, []);

  const onWheel = (event: WheelEvent<HTMLDivElement>) => {
    if (!preview) return;
    event.preventDefault();
    event.stopPropagation();
    const factor = event.deltaY < 0 ? 1.12 : 1 / 1.12;
    setTextureEditorViewState(document.id, { zoom: clampZoom(zoom * factor) });
  };

  const onScroll = (event: UIEvent<HTMLDivElement>) => {
    const target = event.currentTarget;
    setViewport((current) => ({
      ...current,
      scrollLeft: target.scrollLeft,
      scrollTop: target.scrollTop,
      width: target.clientWidth,
      height: target.clientHeight,
    }));
  };

  const beginPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const shouldPan = event.button === 1 || (event.button === 0 && spaceHeld);
    if (!shouldPan || !scrollRef.current) return;
    event.preventDefault();
    event.currentTarget.setPointerCapture(event.pointerId);
    panRef.current = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      scrollLeft: scrollRef.current.scrollLeft,
      scrollTop: scrollRef.current.scrollTop,
    };
    setPanning(true);
  };

  const movePan = (event: ReactPointerEvent<HTMLDivElement>) => {
    const pan = panRef.current;
    const scroll = scrollRef.current;
    if (!pan || pan.pointerId !== event.pointerId || !scroll) return;
    scroll.scrollLeft = pan.scrollLeft - (event.clientX - pan.startX);
    scroll.scrollTop = pan.scrollTop - (event.clientY - pan.startY);
  };

  const endPan = (event: ReactPointerEvent<HTMLDivElement>) => {
    if (panRef.current?.pointerId !== event.pointerId) return;
    panRef.current = null;
    setPanning(false);
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  const beginResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    resizeRef.current = { startX: event.clientX, width: inspectorWidth };
    event.currentTarget.setPointerCapture(event.pointerId);
  };

  const moveResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    const resize = resizeRef.current;
    if (!resize) return;
    const width = resize.width - (event.clientX - resize.startX);
    setInspectorWidth(Math.min(maxInspectorWidth, Math.max(minInspectorWidth, width)));
  };

  const endResize = (event: ReactPointerEvent<HTMLDivElement>) => {
    resizeRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) event.currentTarget.releasePointerCapture(event.pointerId);
  };

  return (
    <section
      className="texture-editor"
      style={{ gridTemplateColumns: `minmax(0, 1fr) 6px ${inspectorWidth}px` }}
    >
      <main className="texture-preview-pane">
        <div
          className={`texture-preview-stage ${spaceHeld ? 'is-pan-ready' : ''} ${panning ? 'is-panning' : ''}`}
          onPointerCancel={endPan}
          onPointerDown={beginPan}
          onPointerMove={movePan}
          onPointerUp={endPan}
          onWheel={onWheel}
        >
          <div aria-hidden="true" className="texture-ruler-corner" />
          <div aria-hidden="true" className="texture-ruler-viewport texture-ruler-horizontal-viewport">
            {preview && (
              <HorizontalRuler width={displayWidth} zoom={zoom} offset={imageLeft - viewport.scrollLeft} />
            )}
          </div>
          <div aria-hidden="true" className="texture-ruler-viewport texture-ruler-vertical-viewport">
            {preview && <VerticalRuler height={displayHeight} zoom={zoom} offset={imageTop - viewport.scrollTop} />}
          </div>
          <div className="texture-preview-scroll" onScroll={onScroll} ref={scrollRef}>
            {preview?.dataUrl && !previewFailed ? (
              <div className="texture-preview-canvas" style={{ width: canvasWidth, height: canvasHeight }}>
                <div
                  className="texture-preview-image-frame"
                  style={{
                    left: imageLeft,
                    top: imageTop,
                    width: renderedWidth,
                    height: renderedHeight,
                  }}
                >
                  <svg aria-hidden="true" className="texture-channel-filter-defs">
                    <filter id={`texture-channel-filter-${document.id.replace(/[^a-zA-Z0-9_-]/g, '-')}`}>
                      <feColorMatrix
                        type="matrix"
                        values={`${viewState.channels.r ? 1 : 0} 0 0 0 0  0 ${viewState.channels.g ? 1 : 0} 0 0 0  0 0 ${viewState.channels.b ? 1 : 0} 0 0  0 0 0 ${viewState.channels.a ? 1 : 0} ${viewState.channels.a ? 0 : 1}`}
                      />
                    </filter>
                  </svg>
                  <img
                    alt={`${asset.name} texture preview`}
                    draggable={false}
                    height={renderedHeight}
                    src={preview.dataUrl}
                    style={{ filter: `url(#texture-channel-filter-${document.id.replace(/[^a-zA-Z0-9_-]/g, '-')})` }}
                    width={renderedWidth}
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
        </div>
      </main>
      <div
        aria-label="Resize texture details"
        className="texture-inspector-resizer"
        onPointerCancel={endResize}
        onPointerDown={beginResize}
        onPointerMove={moveResize}
        onPointerUp={endResize}
        role="separator"
      />
      <TextureInspector asset={asset} />
    </section>
  );
}

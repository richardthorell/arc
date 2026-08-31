import { useEffect, useMemo, useRef, useState } from 'react';
import { File, FileCode2, FileImage, FileType2, Layers3, Star } from 'lucide-react';

import { loadMaterialSphereThumbnail } from '../assets/materialThumbnail';
import { trackEditorJob } from '../jobs/editorJobProgress';
import type { AssetItem, AssetThumbnailProvider } from './contentBrowserTypes';

import './contentAssetCard.css';

const TOOLTIP_DELAY_MS = 550;

type HoverPoint = { x: number; y: number };

const fallbackIcon = (kind: AssetItem['kind']) => {
  if (kind === 'texture') return <FileImage size={32} strokeWidth={1.4} />;
  if (kind === 'material') return <Layers3 size={32} strokeWidth={1.4} />;
  if (kind === 'shader') return <FileCode2 size={32} strokeWidth={1.4} />;
  if (kind === 'scene') return <FileType2 size={32} strokeWidth={1.4} />;
  return <File size={32} strokeWidth={1.4} />;
};

export function ContentAssetCard({
  asset,
  favorite,
  selected,
  thumbnailProvider,
  onActivate,
  onFavorite,
  onReimport,
  onSelect,
}: {
  asset: AssetItem;
  favorite: boolean;
  selected: boolean;
  thumbnailProvider: AssetThumbnailProvider;
  onActivate: () => void;
  onFavorite: () => void;
  onReimport: () => void;
  onSelect: (additive: boolean) => void;
}) {
  const tooltipId = `asset-details-${asset.id.replace(/[^a-z0-9_-]/gi, '-')}`;
  const cardRef = useRef<HTMLDivElement>(null);
  const tooltipRef = useRef<HTMLDivElement>(null);
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const [detailsVisible, setDetailsVisible] = useState(false);
  const [hoverPoint, setHoverPoint] = useState<HoverPoint>({ x: 0, y: 0 });
  const [tooltipPosition, setTooltipPosition] = useState({ left: 0, top: 0 });
  const cardThumbnailProvider = useMemo<AssetThumbnailProvider>(() => {
    if (asset.kind !== 'material' || !asset.guid) return thumbnailProvider;
    return () =>
      trackEditorJob(
        'Rendering material thumbnail',
        () =>
          loadMaterialSphereThumbnail({
            guid: asset.guid!,
            generation: asset.generation,
            maxSize: 128,
          }),
        { priority: 'background' },
      );
  }, [asset.generation, asset.guid, asset.kind, thumbnailProvider]);

  const cancelHover = () => {
    if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
    hoverTimerRef.current = null;
  };

  const scheduleHover = (point: HoverPoint) => {
    cancelHover();
    setHoverPoint(point);
    hoverTimerRef.current = setTimeout(() => {
      hoverTimerRef.current = null;
      setDetailsVisible(true);
    }, TOOLTIP_DELAY_MS);
  };

  useEffect(() => {
    if (!detailsVisible) return;
    const tooltip = tooltipRef.current;
    if (!tooltip) return;
    const rect = tooltip.getBoundingClientRect();
    const margin = 8;
    const left = Math.max(margin, Math.min(hoverPoint.x + 12, window.innerWidth - rect.width - margin));
    const top = Math.max(margin, Math.min(hoverPoint.y + 12, window.innerHeight - rect.height - margin));
    setTooltipPosition({ left, top });
  }, [detailsVisible, hoverPoint]);

  useEffect(
    () => () => {
      cancelHover();
    },
    [],
  );

  return (
    <div
      aria-describedby={detailsVisible ? tooltipId : undefined}
      className={`content-asset ${selected ? 'selected' : ''}`}
      draggable
      onClick={(event) => onSelect(event.metaKey || event.ctrlKey || event.shiftKey)}
      onDoubleClick={onActivate}
      onDragStart={(event) => {
        event.dataTransfer.setData('text/arc-asset-id', asset.id);
        event.dataTransfer.effectAllowed = 'copyMove';
      }}
      onMouseEnter={(event) => scheduleHover({ x: event.clientX, y: event.clientY })}
      onMouseLeave={() => {
        cancelHover();
        setDetailsVisible(false);
      }}
      onMouseMove={(event) => {
        if (!detailsVisible) setHoverPoint({ x: event.clientX, y: event.clientY });
      }}
      ref={cardRef}
      role="option"
      aria-selected={selected}
      tabIndex={0}
      onKeyDown={(event) => {
        if (event.key === 'Enter') onActivate();
        if (event.key === ' ') {
          event.preventDefault();
          onSelect(event.metaKey || event.ctrlKey || event.shiftKey);
        }
      }}
    >
      <span className="content-asset-preview">
        <AssetThumbnail asset={asset} provider={cardThumbnailProvider} />
      </span>
      <span className="content-asset-name">{asset.name}</span>
      <small>{asset.kind}</small>
      <button
        aria-label={favorite ? `Remove ${asset.name} from favorites` : `Add ${asset.name} to favorites`}
        className={`content-asset-favorite ${favorite ? 'active' : ''}`}
        onClick={(event) => {
          event.stopPropagation();
          onFavorite();
        }}
        type="button"
      >
        <Star size={13} fill={favorite ? 'currentColor' : 'none'} />
      </button>
      {detailsVisible && (
        <div
          className="content-asset-tooltip"
          id={tooltipId}
          ref={tooltipRef}
          role="tooltip"
          style={{ left: tooltipPosition.left, top: tooltipPosition.top }}
        >
          <strong>{asset.name}</strong>
          <span>{asset.kind}</span>
          {asset.path && <span>{asset.path}</span>}
          {asset.status && <span>Status: {asset.status}</span>}
          {asset.kind === 'texture' && (
            <button
              onClick={(event) => {
                event.stopPropagation();
                setDetailsVisible(false);
                onReimport();
              }}
              type="button"
            >
              Reimport
            </button>
          )}
        </div>
      )}
    </div>
  );
}

function AssetThumbnail({ asset, provider }: { asset: AssetItem; provider: AssetThumbnailProvider }) {
  const [source, setSource] = useState<string | null>(asset.thumbnail ?? null);
  const [visible, setVisible] = useState(false);
  const containerRef = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const element = containerRef.current;
    if (!element) return;
    if (typeof IntersectionObserver === 'undefined') {
      setVisible(true);
      return;
    }
    const observer = new IntersectionObserver(
      (entries) => {
        if (!entries.some((entry) => entry.isIntersecting)) return;
        setVisible(true);
        observer.disconnect();
      },
      { rootMargin: '96px' },
    );
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!visible) return;
    let cancelled = false;
    void provider(asset).then((thumbnail) => {
      if (!cancelled && thumbnail) setSource(thumbnail);
    });
    return () => {
      cancelled = true;
    };
  }, [asset, provider, visible]);

  return (
    <span className="content-asset-thumbnail" ref={containerRef}>
      {source ? <img alt="" draggable={false} src={source} /> : fallbackIcon(asset.kind)}
      {asset.status === 'importing' && <i />}
    </span>
  );
}

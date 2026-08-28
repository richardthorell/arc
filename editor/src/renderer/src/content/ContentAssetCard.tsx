import { useState } from 'react';
import { Star } from 'lucide-react';

import { AssetThumbnail } from '../inspector/AssetPicker';
import type { AssetThumbnailProvider } from '../inspector/AssetPicker';
import type { AssetItem } from '../services/editorHostTypes';
import { UiFloatingSurface } from '../ui';

const assetTypeLabels: Record<AssetItem['kind'], string> = {
  scene: 'Scene',
  mesh: 'Mesh',
  material: 'Material',
  texture: 'Texture',
  shader: 'Shader',
  prefab: 'Prefab',
  folder: 'Folder',
};

const fileNameFromPath = (path: string) => path.replaceAll('\\', '/').split('/').at(-1) ?? path;

export const assetFileExtension = (asset: AssetItem) => {
  const fileName = fileNameFromPath(asset.path);
  const dot = fileName.lastIndexOf('.');
  return dot > 0 && dot < fileName.length - 1 ? fileName.slice(dot + 1).toLocaleLowerCase() : '';
};

export const assetDisplayName = (asset: AssetItem) => {
  const extension = assetFileExtension(asset);
  const suffix = extension ? `.${extension}` : '';
  const name = asset.name.trim() || fileNameFromPath(asset.path);
  return suffix && name.toLocaleLowerCase().endsWith(suffix) ? name.slice(0, -suffix.length) : name;
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

const formatCount = (value: number | undefined) =>
  value === undefined || !Number.isFinite(value) ? null : Math.max(0, Math.round(value)).toLocaleString();

function AssetDetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="content-asset-hover-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function AssetHoverDetails({ asset }: { asset: AssetItem }) {
  const extension = assetFileExtension(asset);
  const dimensions =
    asset.width !== undefined && asset.height !== undefined
      ? `${asset.width} × ${asset.height}${asset.depth && asset.depth > 1 ? ` × ${asset.depth}` : ''}`
      : null;
  const vertices = formatCount(asset.vertexCount);
  const triangles = formatCount(asset.triangleCount);

  return (
    <UiFloatingSurface className="content-asset-hover" role="tooltip" width={276}>
      <header>
        <strong>{assetDisplayName(asset)}</strong>
        <span>{assetTypeLabels[asset.kind]}</span>
      </header>
      <div className="content-asset-hover-section">
        <AssetDetailRow label="Size" value={formatBytes(asset.sourceBytes)} />
        <AssetDetailRow label="Extension" value={extension ? `.${extension}` : 'None'} />
        <AssetDetailRow label="Status" value={asset.status} />
        <AssetDetailRow label="Path" value={asset.path} />
      </div>
      {(dimensions || asset.mipLevels !== undefined || vertices || triangles) && (
        <div className="content-asset-hover-section asset-specific">
          {dimensions && <AssetDetailRow label="Dimensions" value={dimensions} />}
          {asset.mipLevels !== undefined && <AssetDetailRow label="Mip levels" value={String(asset.mipLevels)} />}
          {vertices && <AssetDetailRow label="Vertices" value={vertices} />}
          {triangles && <AssetDetailRow label="Triangles" value={triangles} />}
        </div>
      )}
      {(asset.importerId || asset.residency || asset.readOnly) && (
        <div className="content-asset-hover-section secondary">
          {asset.importerId && <AssetDetailRow label="Importer" value={asset.importerId} />}
          {asset.residency && <AssetDetailRow label="Residency" value={asset.residency} />}
          {asset.readOnly && <AssetDetailRow label="Source" value="Engine · Read-only" />}
        </div>
      )}
    </UiFloatingSurface>
  );
}

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
  const [detailsVisible, setDetailsVisible] = useState(false);

  return (
    <div
      aria-describedby={detailsVisible ? tooltipId : undefined}
      aria-selected={selected}
      className={`content-asset ${selected ? 'selected' : ''}`}
      draggable={Boolean(asset.guid)}
      role="option"
      tabIndex={0}
      onBlur={(event) => {
        if (!event.relatedTarget || !event.currentTarget.contains(event.relatedTarget as Node))
          setDetailsVisible(false);
      }}
      onClick={(event) => onSelect(event.ctrlKey || event.metaKey)}
      onDoubleClick={onActivate}
      onDragStart={(event) => {
        event.dataTransfer.setData(
          'application/x-arc-asset',
          JSON.stringify({ guid: asset.guid ?? '', type: asset.kind, pathHint: asset.path }),
        );
        event.dataTransfer.effectAllowed = 'copy';
      }}
      onFocus={() => setDetailsVisible(true)}
      onKeyDown={(event) => {
        if (event.target !== event.currentTarget || (event.key !== 'Enter' && event.key !== ' ')) return;
        event.preventDefault();
        onSelect(event.ctrlKey || event.metaKey);
      }}
      onMouseEnter={() => setDetailsVisible(true)}
      onMouseLeave={() => setDetailsVisible(false)}
    >
      <span className="content-asset-preview">
        <AssetThumbnail asset={asset} path={asset.path} provider={thumbnailProvider} />
      </span>
      <span className="content-asset-info">
        <span className="content-asset-name" title={assetDisplayName(asset)}>
          {assetDisplayName(asset)}
        </span>
        <small>{assetTypeLabels[asset.kind]}</small>
        <i aria-label={`Asset status: ${asset.status}`} className={`asset-state ${asset.status}`} />
      </span>
      <span className="content-asset-actions" onClick={(event) => event.stopPropagation()}>
        <button aria-label="Favorite" className={favorite ? 'active' : ''} onClick={onFavorite}>
          <Star size={12} />
        </button>
        {asset.guid && !asset.readOnly && <button onClick={onReimport}>Reimport</button>}
      </span>
      {detailsVisible && (
        <div id={tooltipId} className="content-asset-hover-anchor">
          <AssetHoverDetails asset={asset} />
        </div>
      )}
    </div>
  );
}

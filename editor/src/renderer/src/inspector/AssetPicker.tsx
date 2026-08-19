import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Image, Search, X } from 'lucide-react';
import { createPortal } from 'react-dom';

export type AssetPickerItem = {
  id: string;
  guid?: string;
  typeId?: string;
  name: string;
  path: string;
  kind: string;
  status: 'unknown' | 'queued' | 'ready' | 'dirty' | 'stale' | 'importing' | 'failed' | 'missing';
  scope?: 'builtin' | 'project' | 'user' | 'organization' | 'procedural';
  readOnly?: boolean;
};

export type AssetThumbnailProvider = (path: string) => Promise<string | null>;

export type AssetPickerProps = {
  assets: ReadonlyArray<AssetPickerItem>;
  value: string;
  label: string;
  assetKinds: ReadonlyArray<string>;
  assetTypeIds?: ReadonlyArray<string>;
  assetTypeLabel?: string;
  allowedExtensions?: ReadonlyArray<string>;
  allowEmpty?: boolean;
  mixed?: boolean;
  referenceMode?: 'path' | 'guid';
  thumbnailProvider?: AssetThumbnailProvider;
  onChange: (path: string) => void;
};

const thumbnailCaches = new WeakMap<AssetThumbnailProvider, Map<string, Promise<string | null>>>();
const extensionOf = (path: string) => {
  const index = path.lastIndexOf('.');
  return index >= 0 ? path.slice(index).toLocaleLowerCase() : '';
};
const basenameOf = (path: string) => path.split(/[\\/]/).pop() || path;
const displayNameOf = (asset?: AssetPickerItem, fallback = '') => {
  const raw = asset?.name || basenameOf(fallback);
  const extension = extensionOf(asset?.path || fallback);
  return extension && raw.toLocaleLowerCase().endsWith(extension)
    ? raw.slice(0, Math.max(0, raw.length - extension.length))
    : raw;
};
const sourceLabelOf = (asset: AssetPickerItem | undefined, assetTypeLabel: string) => {
  const scope =
    asset?.scope === 'procedural'
      ? 'Procedural'
      : asset?.scope === 'builtin'
        ? 'Engine'
        : asset?.scope === 'user'
          ? 'User'
          : asset?.scope === 'organization'
            ? 'Organization'
            : asset?.scope === 'project'
              ? 'Project'
              : '';
  return scope ? `${scope} ${assetTypeLabel}` : assetTypeLabel;
};

type PrimitiveMeshKind = 'plane' | 'cube' | 'sphere' | 'cylinder' | 'cone' | 'capsule';
const primitiveMeshPrefix = 'arc://primitive/';
const primitiveMeshKinds = new Set<PrimitiveMeshKind>(['plane', 'cube', 'sphere', 'cylinder', 'cone', 'capsule']);

const primitiveMeshKindOf = (asset: AssetPickerItem | undefined, path: string): PrimitiveMeshKind | null => {
  if (asset?.scope !== 'procedural' && !path.startsWith(primitiveMeshPrefix)) return null;
  const token = (path.startsWith(primitiveMeshPrefix) ? path.slice(primitiveMeshPrefix.length) : asset?.name || '')
    .trim()
    .toLocaleLowerCase() as PrimitiveMeshKind;
  return primitiveMeshKinds.has(token) ? token : null;
};

function PrimitiveMeshIcon({ kind }: { kind: PrimitiveMeshKind }) {
  const common = {
    fill: 'none',
    stroke: 'currentColor',
    strokeLinecap: 'round' as const,
    strokeLinejoin: 'round' as const,
    strokeWidth: 1.6,
  };
  const shape =
    kind === 'plane' ? (
      <>
        <path d="M5 17.5 14.5 9.5 27 14.5 17.5 22.5Z" />
        <path d="m5 17.5 12.5 5 9.5-8" opacity="0.42" />
      </>
    ) : kind === 'cube' ? (
      <>
        <path d="M16 4.5 27 10.5 16 16.5 5 10.5Z" />
        <path d="M5 10.5v12L16 28l11-5.5v-12M16 16.5V28" />
      </>
    ) : kind === 'sphere' ? (
      <>
        <circle cx="16" cy="16" r="11" />
        <ellipse cx="16" cy="16" rx="5" ry="11" />
        <path d="M5 16h22M7.5 10.5h17M7.5 21.5h17" opacity="0.55" />
      </>
    ) : kind === 'cylinder' ? (
      <>
        <ellipse cx="16" cy="7.5" rx="9" ry="4" />
        <path d="M7 7.5v17c0 2.2 4 4 9 4s9-1.8 9-4v-17" />
        <path d="M7 24.5c0 2.2 4 4 9 4s9-1.8 9-4" opacity="0.55" />
      </>
    ) : kind === 'cone' ? (
      <>
        <ellipse cx="16" cy="25" rx="10" ry="4" />
        <path d="M16 4 6 25M16 4l10 21" />
        <path d="M6 25c0 2.2 4.5 4 10 4s10-1.8 10-4" opacity="0.55" />
      </>
    ) : (
      <>
        <rect height="26" rx="7" width="14" x="9" y="3" />
        <path d="M9 16h14" opacity="0.45" />
      </>
    );

  return (
    <svg
      {...common}
      aria-hidden="true"
      data-testid={`primitive-mesh-icon-${kind}`}
      style={{ height: '68%', width: '68%' }}
      viewBox="0 0 32 32"
    >
      {shape}
    </svg>
  );
}

function thumbnailRequest(provider: AssetThumbnailProvider, path: string): Promise<string | null> {
  let cache = thumbnailCaches.get(provider);
  if (!cache) {
    cache = new Map<string, Promise<string | null>>();
    thumbnailCaches.set(provider, cache);
  }
  let request = cache.get(path);
  if (!request) {
    request = provider(path)
      .then((value) => {
        if (!value) cache?.delete(path);
        return value;
      })
      .catch(() => {
        cache?.delete(path);
        return null;
      });
    cache.set(path, request);
  }
  return request;
}

export function AssetPicker({
  assets,
  value,
  label,
  assetKinds,
  assetTypeIds,
  assetTypeLabel = 'Asset',
  allowedExtensions,
  allowEmpty = true,
  mixed = false,
  referenceMode = 'path',
  thumbnailProvider,
  onChange,
}: AssetPickerProps) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const anchorRef = useRef<HTMLButtonElement>(null);
  const candidates = useMemo(
    () =>
      assets.filter(
        (asset) =>
          assetKinds.includes(asset.kind) &&
          (!assetTypeIds?.length || Boolean(asset.typeId && assetTypeIds.includes(asset.typeId))) &&
          (asset.scope === 'procedural' ||
            !allowedExtensions?.length ||
            allowedExtensions.includes(extensionOf(asset.path))),
      ),
    [allowedExtensions, assetKinds, assetTypeIds, assets],
  );
  const valueFor = (asset: AssetPickerItem) => (referenceMode === 'guid' ? asset.guid || asset.id : asset.path);
  const selected = assets.find((asset) => valueFor(asset) === value);

  const acceptDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const path =
      event.dataTransfer.getData('application/x-arc-asset') ||
      event.dataTransfer.getData('application/x-arc-environment');
    const candidate = candidates.find((asset) => asset.path === path);
    if (candidate) onChange(valueFor(candidate));
  };

  return (
    <div className="inspector-property inspector-asset-property">
      <span className="inspector-property-label">{label}</span>
      <div className="asset-reference-control" onDragOver={(event) => event.preventDefault()} onDrop={acceptDrop}>
        <button
          aria-expanded={open}
          aria-label={`Choose ${label} asset`}
          className="asset-reference-main"
          onClick={() => setOpen((current) => !current)}
          ref={anchorRef}
          type="button"
        >
          <AssetThumbnail asset={selected} path={mixed ? '' : value} provider={thumbnailProvider} />
          <span className="asset-reference-copy">
            <strong>{mixed ? 'Mixed' : value ? displayNameOf(selected, value) : 'None'}</strong>
            <small>
              {mixed
                ? 'Choose an asset to replace all values'
                : selected
                  ? sourceLabelOf(selected, assetTypeLabel)
                  : value
                    ? assetTypeLabel
                    : `No ${assetTypeLabel.toLocaleLowerCase()} assigned`}
            </small>
          </span>
          <ChevronDown size={13} />
        </button>
        {allowEmpty && value && !mixed && (
          <button
            aria-label={`Clear ${label}`}
            className="asset-reference-clear"
            onClick={() => onChange('')}
            title="Clear asset reference"
            type="button"
          >
            <X size={13} />
          </button>
        )}
      </div>
      {open && (
        <AssetPickerPopover
          anchorRef={anchorRef}
          assets={candidates}
          assetTypeLabel={assetTypeLabel}
          filter={filter}
          label={label}
          selectedValue={value}
          thumbnailProvider={thumbnailProvider}
          onClose={() => setOpen(false)}
          onFilter={setFilter}
          valueFor={valueFor}
          onSelect={(asset) => {
            onChange(valueFor(asset));
            setOpen(false);
          }}
        />
      )}
    </div>
  );
}

export function TexturePicker(props: Omit<AssetPickerProps, 'assetKinds'>) {
  return <AssetPicker {...props} assetKinds={['texture', 'environment']} assetTypeLabel="Texture" />;
}

export function MaterialPicker(props: Omit<AssetPickerProps, 'assetKinds'>) {
  return <AssetPicker {...props} assetKinds={['material']} assetTypeLabel="Material" />;
}

export function PrefabPicker(props: Omit<AssetPickerProps, 'assetKinds'>) {
  return <AssetPicker {...props} assetKinds={['prefab']} assetTypeLabel="Prefab" />;
}

export function AssetPreview({
  path,
  name,
  label,
  provider,
}: {
  path: string;
  name: string;
  label: string;
  provider?: AssetThumbnailProvider;
}) {
  return (
    <div className="asset-preview-property">
      <div aria-label={label} className="asset-preview-stage">
        <AssetThumbnail path={path} provider={provider} />
        <span>
          <strong>{name || 'No Material'}</strong>
          <small>{path || 'Embedded runtime material'}</small>
        </span>
      </div>
    </div>
  );
}

function AssetPickerPopover({
  anchorRef,
  assets,
  assetTypeLabel,
  filter,
  label,
  selectedValue,
  thumbnailProvider,
  onClose,
  onFilter,
  onSelect,
  valueFor,
}: {
  anchorRef: React.RefObject<HTMLElement | null>;
  assets: ReadonlyArray<AssetPickerItem>;
  assetTypeLabel: string;
  filter: string;
  label: string;
  selectedValue: string;
  thumbnailProvider?: AssetThumbnailProvider;
  onClose: () => void;
  onFilter: (value: string) => void;
  onSelect: (asset: AssetPickerItem) => void;
  valueFor: (asset: AssetPickerItem) => string;
}) {
  const popoverRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ left: 8, top: 8 });
  const shown = assets.filter((asset) =>
    `${asset.name} ${asset.path}`.toLocaleLowerCase().includes(filter.trim().toLocaleLowerCase()),
  );

  useLayoutEffect(() => {
    const anchor = anchorRef.current?.getBoundingClientRect();
    if (!anchor) return;
    const width = 344;
    const left = Math.max(8, Math.min(anchor.left, window.innerWidth - width - 8));
    const below = anchor.bottom + 5;
    setPosition({ left, top: below + 420 > window.innerHeight ? Math.max(8, anchor.top - 425) : below });
  }, [anchorRef]);

  useEffect(() => {
    const outside = (event: PointerEvent) => {
      const target = event.target as Node;
      if (!popoverRef.current?.contains(target) && !anchorRef.current?.contains(target)) onClose();
    };
    const escape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose();
    };
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      document.removeEventListener('keydown', escape);
    };
  }, [anchorRef, onClose]);

  return createPortal(
    <section
      aria-label={`${label} asset picker`}
      className="asset-picker-popover"
      ref={popoverRef}
      role="dialog"
      style={{ left: position.left, top: position.top }}
    >
      <header>
        <strong>{`Select ${assetTypeLabel}`}</strong>
        <span>{shown.length} assets</span>
        <button aria-label="Close asset picker" onClick={onClose} type="button">
          <X size={14} />
        </button>
      </header>
      <label className="asset-picker-search">
        <Search size={14} />
        <input
          aria-label={`Search ${assetTypeLabel.toLocaleLowerCase()} assets`}
          autoFocus
          onChange={(event) => onFilter(event.target.value)}
          placeholder={`Search ${assetTypeLabel.toLocaleLowerCase()}s…`}
          value={filter}
        />
      </label>
      <div className="asset-picker-grid">
        {shown.map((asset) => (
          <button
            aria-label={`Select ${displayNameOf(asset)}`}
            className={valueFor(asset) === selectedValue ? 'is-selected' : ''}
            key={asset.id}
            onClick={() => onSelect(asset)}
            type="button"
          >
            <AssetThumbnail asset={asset} path={asset.path} provider={thumbnailProvider} />
            <strong>{displayNameOf(asset)}</strong>
            <small>
              {sourceLabelOf(asset, assetTypeLabel)} · {asset.status}
            </small>
          </button>
        ))}
        {!shown.length && (
          <div className="asset-picker-empty">
            <Image size={22} />
            <span>{`No matching ${assetTypeLabel.toLocaleLowerCase()}s`}</span>
          </div>
        )}
      </div>
    </section>,
    document.body,
  );
}

export function AssetThumbnail({
  asset,
  path,
  provider,
}: {
  asset?: AssetPickerItem;
  path: string;
  provider?: AssetThumbnailProvider;
}) {
  const elementRef = useRef<HTMLSpanElement>(null);
  const [source, setSource] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [visible, setVisible] = useState(() => typeof IntersectionObserver === 'undefined');
  const primitiveKind = primitiveMeshKindOf(asset, path);
  useEffect(() => {
    if (visible || typeof IntersectionObserver === 'undefined' || !elementRef.current) return;
    const observer = new IntersectionObserver(
      (entries) => {
        if (entries.some((entry) => entry.isIntersecting)) {
          setVisible(true);
          observer.disconnect();
        }
      },
      { rootMargin: '96px' },
    );
    observer.observe(elementRef.current);
    return () => observer.disconnect();
  }, [visible]);
  useEffect(() => {
    let active = true;
    setSource(null);
    setFailed(false);
    if (!visible || !path || !provider || primitiveKind) return;
    const request = thumbnailRequest(provider, path);
    void request.then((value) => {
      if (active) setSource(value);
    });
    return () => {
      active = false;
    };
  }, [asset?.status, path, primitiveKind, provider, visible]);

  return (
    <span className={`asset-thumbnail ${source ? 'has-image' : ''}`} ref={elementRef}>
      {primitiveKind ? (
        <PrimitiveMeshIcon kind={primitiveKind} />
      ) : source && !failed ? (
        <img alt="" draggable={false} onError={() => setFailed(true)} src={source} />
      ) : (
        <>
          <Image aria-hidden="true" size={17} />
          <em>{path ? extensionOf(path).slice(1, 5).toUpperCase() : '—'}</em>
        </>
      )}
      {asset?.status === 'importing' && <i />}
    </span>
  );
}

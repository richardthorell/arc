import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, Image, Search, X } from 'lucide-react';
import { createPortal } from 'react-dom';

export type AssetPickerItem = {
  id: string;
  name: string;
  path: string;
  kind: string;
  status: 'ready' | 'dirty' | 'importing' | 'missing';
};

export type AssetThumbnailProvider = (path: string) => Promise<string | null>;

export type AssetPickerProps = {
  assets: ReadonlyArray<AssetPickerItem>;
  value: string;
  label: string;
  assetKinds: ReadonlyArray<string>;
  assetTypeLabel?: string;
  allowedExtensions?: ReadonlyArray<string>;
  allowEmpty?: boolean;
  thumbnailProvider?: AssetThumbnailProvider;
  onChange: (path: string) => void;
};

const thumbnailCaches = new WeakMap<AssetThumbnailProvider, Map<string, Promise<string | null>>>();
const extensionOf = (path: string) => path.slice(path.lastIndexOf('.')).toLocaleLowerCase();

function thumbnailRequest(provider: AssetThumbnailProvider, path: string): Promise<string | null> {
  let cache = thumbnailCaches.get(provider);
  if (!cache) {
    cache = new Map<string, Promise<string | null>>();
    thumbnailCaches.set(provider, cache);
  }
  let request = cache.get(path);
  if (!request) {
    request = provider(path).catch(() => null);
    cache.set(path, request);
  }
  return request;
}

export function AssetPicker({
  assets, value, label, assetKinds, assetTypeLabel = 'Asset', allowedExtensions, allowEmpty = true, thumbnailProvider, onChange,
}: AssetPickerProps) {
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const anchorRef = useRef<HTMLButtonElement>(null);
  const candidates = useMemo(() => assets.filter((asset) =>
    assetKinds.includes(asset.kind) && (!allowedExtensions?.length || allowedExtensions.includes(extensionOf(asset.path)))),
  [allowedExtensions, assetKinds, assets]);
  const selected = assets.find((asset) => asset.path === value);

  const acceptDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const path = event.dataTransfer.getData('application/x-arc-asset') ||
      event.dataTransfer.getData('application/x-arc-environment');
    const candidate = candidates.find((asset) => asset.path === path);
    if (candidate) onChange(candidate.path);
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
          <AssetThumbnail asset={selected} path={value} provider={thumbnailProvider} />
          <span className="asset-reference-copy">
            <strong>{selected?.name || (value ? value.split(/[\\/]/).pop() : 'None')}</strong>
            <small>{value || `No ${assetTypeLabel.toLocaleLowerCase()} assigned`}</small>
          </span>
          <ChevronDown size={13} />
        </button>
        {allowEmpty && value && <button aria-label={`Clear ${label}`} className="asset-reference-clear" onClick={() => onChange('')}
          title="Clear asset reference" type="button"><X size={13} /></button>}
      </div>
      {open && <AssetPickerPopover anchorRef={anchorRef} assets={candidates} assetTypeLabel={assetTypeLabel} filter={filter} label={label}
        selectedPath={value} thumbnailProvider={thumbnailProvider} onClose={() => setOpen(false)} onFilter={setFilter}
        onSelect={(path) => { onChange(path); setOpen(false); }} />}
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

export function AssetPreview({ path, name, label, provider }: {
  path: string;
  name: string;
  label: string;
  provider?: AssetThumbnailProvider;
}) {
  return <div className="asset-preview-property">
    <div aria-label={label} className="asset-preview-stage">
      <AssetThumbnail path={path} provider={provider} />
      <span><strong>{name || 'No Material'}</strong><small>{path || 'Embedded runtime material'}</small></span>
    </div>
  </div>;
}

function AssetPickerPopover({ anchorRef, assets, assetTypeLabel, filter, label, selectedPath, thumbnailProvider, onClose, onFilter, onSelect }: {
  anchorRef: React.RefObject<HTMLElement | null>;
  assets: ReadonlyArray<AssetPickerItem>;
  assetTypeLabel: string;
  filter: string;
  label: string;
  selectedPath: string;
  thumbnailProvider?: AssetThumbnailProvider;
  onClose: () => void;
  onFilter: (value: string) => void;
  onSelect: (path: string) => void;
}) {
  const popoverRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ left: 8, top: 8 });
  const shown = assets.filter((asset) => `${asset.name} ${asset.path}`.toLocaleLowerCase().includes(filter.trim().toLocaleLowerCase()));

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
    const escape = (event: KeyboardEvent) => { if (event.key === 'Escape') onClose(); };
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      document.removeEventListener('keydown', escape);
    };
  }, [anchorRef, onClose]);

  return createPortal(
    <section aria-label={`${label} asset picker`} className="asset-picker-popover" ref={popoverRef} role="dialog"
      style={{ left: position.left, top: position.top }}>
      <header><strong>{`Select ${assetTypeLabel}`}</strong><span>{shown.length} assets</span><button aria-label="Close asset picker" onClick={onClose} type="button"><X size={14} /></button></header>
      <label className="asset-picker-search"><Search size={14} /><input aria-label={`Search ${assetTypeLabel.toLocaleLowerCase()} assets`} autoFocus onChange={(event) => onFilter(event.target.value)}
        placeholder={`Search ${assetTypeLabel.toLocaleLowerCase()}s…`} value={filter} /></label>
      <div className="asset-picker-grid">
        {shown.map((asset) => <button aria-label={`Select ${asset.name}`} className={asset.path === selectedPath ? 'is-selected' : ''}
          key={asset.id} onClick={() => onSelect(asset.path)} type="button">
          <AssetThumbnail asset={asset} path={asset.path} provider={thumbnailProvider} />
          <strong>{asset.name}</strong><small>{extensionOf(asset.path).slice(1).toUpperCase()} · {asset.status}</small>
        </button>)}
        {!shown.length && <div className="asset-picker-empty"><Image size={22} /><span>{`No matching ${assetTypeLabel.toLocaleLowerCase()}s`}</span></div>}
      </div>
    </section>,
    document.body,
  );
}

export function AssetThumbnail({ asset, path, provider }: { asset?: AssetPickerItem; path: string; provider?: AssetThumbnailProvider }) {
  const elementRef = useRef<HTMLSpanElement>(null);
  const [source, setSource] = useState<string | null>(null);
  const [failed, setFailed] = useState(false);
  const [visible, setVisible] = useState(() => typeof IntersectionObserver === 'undefined');
  useEffect(() => {
    if (visible || typeof IntersectionObserver === 'undefined' || !elementRef.current) return;
    const observer = new IntersectionObserver((entries) => {
      if (entries.some((entry) => entry.isIntersecting)) {
        setVisible(true);
        observer.disconnect();
      }
    }, { rootMargin: '96px' });
    observer.observe(elementRef.current);
    return () => observer.disconnect();
  }, [visible]);
  useEffect(() => {
    let active = true;
    setSource(null);
    setFailed(false);
    if (!visible || !path || !provider) return;
    const request = thumbnailRequest(provider, path);
    void request.then((value) => { if (active) setSource(value); });
    return () => { active = false; };
  }, [path, provider, visible]);

  return <span className={`asset-thumbnail ${source ? 'has-image' : ''}`} ref={elementRef}>
    {source && !failed
      ? <img alt="" draggable={false} onError={() => setFailed(true)} src={source} />
      : <><Image aria-hidden="true" size={17} /><em>{path ? extensionOf(path).slice(1, 5).toUpperCase() : '—'}</em></>}
    {asset?.status === 'importing' && <i />}
  </span>;
}

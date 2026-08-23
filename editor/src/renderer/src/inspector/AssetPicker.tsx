import { useEffect, useLayoutEffect, useMemo, useRef, useState } from 'react';
import { ChevronDown, ExternalLink, Image, Plus, Search, X } from 'lucide-react';
import { createPortal } from 'react-dom';

import { buildAssetCreation } from '../content/assetCreation';
import { readArcAssetDragPayload } from '../services/assetDragPayload';
import { openAssetEditorDocument } from '../editors/editorRegistry';
import { MaterialParameterSubsection } from './MaterialParameterSubsection';

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
  createNewLabel?: string;
  onCreateNew?: (name: string) => Promise<string>;
  onOpen?: (asset: AssetPickerItem) => void;
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
const normalizedPath = (value: string) => value.replaceAll('\\', '/').toLocaleLowerCase();
const projectContentRootFromAssets = (assets: ReadonlyArray<AssetPickerItem>) => {
  const projectAsset = assets.find(
    (asset) =>
      (asset.scope === undefined || asset.scope === 'project') &&
      !asset.path.startsWith('arc://') &&
      !/^[a-z]:[\\/]/i.test(asset.path) &&
      !asset.path.startsWith('/'),
  );
  return projectAsset?.path.replaceAll('\\', '/').split('/')[0] || 'Content';
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
  const fillId = `primitive-fill-${kind}`;
  const common = { fill: `url(#${fillId})`, stroke: '#8fc8ff', strokeWidth: 1.35 };

  return (
    <svg
      aria-hidden="true"
      data-testid={`primitive-mesh-icon-${kind}`}
      style={{
        height: '88%',
        width: '88%',
        margin: 'auto',
        filter: 'drop-shadow(0 5px 6px rgba(0, 0, 0, 0.38))',
      }}
      viewBox="0 0 64 64"
    >
      <defs>
        <linearGradient id={fillId} x1="0" y1="0" x2="1" y2="1">
          <stop offset="0" stopColor="#52789a" />
          <stop offset="1" stopColor="#172a3a" />
        </linearGradient>
      </defs>
      {kind === 'cube' && (
        <>
          <path {...common} d="M14 22 32 12l18 10-18 11z" />
          <path {...common} d="M14 22v22l18 10V33z" />
          <path {...common} d="M50 22v22L32 54V33z" />
        </>
      )}
      {kind === 'sphere' && (
        <>
          <circle {...common} cx="32" cy="32" r="21" />
          <ellipse cx="32" cy="32" rx="10" ry="21" fill="none" stroke="#8fc8ff" opacity="0.72" />
          <path d="M12 32h40M17 21c9 5 21 5 30 0M17 43c9-5 21-5 30 0" fill="none" stroke="#8fc8ff" opacity="0.58" />
        </>
      )}
      {kind === 'cylinder' && (
        <>
          <path {...common} d="M16 18c0-6 32-6 32 0v28c0 7-32 7-32 0z" />
          <ellipse {...common} cx="32" cy="18" rx="16" ry="6" />
          <path d="M16 45c3 7 29 7 32 0" fill="none" stroke="#8fc8ff" />
        </>
      )}
      {kind === 'cone' && (
        <>
          <path {...common} d="M32 10 13 46c1 9 37 9 38 0z" />
          <ellipse {...common} cx="32" cy="46" rx="19" ry="7" />
        </>
      )}
      {kind === 'capsule' && (
        <>
          <path {...common} d="M18 23a14 14 0 0 1 28 0v18a14 14 0 0 1-28 0z" />
          <path d="M18 23c5 4 23 4 28 0M18 41c5-4 23-4 28 0" fill="none" stroke="#8fc8ff" opacity="0.62" />
        </>
      )}
      {kind === 'plane' && (
        <>
          <path {...common} d="m8 40 29-25 19 11-29 25z" />
          <path d="m17 32 19 11M27 24l19 11M19 46l29-25" fill="none" stroke="#8fc8ff" opacity="0.55" />
        </>
      )}
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
  createNewLabel,
  onCreateNew,
  onOpen,
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
    const dropped = readArcAssetDragPayload(event.dataTransfer);
    if (!dropped) return;
    const candidate = candidates.find(
      (asset) =>
        (dropped.guid && (asset.guid === dropped.guid || asset.id === dropped.guid)) ||
        normalizedPath(asset.path) === normalizedPath(dropped.pathHint),
    );
    if (!candidate) return;
    event.preventDefault();
    event.dataTransfer.dropEffect = 'copy';
    onChange(valueFor(candidate));
  };

  const canOpen = Boolean(selected && onOpen && !mixed);
  const canClear = Boolean(allowEmpty && value && !mixed);

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
        {(canOpen || canClear) && (
          <span style={{ display: 'flex', alignItems: 'center' }}>
            {canOpen && selected && onOpen && (
              <button
                aria-label={`Open ${displayNameOf(selected, value)} in ${assetTypeLabel} Editor`}
                className="asset-reference-clear"
                onClick={() => onOpen(selected)}
                title={`Open in ${assetTypeLabel} Editor`}
                type="button"
              >
                <ExternalLink size={13} />
              </button>
            )}
            {canClear && (
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
          </span>
        )}
      </div>
      {open && (
        <AssetPickerPopover
          anchorRef={anchorRef}
          assets={candidates}
          assetTypeLabel={assetTypeLabel}
          createNewLabel={createNewLabel}
          filter={filter}
          label={label}
          selectedValue={value}
          thumbnailProvider={thumbnailProvider}
          onClose={() => setOpen(false)}
          onCreateNew={
            onCreateNew
              ? async (name) => {
                  const next = await onCreateNew(name);
                  onChange(next);
                  setOpen(false);
                }
              : undefined
          }
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

export function MaterialPicker(
  props: Omit<AssetPickerProps, 'assetKinds' | 'assetTypeLabel' | 'createNewLabel' | 'onCreateNew' | 'onOpen'>,
) {
  const openMaterial = (asset: AssetPickerItem) => {
    if (asset.kind !== 'material' || asset.scope === 'procedural') return;
    openAssetEditorDocument({
      id: asset.id,
      guid: asset.guid,
      typeId: asset.typeId,
      name: asset.name,
      path: asset.path,
      kind: 'material',
      status: asset.status,
      scope: asset.scope,
      readOnly: asset.readOnly,
    });
  };

  const createMaterial = async (name: string) => {
    const projectSnapshot = await window.arc.projects.snapshot();
    const activeProject = projectSnapshot?.activeProject;
    if (activeProject && !activeProject.writable) throw new Error('The active project is read-only');
    const contentRoot = activeProject?.descriptor.paths.content || projectContentRootFromAssets(props.assets);
    const definition = buildAssetCreation(
      { root: activeProject?.projectRoot ?? '', assetRoot: contentRoot },
      { kind: 'material', name, folder: contentRoot },
    );
    if (props.assets.some((asset) => normalizedPath(asset.path) === normalizedPath(definition.asset.path))) {
      throw new Error(`A material already exists at ${definition.asset.path}`);
    }
    await window.arc.projects.writeText(definition.asset.path, definition.contents);
    openAssetEditorDocument(definition.asset);
    return definition.asset.path;
  };

  return (
    <>
      <AssetPicker
        {...props}
        assetKinds={['material']}
        assetTypeLabel="Material"
        createNewLabel="Create New Material…"
        onCreateNew={createMaterial}
        onOpen={openMaterial}
      />
      <MaterialParameterSubsection
        assets={props.assets}
        mixed={props.mixed}
        referenceMode={props.referenceMode}
        value={props.value}
      />
    </>
  );
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
  createNewLabel,
  filter,
  label,
  selectedValue,
  thumbnailProvider,
  onClose,
  onCreateNew,
  onFilter,
  onSelect,
  valueFor,
}: {
  anchorRef: React.RefObject<HTMLElement | null>;
  assets: ReadonlyArray<AssetPickerItem>;
  assetTypeLabel: string;
  createNewLabel?: string;
  filter: string;
  label: string;
  selectedValue: string;
  thumbnailProvider?: AssetThumbnailProvider;
  onClose: () => void;
  onCreateNew?: (name: string) => Promise<void>;
  onFilter: (value: string) => void;
  onSelect: (asset: AssetPickerItem) => void;
  valueFor: (asset: AssetPickerItem) => string;
}) {
  const popoverRef = useRef<HTMLDivElement>(null);
  const [position, setPosition] = useState({ left: 8, top: 8 });
  const [createMode, setCreateMode] = useState(false);
  const [createName, setCreateName] = useState('');
  const [createError, setCreateError] = useState('');
  const [creating, setCreating] = useState(false);
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
      if (event.key === 'Escape') {
        if (createMode && !creating) {
          setCreateMode(false);
          setCreateError('');
        } else {
          onClose();
        }
      }
    };
    document.addEventListener('pointerdown', outside, true);
    document.addEventListener('keydown', escape);
    return () => {
      document.removeEventListener('pointerdown', outside, true);
      document.removeEventListener('keydown', escape);
    };
  }, [anchorRef, createMode, creating, onClose]);

  const submitCreate = async () => {
    if (!onCreateNew || !createName.trim() || creating) return;
    setCreating(true);
    setCreateError('');
    try {
      await onCreateNew(createName);
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : String(error));
    } finally {
      setCreating(false);
    }
  };

  return createPortal(
    <section
      aria-label={`${label} asset picker`}
      className="asset-picker-popover"
      ref={popoverRef}
      role="dialog"
      style={{
        left: position.left,
        top: position.top,
        ...(createMode ? { gridTemplateRows: '31px minmax(0, 1fr)' } : {}),
      }}
    >
      <header>
        <strong>{createMode ? `Create ${assetTypeLabel}` : `Select ${assetTypeLabel}`}</strong>
        <span>{createMode ? 'Project asset' : `${shown.length} assets`}</span>
        <button aria-label="Close asset picker" onClick={onClose} type="button">
          <X size={14} />
        </button>
      </header>
      {!createMode && (
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
      )}
      <div className="asset-picker-grid">
        {createMode ? (
          <form
            onSubmit={(event) => {
              event.preventDefault();
              void submitCreate();
            }}
            style={{
              gridColumn: '1 / -1',
              display: 'grid',
              alignContent: 'start',
              gap: 10,
              minHeight: 132,
              padding: 12,
            }}
          >
            <strong>{createNewLabel ?? `Create New ${assetTypeLabel}`}</strong>
            <input
              aria-label={`New ${assetTypeLabel.toLocaleLowerCase()} name`}
              autoFocus
              disabled={creating}
              onChange={(event) => setCreateName(event.target.value)}
              placeholder={`New ${assetTypeLabel}`}
              value={createName}
              style={{ width: '100%', boxSizing: 'border-box', minHeight: 30 }}
            />
            {createError && (
              <small
                role="alert"
                style={{
                  color: 'var(--arc-color-danger)',
                  lineHeight: 1.35,
                  overflowWrap: 'anywhere',
                  whiteSpace: 'normal',
                }}
              >
                {createError}
              </small>
            )}
            <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 6 }}>
              <button
                disabled={creating}
                onClick={() => {
                  setCreateMode(false);
                  setCreateError('');
                }}
                type="button"
              >
                Cancel
              </button>
              <button disabled={creating || !createName.trim()} type="submit">
                {creating ? 'Creating…' : 'Create'}
              </button>
            </div>
          </form>
        ) : (
          <>
            {onCreateNew && (
              <button
                aria-label={createNewLabel ?? `Create New ${assetTypeLabel}`}
                onClick={() => {
                  setCreateName(`New ${assetTypeLabel}`);
                  setCreateError('');
                  setCreateMode(true);
                }}
                type="button"
              >
                <span className="asset-thumbnail">
                  <Plus aria-hidden="true" size={18} />
                </span>
                <strong>{createNewLabel ?? `Create New ${assetTypeLabel}`}</strong>
                <small>Project asset</small>
              </button>
            )}
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
            {!shown.length && !onCreateNew && (
              <div className="asset-picker-empty">
                <Image size={22} />
                <span>{`No matching ${assetTypeLabel.toLocaleLowerCase()}s`}</span>
              </div>
            )}
          </>
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

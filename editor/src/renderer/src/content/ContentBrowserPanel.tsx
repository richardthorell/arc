import { useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Folder, Globe2, Grid2X2, List, Lock, Search, Star } from 'lucide-react';

import type { ArcAssetSourceDescriptor } from '../../../common/assetSourceTypes';
import type { CommandId } from '../app/workbenchTypes';
import { openAssetEditorDocument } from '../editors/editorRegistry';
import type { AssetItem, ProjectSnapshot } from '../services/editorHostTypes';
import { AssetThumbnail } from '../inspector/AssetPicker';
import type { AssetThumbnailProvider } from '../inspector/AssetPicker';
import {
  buildAssetCreation,
  projectAssetRootPath,
  type AssetCreationRequest,
  type ShaderAssetTemplate,
} from './assetCreation';
import { RemoteAssetBrowser } from './RemoteAssetBrowser';

import './contentBrowser.css';

type CacheSnapshot = {
  cacheLocalBytes: number;
  cacheLocalHits: number;
  cacheLocalMisses: number;
  cacheSharedHits: number;
  cacheSharedMisses: number;
  cacheHitRate: number;
  cacheCorruptEntries: number;
  cacheEvictions: number;
};

type Props = {
  project: ProjectSnapshot | null;
  cache: CacheSnapshot | null;
  selectedAssetId: string | null;
  onSelectAsset: (assetId: string) => void;
  onCommand: (command: CommandId) => void;
  onInstantiatePrefab: (path: string) => void;
  onAssetAction: (type: 'asset.reimport' | 'asset.cancelImport', guid: string) => void;
  thumbnailProvider: AssetThumbnailProvider;
};

type CreateKind = 'material' | 'shader';
type CreateContextMenu = { x: number; y: number; folder: string };

const parentFolder = (path: string) => path.replaceAll('\\', '/').split('/').slice(0, -1).join('/');
const assetPayload = (asset: AssetItem) =>
  JSON.stringify({ guid: asset.guid ?? '', type: asset.kind, pathHint: asset.path });
const normalizedPath = (path: string) => path.replaceAll('\\', '/').toLocaleLowerCase();

export function ContentBrowserPanel({
  project,
  cache,
  selectedAssetId,
  onSelectAsset,
  onCommand,
  onInstantiatePrefab,
  onAssetAction,
  thumbnailProvider,
}: Props) {
  const [folder, setFolder] = useState('');
  const [search, setSearch] = useState('');
  const [kind, setKind] = useState<AssetItem['kind'] | 'all'>('all');
  const [state, setState] = useState<AssetItem['status'] | 'all'>('all');
  const [sort, setSort] = useState<'name' | 'type' | 'state'>('name');
  const [view, setView] = useState<'grid' | 'list'>('grid');
  const [browserSource, setBrowserSource] = useState('project');
  const [onlineSources, setOnlineSources] = useState<ArcAssetSourceDescriptor[]>([]);
  const [selection, setSelection] = useState<Set<string>>(() => new Set(selectedAssetId ? [selectedAssetId] : []));
  const [favorites, setFavorites] = useState<Set<string>>(() => {
    try {
      return new Set(JSON.parse(localStorage.getItem('arc.content.favorites') ?? '[]') as string[]);
    } catch {
      return new Set();
    }
  });
  const [createMenuOpen, setCreateMenuOpen] = useState(false);
  const [createContextMenu, setCreateContextMenu] = useState<CreateContextMenu | null>(null);
  const [createKind, setCreateKind] = useState<CreateKind | null>(null);
  const [createFolder, setCreateFolder] = useState('');
  const [createName, setCreateName] = useState('');
  const [shaderTemplate, setShaderTemplate] = useState<ShaderAssetTemplate>('surface');
  const [createError, setCreateError] = useState('');
  const [creating, setCreating] = useState(false);

  useEffect(() => {
    let mounted = true;
    void window.arc.assetSources
      .list()
      .then((sources) => {
        if (mounted) setOnlineSources(sources);
      })
      .catch(() => {
        if (mounted) setOnlineSources([]);
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    const closeOnEscape = (event: KeyboardEvent) => {
      if (event.key !== 'Escape') return;
      setCreateMenuOpen(false);
      setCreateContextMenu(null);
      if (!creating) setCreateKind(null);
    };
    window.addEventListener('keydown', closeOnEscape);
    return () => window.removeEventListener('keydown', closeOnEscape);
  }, [creating]);

  const assets = useMemo(() => project?.assets ?? [], [project?.assets]);
  const localScope = browserSource === 'builtin' ? 'builtin' : 'project';
  const scopedAssets = useMemo(
    () => assets.filter((asset) => (asset.scope ?? 'project') === localScope),
    [assets, localScope],
  );
  const folders = useMemo(
    () => Array.from(new Set(scopedAssets.map((asset) => parentFolder(asset.path)).filter(Boolean))).sort(),
    [scopedAssets],
  );
  const filtered = useMemo(
    () =>
      scopedAssets
        .filter((asset) => {
          const assetFolder = parentFolder(asset.path);
          return (
            (!folder || assetFolder === folder || assetFolder.startsWith(`${folder}/`)) &&
            (kind === 'all' || asset.kind === kind) &&
            (state === 'all' || asset.status === state) &&
            (!search || `${asset.name} ${asset.path} ${asset.guid ?? ''}`.toLowerCase().includes(search.toLowerCase()))
          );
        })
        .sort((left, right) => {
          const a = sort === 'name' ? left.name : sort === 'type' ? left.kind : left.status;
          const b = sort === 'name' ? right.name : sort === 'type' ? right.kind : right.status;
          return a.localeCompare(b);
        }),
    [folder, kind, scopedAssets, search, sort, state],
  );
  const selected = scopedAssets.find((asset) => asset.id === selectedAssetId) ?? null;
  const activeOnlineSource = onlineSources.find((source) => source.id === browserSource) ?? null;
  const crumbs = folder ? folder.split('/') : [];
  const contentRoot = project ? projectAssetRootPath(project) : 'Content';
  const creationFolder = browserSource === 'project' ? folder : contentRoot;

  const select = (asset: AssetItem, additive: boolean) => {
    setSelection((current) => {
      const next = additive ? new Set(current) : new Set<string>();
      if (next.has(asset.id)) next.delete(asset.id);
      else next.add(asset.id);
      return next;
    });
    onSelectAsset(asset.id);
  };
  const toggleFavorite = (asset: AssetItem) => {
    setFavorites((current) => {
      const next = new Set(current);
      const id = asset.guid ?? asset.path;
      if (next.has(id)) next.delete(id);
      else next.add(id);
      localStorage.setItem('arc.content.favorites', JSON.stringify([...next]));
      return next;
    });
  };

  const activateAsset = (asset: AssetItem) => {
    if (openAssetEditorDocument(asset)) return;
    if (asset.kind === 'prefab') onInstantiatePrefab(asset.path);
  };

  const beginCreate = (nextKind: CreateKind, targetFolder = creationFolder) => {
    setCreateMenuOpen(false);
    setCreateContextMenu(null);
    setCreateKind(nextKind);
    setCreateFolder(targetFolder);
    setCreateName(nextKind === 'material' ? 'New Material' : 'New Shader');
    setShaderTemplate('surface');
    setCreateError('');
  };

  const openContextCreate = (event: React.MouseEvent, targetFolder = folder) => {
    if (browserSource !== 'project') return;
    event.preventDefault();
    event.stopPropagation();
    setCreateMenuOpen(false);
    setCreateContextMenu({ x: event.clientX, y: event.clientY, folder: targetFolder });
  };

  const createAsset = async () => {
    if (!project || !createKind || creating) return;
    setCreateError('');
    setCreating(true);
    try {
      const request: AssetCreationRequest =
        createKind === 'material'
          ? { kind: 'material', name: createName, folder: createFolder }
          : { kind: 'shader', name: createName, folder: createFolder, template: shaderTemplate };
      const definition = buildAssetCreation(project, request);
      if (project.assets.some((asset) => normalizedPath(asset.path) === normalizedPath(definition.asset.path))) {
        throw new Error(`An asset already exists at ${definition.asset.path}`);
      }
      await window.arc.projects.writeText(definition.asset.path, definition.contents);
      setSelection(new Set([definition.asset.id]));
      onSelectAsset(definition.asset.id);
      openAssetEditorDocument(definition.asset);
      setCreateKind(null);
      setCreateName('');
    } catch (error) {
      setCreateError(error instanceof Error ? error.message : String(error));
    } finally {
      setCreating(false);
    }
  };

  const createMenu = (targetFolder: string, context = false) => (
    <div
      className={`content-create-menu ${context ? 'context' : ''}`}
      role="menu"
      aria-label="Create asset"
      {...(context && createContextMenu
        ? { style: { left: createContextMenu.x, top: createContextMenu.y } }
        : {})}
    >
      <button role="menuitem" onClick={() => beginCreate('material', targetFolder)}>
        <span className="content-create-type-icon material" aria-hidden="true" />
        <span>
          <strong>Material</strong>
          <small>PBR material graph</small>
        </span>
      </button>
      <button role="menuitem" onClick={() => beginCreate('shader', targetFolder)}>
        <span className="content-create-type-icon shader" aria-hidden="true">{'</>'}</span>
        <span>
          <strong>Shader</strong>
          <small>GLSL source asset</small>
        </span>
      </button>
    </div>
  );

  return (
    <section
      className="content-browser-v2"
      onClick={() => createContextMenu && setCreateContextMenu(null)}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        if (!activeOnlineSource && event.dataTransfer.files.length > 0) onCommand('file.importScene');
      }}
    >
      <aside className="content-folder-tree">
        <button
          className={browserSource === 'project' && !folder ? 'active' : ''}
          onClick={() => {
            setBrowserSource('project');
            setFolder('');
          }}
          onContextMenu={(event) => openContextCreate(event, '')}
        >
          <Folder size={14} />
          Content
        </button>
        <button
          className={browserSource === 'builtin' && !folder ? 'active' : ''}
          onClick={() => {
            setBrowserSource('builtin');
            setFolder('');
          }}
        >
          <Lock size={14} />
          Engine
        </button>
        {(browserSource === 'project' || browserSource === 'builtin') &&
          folders.map((path) => (
            <button
              className={folder === path ? 'active' : ''}
              key={path}
              onClick={() => setFolder(path)}
              onContextMenu={(event) => browserSource === 'project' && openContextCreate(event, path)}
              style={{ paddingLeft: `${12 + path.split('/').length * 10}px` }}
            >
              <Folder size={13} />
              {path.split('/').at(-1)}
            </button>
          ))}
        {onlineSources.length > 0 && <div className="content-source-heading">Sources</div>}
        {onlineSources.map((source) => (
          <button
            className={`content-source-button ${browserSource === source.id ? 'active' : ''}`}
            key={source.id}
            onClick={() => setBrowserSource(source.id)}
          >
            <Globe2 size={13} />
            {source.displayName}
          </button>
        ))}
      </aside>
      {activeOnlineSource ? (
        <RemoteAssetBrowser source={activeOnlineSource} />
      ) : (
        <>
          <div className="content-browser-main">
            <header className="content-browser-v2-toolbar">
              <div className="content-create-wrap">
                <button
                  aria-haspopup="menu"
                  aria-expanded={createMenuOpen}
                  onClick={(event) => {
                    event.stopPropagation();
                    setCreateContextMenu(null);
                    setCreateMenuOpen((open) => !open);
                  }}
                >
                  + Create <ChevronDown size={12} />
                </button>
                {createMenuOpen && createMenu(creationFolder)}
              </div>
              <button onClick={() => onCommand('file.importScene')}>Import</button>
              <nav>
                <button onClick={() => setFolder('')}>{browserSource === 'builtin' ? 'Engine' : 'Content'}</button>
                {crumbs.map((crumb, index) => (
                  <span key={`${crumb}-${index}`}>
                    <ChevronRight size={12} />
                    <button onClick={() => setFolder(crumbs.slice(0, index + 1).join('/'))}>{crumb}</button>
                  </span>
                ))}
              </nav>
              <label>
                <Search size={13} />
                <input
                  aria-label="Search assets"
                  placeholder="Search assets"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </label>
              <select
                aria-label="Asset type"
                value={kind}
                onChange={(event) => setKind(event.target.value as typeof kind)}
              >
                <option value="all">All types</option>
                {['scene', 'mesh', 'material', 'texture', 'shader', 'prefab'].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
              <select
                aria-label="Asset state"
                value={state}
                onChange={(event) => setState(event.target.value as typeof state)}
              >
                <option value="all">All states</option>
                {['ready', 'stale', 'importing', 'failed', 'missing'].map((value) => (
                  <option key={value}>{value}</option>
                ))}
              </select>
              <select
                aria-label="Sort assets"
                value={sort}
                onChange={(event) => setSort(event.target.value as typeof sort)}
              >
                <option value="name">Name</option>
                <option value="type">Type</option>
                <option value="state">State</option>
              </select>
              <button
                className={view === 'grid' ? 'active' : ''}
                aria-label="Grid view"
                onClick={() => setView('grid')}
              >
                <Grid2X2 size={14} />
              </button>
              <button
                className={view === 'list' ? 'active' : ''}
                aria-label="List view"
                onClick={() => setView('list')}
              >
                <List size={14} />
              </button>
            </header>
            {cache && (
              <div className="asset-cache-summary">
                <span>
                  DDC <b>{(cache.cacheLocalBytes / (1024 * 1024)).toFixed(1)} MiB</b>
                </span>
                <span>Hit {(cache.cacheHitRate * 100).toFixed(1)}%</span>
                <span>Evicted {cache.cacheEvictions}</span>
              </div>
            )}
            <div
              className={`content-assets ${view}`}
              role="listbox"
              aria-multiselectable="true"
              onContextMenu={(event) => {
                if (!(event.target as HTMLElement).closest('.content-asset')) openContextCreate(event);
              }}
            >
              {filtered.map((asset) => (
                <button
                  className={`content-asset ${selection.has(asset.id) ? 'selected' : ''}`}
                  draggable={Boolean(asset.guid)}
                  key={asset.id}
                  onClick={(event) => select(asset, event.ctrlKey || event.metaKey)}
                  onDoubleClick={() => activateAsset(asset)}
                  onDragStart={(event) => {
                    event.dataTransfer.setData('application/x-arc-asset', assetPayload(asset));
                    event.dataTransfer.effectAllowed = 'copy';
                  }}
                >
                  <span className="content-asset-preview">
                    <AssetThumbnail asset={asset} path={asset.path} provider={thumbnailProvider} />
                    <i className={`asset-state ${asset.status}`} />
                  </span>
                  <span className="content-asset-name">{asset.name}</span>
                  <small>
                    {asset.kind} · {asset.status}
                    {asset.readOnly ? ' · Built-in' : ''}
                  </small>
                  <span className="content-asset-actions" onClick={(event) => event.stopPropagation()}>
                    <button
                      aria-label="Favorite"
                      className={favorites.has(asset.guid ?? asset.path) ? 'active' : ''}
                      onClick={() => toggleFavorite(asset)}
                    >
                      <Star size={12} />
                    </button>
                    {asset.guid && !asset.readOnly && (
                      <button onClick={() => onAssetAction('asset.reimport', asset.guid!)}>Reimport</button>
                    )}
                  </span>
                </button>
              ))}
              {filtered.length === 0 && <div className="content-empty">No assets match this folder and filter.</div>}
            </div>
          </div>
          {selected && (
            <aside className="content-asset-details">
              <strong>{selected.name}</strong>
              <code>{selected.guid ?? 'Legacy path'}</code>
              <span>{selected.path}</span>
              <span>
                {selected.kind} · {selected.status}
                {selected.readOnly ? ' · Built-in · Read-only' : ''}
              </span>
              <details>
                <summary>Dependencies ({selected.dependencies?.length ?? 0})</summary>
                {(selected.dependencies ?? []).map((item) => (
                  <code key={item}>{item}</code>
                ))}
              </details>
              <details>
                <summary>References ({selected.reverseDependencies?.length ?? 0})</summary>
                {(selected.reverseDependencies ?? []).map((item) => (
                  <code key={item}>{item}</code>
                ))}
              </details>
              {selected.guid && (!selected.readOnly || selected.status === 'importing') && (
                <button
                  onClick={() =>
                    onAssetAction(
                      selected.status === 'importing' ? 'asset.cancelImport' : 'asset.reimport',
                      selected.guid!,
                    )
                  }
                >
                  {selected.status === 'importing' ? 'Cancel Import' : 'Reimport'}
                </button>
              )}
            </aside>
          )}
        </>
      )}
      {createContextMenu && createMenu(createContextMenu.folder, true)}
      {createKind && (
        <div
          className="content-create-backdrop"
          onMouseDown={(event) => {
            if (event.currentTarget === event.target && !creating) setCreateKind(null);
          }}
        >
          <form
            className="content-create-dialog"
            role="dialog"
            aria-modal="true"
            aria-labelledby="content-create-title"
            onSubmit={(event) => {
              event.preventDefault();
              void createAsset();
            }}
          >
            <header>
              <strong id="content-create-title">Create {createKind === 'material' ? 'Material' : 'Shader'}</strong>
              <small>{createFolder || contentRoot}</small>
            </header>
            <label>
              Name
              <input
                autoFocus
                aria-label="Asset name"
                value={createName}
                onChange={(event) => setCreateName(event.target.value)}
                disabled={creating}
              />
            </label>
            {createKind === 'shader' && (
              <label>
                Template
                <select
                  aria-label="Shader template"
                  value={shaderTemplate}
                  onChange={(event) => setShaderTemplate(event.target.value as ShaderAssetTemplate)}
                  disabled={creating}
                >
                  <option value="surface">Surface Shader</option>
                  <option value="unlit">Unlit Shader</option>
                  <option value="compute">Compute Shader</option>
                  <option value="post-process">Post Process Shader</option>
                  <option value="empty">Empty Shader</option>
                </select>
              </label>
            )}
            {createError && <div className="content-create-error">{createError}</div>}
            <footer>
              <button type="button" onClick={() => setCreateKind(null)} disabled={creating}>
                Cancel
              </button>
              <button type="submit" disabled={creating || !createName.trim()}>
                {creating ? 'Creating…' : `Create ${createKind === 'material' ? 'Material' : 'Shader'}`}
              </button>
            </footer>
          </form>
        </div>
      )}
    </section>
  );
}

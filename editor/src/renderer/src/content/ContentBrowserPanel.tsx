import { Fragment, useEffect, useMemo, useState } from 'react';
import { ChevronDown, ChevronRight, Folder, Globe2, Grid2X2, List, Lock, Search, Star } from 'lucide-react';

import type { ArcAssetSourceDescriptor } from '../../../common/assetSourceTypes';
import type { CommandId } from '../app/workbenchTypes';
import { openAssetEditorDocument } from '../editors/editorRegistry';
import type { AssetThumbnailProvider } from '../inspector/AssetPicker';
import type { AssetItem, ProjectSnapshot } from '../services/editorHostTypes';
import { UiButton, UiIconButton, UiSearchInput, UiSelect, UiTreeRow } from '../ui';
import {
  buildAssetCreation,
  projectAssetRootPath,
  type AssetCreationRequest,
  type ShaderAssetTemplate,
} from './assetCreation';
import { ContentAssetCard } from './ContentAssetCard';
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
  onTextureStreamingMode?: (guid: string, mode: 'resident' | 'streamed_mips' | 'virtual_tiles') => void;
  thumbnailProvider: AssetThumbnailProvider;
};

type CreateKind = 'material' | 'shader';
type CreateContextMenu = { x: number; y: number; folder: string };
type LocalBrowserSource = 'project' | 'builtin';
type FolderTreeNode = {
  name: string;
  path: string;
  children: FolderTreeNode[];
};

const cleanPath = (path: string) =>
  path
    .replaceAll('\\', '/')
    .replace(/\/+/g, '/')
    .replace(/^\/|\/$/g, '');
const parentFolder = (path: string) => cleanPath(path).split('/').slice(0, -1).join('/');
const normalizedPath = (path: string) => cleanPath(path).toLocaleLowerCase();
const favoriteId = (asset: AssetItem) => asset.guid ?? asset.path;
const folderKey = (source: LocalBrowserSource, path: string) => `${source}:${normalizedPath(path)}`;
const modelFileExtensions = new Set(['fbx', 'glb', 'gltf', 'obj']);
const isModelFile = (file: File) => modelFileExtensions.has(file.name.split('.').at(-1)?.toLocaleLowerCase() ?? '');
const contentTreeWidthStorageKey = 'arc.content.treeWidth';
const defaultContentTreeWidth = 190;
const minContentTreeWidth = 140;
const maxContentTreeWidth = 420;
const clampContentTreeWidth = (value: number) => Math.min(maxContentTreeWidth, Math.max(minContentTreeWidth, value));
const readContentTreeWidth = () => {
  const stored = Number.parseInt(localStorage.getItem(contentTreeWidthStorageKey) ?? '', 10);
  return Number.isFinite(stored) ? clampContentTreeWidth(stored) : defaultContentTreeWidth;
};
const assetTypeOptions = [
  { value: 'all', label: 'All types' },
  { value: 'scene', label: 'Scene' },
  { value: 'mesh', label: 'Mesh' },
  { value: 'material', label: 'Material' },
  { value: 'texture', label: 'Texture' },
  { value: 'shader', label: 'Shader' },
  { value: 'prefab', label: 'Prefab' },
];
const assetStateOptions = [
  { value: 'all', label: 'All states' },
  { value: 'ready', label: 'Ready' },
  { value: 'stale', label: 'Stale' },
  { value: 'importing', label: 'Importing' },
  { value: 'failed', label: 'Failed' },
  { value: 'missing', label: 'Missing' },
];
const assetSortOptions = [
  { value: 'name', label: 'Name' },
  { value: 'type', label: 'Type' },
  { value: 'state', label: 'State' },
];

const relativeFolderPath = (assetPath: string, source: LocalBrowserSource, projectRootName: string) => {
  const segments = parentFolder(assetPath).split('/').filter(Boolean);
  if (segments.length === 0) return '';

  const aliases = new Set(
    (source === 'project' ? [projectRootName, 'Content'] : ['Engine', 'Builtin'])
      .filter(Boolean)
      .map((value) => value.toLocaleLowerCase()),
  );
  const rootIndex = segments.findIndex((segment) => aliases.has(segment.replace(/:$/, '').toLocaleLowerCase()));
  return (rootIndex >= 0 ? segments.slice(rootIndex + 1) : segments).join('/');
};

export const buildContentFolderTree = (
  assets: AssetItem[],
  source: LocalBrowserSource,
  projectRootName = 'Content',
): FolderTreeNode[] => {
  const roots: FolderTreeNode[] = [];
  const nodes = new Map<string, FolderTreeNode>();

  for (const asset of assets) {
    const folder = relativeFolderPath(asset.path, source, projectRootName);
    if (!folder) continue;

    let currentPath = '';
    let parent: FolderTreeNode | null = null;
    for (const segment of folder.split('/').filter(Boolean)) {
      currentPath = currentPath ? `${currentPath}/${segment}` : segment;
      const key = normalizedPath(currentPath);
      let node = nodes.get(key);
      if (!node) {
        node = { name: segment, path: currentPath, children: [] };
        nodes.set(key, node);
        if (parent) parent.children.push(node);
        else roots.push(node);
      }
      parent = node;
    }
  }

  const sortNodes = (items: FolderTreeNode[]) => {
    items.sort((left, right) => left.name.localeCompare(right.name));
    items.forEach((item) => sortNodes(item.children));
  };
  sortNodes(roots);
  return roots;
};

function FolderTreeRows({
  nodes,
  source,
  depth,
  browserSource,
  folder,
  expandedFolders,
  onSelect,
  onContextMenu,
}: {
  nodes: FolderTreeNode[];
  source: LocalBrowserSource;
  depth: number;
  browserSource: string;
  folder: string;
  expandedFolders: ReadonlySet<string>;
  onSelect: (source: LocalBrowserSource, path: string, hasChildren: boolean) => void;
  onContextMenu?: (event: React.MouseEvent, path: string) => void;
}) {
  return (
    <>
      {nodes.map((node) => {
        const hasChildren = node.children.length > 0;
        const expanded = expandedFolders.has(folderKey(source, node.path));
        const selected = browserSource === source && normalizedPath(folder) === normalizedPath(node.path);
        return (
          <Fragment key={`${source}:${normalizedPath(node.path)}`}>
            <UiTreeRow
              depth={depth}
              selected={selected}
              className={`content-tree-row ${selected ? 'active' : ''}`}
              aria-expanded={hasChildren ? expanded : undefined}
              onClick={() => onSelect(source, node.path, hasChildren)}
              onContextMenu={(event) => onContextMenu?.(event, node.path)}
            >
              {hasChildren ? (
                expanded ? (
                  <ChevronDown size={13} aria-hidden="true" />
                ) : (
                  <ChevronRight size={13} aria-hidden="true" />
                )
              ) : (
                <span aria-hidden="true" style={{ width: 13 }} />
              )}
              <Folder className="entity-icon entity-icon-folder" size={14} aria-hidden="true" />
              <span>{node.name}</span>
            </UiTreeRow>
            {hasChildren && expanded && (
              <FolderTreeRows
                nodes={node.children}
                source={source}
                depth={depth + 1}
                browserSource={browserSource}
                folder={folder}
                expandedFolders={expandedFolders}
                onSelect={onSelect}
                onContextMenu={onContextMenu}
              />
            )}
          </Fragment>
        );
      })}
    </>
  );
}

export function ContentBrowserPanel({
  project,
  cache,
  selectedAssetId,
  onSelectAsset,
  onCommand,
  onInstantiatePrefab,
  onAssetAction,
  onTextureStreamingMode,
  thumbnailProvider,
}: Props) {
  const [folder, setFolder] = useState('');
  const [search, setSearch] = useState('');
  const [kind, setKind] = useState<AssetItem['kind'] | 'all'>('all');
  const [state, setState] = useState<AssetItem['status'] | 'all'>('all');
  const [sort, setSort] = useState<'name' | 'type' | 'state'>('name');
  const [view, setView] = useState<'grid' | 'list'>('grid');
  const [browserSource, setBrowserSource] = useState('project');
  const [treeWidth, setTreeWidth] = useState(readContentTreeWidth);
  const [expandedFolders, setExpandedFolders] = useState<Set<string>>(() => new Set([folderKey('project', '')]));
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
  const projectAssets = useMemo(() => assets.filter((asset) => (asset.scope ?? 'project') === 'project'), [assets]);
  const builtinAssets = useMemo(() => assets.filter((asset) => asset.scope === 'builtin'), [assets]);
  const favoriteAssets = useMemo(() => assets.filter((asset) => favorites.has(favoriteId(asset))), [assets, favorites]);
  const contentRoot = project ? projectAssetRootPath(project) : 'Content';
  const contentRootName = cleanPath(contentRoot).split('/').at(-1) || 'Content';
  const projectFolders = useMemo(
    () => buildContentFolderTree(projectAssets, 'project', contentRootName),
    [contentRootName, projectAssets],
  );
  const builtinFolders = useMemo(
    () => buildContentFolderTree(builtinAssets, 'builtin', contentRootName),
    [builtinAssets, contentRootName],
  );
  const scopedAssets = useMemo(() => {
    if (browserSource === 'favorites') return favoriteAssets;
    if (browserSource === 'builtin') return builtinAssets;
    if (browserSource === 'project') return projectAssets;
    return [];
  }, [browserSource, builtinAssets, favoriteAssets, projectAssets]);
  const filtered = useMemo(
    () =>
      scopedAssets
        .filter((asset) => {
          const source: LocalBrowserSource = browserSource === 'builtin' ? 'builtin' : 'project';
          const assetFolder = relativeFolderPath(asset.path, source, contentRootName);
          const assetFolderNormalized = normalizedPath(assetFolder);
          const folderNormalized = normalizedPath(folder);
          const inFolder =
            browserSource === 'favorites' ||
            !folderNormalized ||
            assetFolderNormalized === folderNormalized ||
            assetFolderNormalized.startsWith(`${folderNormalized}/`);
          return (
            inFolder &&
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
    [browserSource, contentRootName, folder, kind, scopedAssets, search, sort, state],
  );
  const selected = scopedAssets.find((asset) => asset.id === selectedAssetId) ?? null;
  const activeOnlineSource = onlineSources.find((source) => source.id === browserSource) ?? null;
  const crumbs = browserSource === 'favorites' || !folder ? [] : folder.split('/');
  const sourceTitle = browserSource === 'builtin' ? 'Engine' : browserSource === 'favorites' ? 'Favorites' : 'Content';
  const projectFolderPath = (relativePath: string) =>
    relativePath ? `${contentRoot}/${cleanPath(relativePath)}` : contentRoot;
  const creationFolder = browserSource === 'project' ? projectFolderPath(folder) : contentRoot;

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
      const id = favoriteId(asset);
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

  const selectTreeFolder = (source: LocalBrowserSource, path: string, hasChildren: boolean) => {
    const wasSelected = browserSource === source && normalizedPath(folder) === normalizedPath(path);
    setBrowserSource(source);
    setFolder(path);
    if (!hasChildren) return;

    const key = folderKey(source, path);
    setExpandedFolders((current) => {
      const next = new Set(current);
      if (wasSelected && next.has(key)) next.delete(key);
      else next.add(key);
      return next;
    });
  };

  const selectSourceRoot = (source: LocalBrowserSource, hasChildren: boolean) =>
    selectTreeFolder(source, '', hasChildren);

  const navigateFolder = (source: LocalBrowserSource, path: string) => {
    setBrowserSource(source);
    setFolder(path);
    setExpandedFolders((current) => {
      const next = new Set(current);
      next.add(folderKey(source, ''));
      const parts = path.split('/').filter(Boolean);
      for (let index = 1; index < parts.length; index += 1) {
        next.add(folderKey(source, parts.slice(0, index).join('/')));
      }
      return next;
    });
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

  const openProjectContextCreate = (event: React.MouseEvent, targetFolder = '') => {
    event.preventDefault();
    event.stopPropagation();
    setCreateMenuOpen(false);
    setCreateContextMenu({
      x: event.clientX,
      y: event.clientY,
      folder: projectFolderPath(targetFolder),
    });
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
      {...(context && createContextMenu ? { style: { left: createContextMenu.x, top: createContextMenu.y } } : {})}
    >
      <button role="menuitem" onClick={() => beginCreate('material', targetFolder)}>
        <span className="content-create-type-icon material" aria-hidden="true" />
        <span>
          <strong>Material</strong>
          <small>PBR material graph</small>
        </span>
      </button>
      <button role="menuitem" onClick={() => beginCreate('shader', targetFolder)}>
        <span className="content-create-type-icon shader" aria-hidden="true">
          {'</>'}
        </span>
        <span>
          <strong>Shader</strong>
          <small>GLSL source asset</small>
        </span>
      </button>
    </div>
  );

  const setContentTreeWidth = (nextWidth: number) => {
    const clamped = clampContentTreeWidth(nextWidth);
    setTreeWidth(clamped);
    localStorage.setItem(contentTreeWidthStorageKey, String(clamped));
  };

  const beginContentTreeResize = (event: React.PointerEvent<HTMLDivElement>) => {
    event.preventDefault();
    const startX = event.clientX;
    const startWidth = treeWidth;
    let latestWidth = startWidth;

    const move = (moveEvent: PointerEvent) => {
      latestWidth = clampContentTreeWidth(startWidth + moveEvent.clientX - startX);
      setTreeWidth(latestWidth);
    };
    const finish = () => {
      localStorage.setItem(contentTreeWidthStorageKey, String(latestWidth));
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', finish);
    };

    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', finish);
  };

  const handleContentTreeResizeKeyDown = (event: React.KeyboardEvent<HTMLDivElement>) => {
    let nextWidth: number | null = null;
    if (event.key === 'ArrowLeft') nextWidth = treeWidth - 12;
    else if (event.key === 'ArrowRight') nextWidth = treeWidth + 12;
    else if (event.key === 'Home') nextWidth = minContentTreeWidth;
    else if (event.key === 'End') nextWidth = maxContentTreeWidth;
    if (nextWidth === null) return;
    event.preventDefault();
    setContentTreeWidth(nextWidth);
  };

  const projectRootExpanded = expandedFolders.has(folderKey('project', ''));
  const builtinRootExpanded = expandedFolders.has(folderKey('builtin', ''));

  return (
    <section
      className="content-browser-v2"
      style={{ '--content-tree-width': `${treeWidth}px` } as React.CSSProperties}
      onClick={() => createContextMenu && setCreateContextMenu(null)}
      onDragOver={(event) => event.preventDefault()}
      onDrop={(event) => {
        if (activeOnlineSource || !project) return;
        const files = Array.from(event.dataTransfer.files).filter(isModelFile);
        if (!files.length) return;
        event.preventDefault();
        event.stopPropagation();
        const destination = projectFolderPath(folder);
        void Promise.all(files.map((file) => window.arc.projects.importModel(file, destination))).catch((error) =>
          console.error('[ARC] Model import failed', error),
        );
      }}
    >
      <aside className="content-folder-tree" role="tree" aria-label="Content folders">
        <UiTreeRow
          selected={browserSource === 'favorites'}
          className={`content-tree-row ${browserSource === 'favorites' ? 'active' : ''}`}
          onClick={() => {
            setBrowserSource('favorites');
            setFolder('');
          }}
        >
          <span aria-hidden="true" style={{ width: 13 }} />
          <Star className="entity-icon entity-icon-light" size={14} fill="currentColor" aria-hidden="true" />
          <span>Favorites</span>
        </UiTreeRow>
        <UiTreeRow
          selected={browserSource === 'project' && !folder}
          className={`content-tree-row ${browserSource === 'project' && !folder ? 'active' : ''}`}
          aria-expanded={projectFolders.length > 0 ? projectRootExpanded : undefined}
          onClick={() => selectSourceRoot('project', projectFolders.length > 0)}
          onContextMenu={(event) => openProjectContextCreate(event, '')}
        >
          {projectFolders.length > 0 ? (
            projectRootExpanded ? (
              <ChevronDown size={13} aria-hidden="true" />
            ) : (
              <ChevronRight size={13} aria-hidden="true" />
            )
          ) : (
            <span aria-hidden="true" style={{ width: 13 }} />
          )}
          <Folder className="entity-icon entity-icon-folder" size={14} aria-hidden="true" />
          <span>Content</span>
        </UiTreeRow>
        {projectRootExpanded && (
          <FolderTreeRows
            nodes={projectFolders}
            source="project"
            depth={1}
            browserSource={browserSource}
            folder={folder}
            expandedFolders={expandedFolders}
            onSelect={selectTreeFolder}
            onContextMenu={openProjectContextCreate}
          />
        )}
        <UiTreeRow
          selected={browserSource === 'builtin' && !folder}
          className={`content-tree-row ${browserSource === 'builtin' && !folder ? 'active' : ''}`}
          aria-expanded={builtinFolders.length > 0 ? builtinRootExpanded : undefined}
          onClick={() => selectSourceRoot('builtin', builtinFolders.length > 0)}
        >
          {builtinFolders.length > 0 ? (
            builtinRootExpanded ? (
              <ChevronDown size={13} aria-hidden="true" />
            ) : (
              <ChevronRight size={13} aria-hidden="true" />
            )
          ) : (
            <span aria-hidden="true" style={{ width: 13 }} />
          )}
          <Lock className="entity-icon" size={14} aria-hidden="true" />
          <span>Engine</span>
        </UiTreeRow>
        {builtinRootExpanded && (
          <FolderTreeRows
            nodes={builtinFolders}
            source="builtin"
            depth={1}
            browserSource={browserSource}
            folder={folder}
            expandedFolders={expandedFolders}
            onSelect={selectTreeFolder}
          />
        )}
        {onlineSources.length > 0 && <div className="content-source-heading">Sources</div>}
        {onlineSources.map((source) => {
          const selectedSource = browserSource === source.id;
          return (
            <UiTreeRow
              selected={selectedSource}
              className={`content-source-button ${selectedSource ? 'active' : ''}`}
              key={source.id}
              onClick={() => {
                setBrowserSource(source.id);
                setFolder('');
              }}
            >
              <span aria-hidden="true" style={{ width: 13 }} />
              <Globe2 size={13} aria-hidden="true" />
              <span>{source.displayName}</span>
            </UiTreeRow>
          );
        })}
      </aside>
      <div
        aria-label="Resize content folder tree"
        aria-orientation="vertical"
        aria-valuemax={maxContentTreeWidth}
        aria-valuemin={minContentTreeWidth}
        aria-valuenow={treeWidth}
        className="content-folder-resizer"
        role="separator"
        tabIndex={0}
        title="Drag to resize folder tree. Double-click to reset."
        onDoubleClick={() => setContentTreeWidth(defaultContentTreeWidth)}
        onKeyDown={handleContentTreeResizeKeyDown}
        onPointerDown={beginContentTreeResize}
      />
      {activeOnlineSource ? (
        <RemoteAssetBrowser source={activeOnlineSource} />
      ) : (
        <>
          <div className="content-browser-main">
            <header className="content-browser-v2-toolbar">
              <div className="content-create-wrap">
                <UiButton
                  aria-haspopup="menu"
                  aria-expanded={createMenuOpen}
                  type="button"
                  variant="toolbar"
                  onClick={(event) => {
                    event.stopPropagation();
                    setCreateContextMenu(null);
                    setCreateMenuOpen((open) => !open);
                  }}
                >
                  + Create <ChevronDown size={12} />
                </UiButton>
                {createMenuOpen && createMenu(creationFolder)}
              </div>
              <UiButton
                title="Import scene or model asset"
                type="button"
                variant="toolbar"
                onClick={() => onCommand('file.importScene')}
              >
                Import
              </UiButton>
              <nav aria-label="Content path">
                <UiButton
                  className="content-breadcrumb-button"
                  type="button"
                  variant="ghost"
                  onClick={() => {
                    if (browserSource === 'project' || browserSource === 'builtin') {
                      navigateFolder(browserSource, '');
                    }
                  }}
                >
                  {sourceTitle}
                </UiButton>
                {crumbs.map((crumb, index) => (
                  <span key={`${crumb}-${index}`}>
                    <ChevronRight size={12} aria-hidden="true" />
                    <UiButton
                      className="content-breadcrumb-button"
                      type="button"
                      variant="ghost"
                      onClick={() =>
                        navigateFolder(
                          browserSource === 'builtin' ? 'builtin' : 'project',
                          crumbs.slice(0, index + 1).join('/'),
                        )
                      }
                    >
                      {crumb}
                    </UiButton>
                  </span>
                ))}
              </nav>
              <label className="content-browser-search">
                <Search size={13} aria-hidden="true" />
                <UiSearchInput
                  aria-label="Search assets"
                  placeholder="Search assets"
                  value={search}
                  onChange={(event) => setSearch(event.target.value)}
                />
              </label>
              <UiSelect
                ariaLabel="Asset type"
                className="content-toolbar-select content-toolbar-select-type"
                options={assetTypeOptions}
                value={kind}
                onValueChange={(value) => setKind(value as typeof kind)}
              />
              <UiSelect
                ariaLabel="Asset state"
                className="content-toolbar-select content-toolbar-select-state"
                options={assetStateOptions}
                value={state}
                onValueChange={(value) => setState(value as typeof state)}
              />
              <UiSelect
                ariaLabel="Sort assets"
                className="content-toolbar-select content-toolbar-select-sort"
                options={assetSortOptions}
                value={sort}
                onValueChange={(value) => setSort(value as typeof sort)}
              />
              <UiIconButton
                active={view === 'grid'}
                label="Grid view"
                type="button"
                variant="toolbar"
                onClick={() => setView('grid')}
              >
                <Grid2X2 size={14} />
              </UiIconButton>
              <UiIconButton
                active={view === 'list'}
                label="List view"
                type="button"
                variant="toolbar"
                onClick={() => setView('list')}
              >
                <List size={14} />
              </UiIconButton>
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
                if (browserSource === 'project' && !(event.target as HTMLElement).closest('.content-asset')) {
                  openProjectContextCreate(event, folder);
                }
              }}
            >
              {filtered.map((asset) => (
                <ContentAssetCard
                  asset={asset}
                  favorite={favorites.has(favoriteId(asset))}
                  key={asset.id}
                  selected={selection.has(asset.id)}
                  thumbnailProvider={thumbnailProvider}
                  onActivate={() => activateAsset(asset)}
                  onFavorite={() => toggleFavorite(asset)}
                  onReimport={() => asset.guid && onAssetAction('asset.reimport', asset.guid)}
                  onSelect={(additive) => select(asset, additive)}
                />
              ))}
              {filtered.length === 0 && (
                <div className="content-empty">
                  {browserSource === 'favorites'
                    ? 'No favorite assets yet. Star an asset to add it here.'
                    : 'No assets match this folder and filter.'}
                </div>
              )}
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
              {selected.kind === 'texture' && (
                <div className="content-texture-streaming">
                  <span>
                    {selected.width || 0} × {selected.height || 0} · {selected.textureFormat || 'Unknown'} ·{' '}
                    {selected.mipLevels || 0} mips
                    {selected.tileCount ? ` · ${selected.tileCount} tiles` : ''}
                  </span>
                  <label>
                    Streaming mode
                    <select
                      aria-label="Texture streaming mode"
                      value={selected.streamingMode ?? 'resident'}
                      disabled={selected.readOnly || selected.status === 'importing'}
                      onChange={(event) =>
                        selected.guid &&
                        onTextureStreamingMode?.(
                          selected.guid,
                          event.target.value as 'resident' | 'streamed_mips' | 'virtual_tiles',
                        )
                      }
                    >
                      <option value="resident">Resident</option>
                      <option value="streamed_mips" disabled={Boolean(selected.streamingEligibilityError)}>
                        Streamed Mips
                      </option>
                      <option value="virtual_tiles" disabled={Boolean(selected.streamingEligibilityError)}>
                        Virtual Tiles
                      </option>
                    </select>
                  </label>
                  {selected.streamingEligibilityError && (
                    <span className="content-texture-streaming-error">{selected.streamingEligibilityError}</span>
                  )}
                  <span>
                    Settings v{selected.settingsVersion ?? 1}
                    {selected.artifactSize ? ` · ${(selected.artifactSize / (1024 * 1024)).toFixed(2)} MiB cooked` : ''}
                  </span>
                </div>
              )}
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

from pathlib import Path


def replace(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise SystemExit(f'{label} anchor not found')
    return text.replace(old, new, 1)


panel = Path('editor/src/renderer/src/content/ContentBrowserPanel.tsx')
text = panel.read_text()
text = replace(
    text,
    "import { ChevronDown, ChevronRight, Folder, Globe2, Grid2X2, List, Lock, Search, Star } from 'lucide-react';",
    "import { ChevronDown, ChevronRight, Clock3, Download, Folder, Globe2, Grid2X2, List, Lock, Search, Star } from 'lucide-react';",
    'lucide import',
)
text = replace(
    text,
    "import { assetPresentationKind, type AssetPresentationKind } from './assetPresentation';\nimport { RemoteAssetBrowser } from './RemoteAssetBrowser';",
    """import { assetPresentationKind, type AssetPresentationKind } from './assetPresentation';
import {
  assetIdsMatchingImportedPaths,
  assetsForVirtualView,
  assetVirtualViewContains,
  assetVirtualViewForKind,
  defaultAssetVirtualViews,
  isAssetVirtualViewKind,
  loadAssetVirtualViews,
  recordDownloadedAssets,
  recordRecentAsset,
  removeAssetFromVirtualCollection,
  saveAssetVirtualViews,
  setFavoriteAsset,
} from './assetVirtualViewIntegration';
import { getAssetVirtualViewActions } from './assetVirtualViewActions';
import type { AssetVirtualView, AssetVirtualViewKind } from './assetVirtualViews';
import { RemoteAssetBrowser } from './RemoteAssetBrowser';""",
    'virtual imports',
)
text = text.replace("const favoriteId = (asset: AssetItem) => asset.guid ?? asset.path;\n", '', 1)
text = replace(
    text,
    """  const [metadataEntries, setMetadataEntries] = useState<AssetMetadataEntries>({});
  const [metadataAssetId, setMetadataAssetId] = useState<string | null>(null);
  const [favorites, setFavorites] = useState<Set<string>>(() => {
    try {
      return new Set(JSON.parse(localStorage.getItem('arc.content.favorites') ?? '[]') as string[]);
    } catch {
      return new Set();
    }
  });
  const [createMenuOpen, setCreateMenuOpen] = useState(false);""",
    """  const [metadataEntries, setMetadataEntries] = useState<AssetMetadataEntries>({});
  const [metadataAssetId, setMetadataAssetId] = useState<string | null>(null);
  const [virtualViews, setVirtualViews] = useState<AssetVirtualView[]>(defaultAssetVirtualViews);
  const [virtualViewsLoaded, setVirtualViewsLoaded] = useState(false);
  const [searchOriginSource, setSearchOriginSource] = useState('project');
  const [pendingDownloadedPaths, setPendingDownloadedPaths] = useState<readonly string[]>([]);
  const [createMenuOpen, setCreateMenuOpen] = useState(false);""",
    'state block',
)
text = replace(
    text,
    """  const activeProjectRoot = project?.root ?? null;
  useEffect(() => {
    let cancelled = false;
    setMetadataEntries({});
    setMetadataAssetId(null);
    if (!activeProjectRoot) return () => undefined;
    void loadAssetMetadata().then((entries) => {
      if (!cancelled) setMetadataEntries(entries);
    });
    return () => {
      cancelled = true;
    };
  }, [activeProjectRoot]);
""",
    """  const activeProjectRoot = project?.root ?? null;
  const activeProjectAssets = project?.assets;
  useEffect(() => {
    let cancelled = false;
    setMetadataEntries({});
    setMetadataAssetId(null);
    if (!activeProjectRoot) return () => undefined;
    void loadAssetMetadata().then((entries) => {
      if (!cancelled) setMetadataEntries(entries);
    });
    return () => {
      cancelled = true;
    };
  }, [activeProjectRoot]);

  useEffect(() => {
    setVirtualViewsLoaded(false);
    setBrowserSource('project');
    setSearchOriginSource('project');
    setFolder('');
    setSearch('');
    setPendingDownloadedPaths([]);
    if (!activeProjectRoot) {
      setVirtualViews(defaultAssetVirtualViews());
      return;
    }
    setVirtualViews(loadAssetVirtualViews(localStorage, activeProjectRoot, activeProjectAssets ?? []));
    setVirtualViewsLoaded(true);
  }, [activeProjectAssets, activeProjectRoot]);

  useEffect(() => {
    if (!activeProjectRoot || !virtualViewsLoaded) return;
    saveAssetVirtualViews(localStorage, activeProjectRoot, virtualViews);
  }, [activeProjectRoot, virtualViews, virtualViewsLoaded]);
""",
    'project effects',
)
text = replace(
    text,
    """  const projectAssets = useMemo(() => assets.filter((asset) => (asset.scope ?? 'project') === 'project'), [assets]);
  const builtinAssets = useMemo(() => assets.filter((asset) => asset.scope === 'builtin'), [assets]);
  const favoriteAssets = useMemo(() => assets.filter((asset) => favorites.has(favoriteId(asset))), [assets, favorites]);
  const contentRoot = project ? projectAssetRootPath(project) : 'Content';""",
    """  const projectAssets = useMemo(() => assets.filter((asset) => (asset.scope ?? 'project') === 'project'), [assets]);
  const builtinAssets = useMemo(() => assets.filter((asset) => asset.scope === 'builtin'), [assets]);
  const searchResultIds = useMemo(
    () => (search.trim() ? searchAssetLibrary(assets, { text: search }).assets.map((asset) => asset.id) : []),
    [assets, search],
  );
  const virtualAssets = useMemo(() => {
    if (!isAssetVirtualViewKind(browserSource)) return null;
    return assetsForVirtualView(virtualViews, browserSource, assets, searchResultIds);
  }, [assets, browserSource, searchResultIds, virtualViews]);
  const contentRoot = project ? projectAssetRootPath(project) : 'Content';""",
    'asset projection',
)
text = replace(
    text,
    """  const scopedAssets = useMemo(() => {
    if (browserSource === 'favorites') return favoriteAssets;
    if (browserSource === 'builtin') return builtinAssets;
    if (browserSource === 'project') return projectAssets;
    return [];
  }, [browserSource, builtinAssets, favoriteAssets, projectAssets]);
  const searchPathPrefix = useMemo(() => {
    if (browserSource === 'favorites') return '';
    const root = browserSource === 'builtin' ? 'Engine' : contentRoot;
    return folder ? `${root}/${cleanPath(folder)}` : root;
  }, [browserSource, contentRoot, folder]);""",
    """  const scopedAssets = useMemo(() => {
    if (virtualAssets) return virtualAssets;
    if (browserSource === 'builtin') return builtinAssets;
    if (browserSource === 'project') return projectAssets;
    return [];
  }, [browserSource, builtinAssets, projectAssets, virtualAssets]);
  const searchPathPrefix = useMemo(() => {
    if (isAssetVirtualViewKind(browserSource)) return '';
    const root = browserSource === 'builtin' ? 'Engine' : contentRoot;
    return folder ? `${root}/${cleanPath(folder)}` : root;
  }, [browserSource, contentRoot, folder]);""",
    'scoped projection',
)
text = replace(
    text,
    """  const activeOnlineSource = onlineSources.find((source) => source.id === browserSource) ?? null;
  const crumbs = browserSource === 'favorites' || !folder ? [] : folder.split('/');
  const sourceTitle = browserSource === 'builtin' ? 'Engine' : browserSource === 'favorites' ? 'Favorites' : 'Content';""",
    """  const activeOnlineSource = onlineSources.find((source) => source.id === browserSource) ?? null;
  const activeVirtualView = isAssetVirtualViewKind(browserSource)
    ? assetVirtualViewForKind(virtualViews, browserSource, searchResultIds)
    : null;
  const crumbs = isAssetVirtualViewKind(browserSource) || !folder ? [] : folder.split('/');
  const sourceTitle =
    browserSource === 'builtin'
      ? 'Engine'
      : browserSource === 'favorites'
        ? 'Favorites'
        : browserSource === 'recent'
          ? 'Recent'
          : browserSource === 'downloads'
            ? 'Downloads'
            : browserSource === 'search-results'
              ? 'Search Results'
              : 'Content';""",
    'source titles',
)
text = replace(
    text,
    """  const select = (asset: AssetItem, additive: boolean) => {
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
  };""",
    """  const select = (asset: AssetItem, additive: boolean) => {
    setSelection((current) => {
      const next = additive ? new Set(current) : new Set<string>();
      if (next.has(asset.id)) next.delete(asset.id);
      else next.add(asset.id);
      return next;
    });
    setVirtualViews((current) => recordRecentAsset(current, asset.id));
    onSelectAsset(asset.id);
  };
  const toggleFavorite = (asset: AssetItem) => {
    setVirtualViews((current) =>
      setFavoriteAsset(current, asset.id, !assetVirtualViewContains(current, 'favorites', asset.id)),
    );
  };

  const activateAsset = (asset: AssetItem) => {
    if (openAssetEditorDocument(asset)) return;
    if (asset.kind === 'prefab') onInstantiatePrefab(asset.path);
  };

  const selectVirtualSource = (source: AssetVirtualViewKind) => {
    setBrowserSource(source);
    setFolder('');
    if (source !== 'search-results') setSearch('');
  };

  const changeSearch = (value: string) => {
    setSearch(value);
    if (value.trim()) {
      if (browserSource !== 'search-results') setSearchOriginSource(browserSource);
      setBrowserSource('search-results');
      setFolder('');
    } else if (browserSource === 'search-results') {
      setBrowserSource(searchOriginSource);
    }
  };

  const removeFromActiveVirtualView = (asset: AssetItem) => {
    if (!activeVirtualView || (activeVirtualView.kind !== 'favorites' && activeVirtualView.kind !== 'downloads')) return;
    setVirtualViews((current) => removeAssetFromVirtualCollection(current, activeVirtualView.kind, asset.id));
  };""",
    'view actions',
)
marker = "  const selectTreeFolder = (source: LocalBrowserSource, path: string, hasChildren: boolean) => {"
if marker not in text:
    raise SystemExit('download effect anchor not found')
text = text.replace(
    marker,
    """  useEffect(() => {
    if (pendingDownloadedPaths.length === 0) return;
    const assetIds = assetIdsMatchingImportedPaths(assets, pendingDownloadedPaths);
    if (assetIds.length === 0) return;
    setVirtualViews((current) => recordDownloadedAssets(current, assetIds));
    setPendingDownloadedPaths([]);
  }, [assets, pendingDownloadedPaths]);

""" + marker,
    1,
)
text = replace(
    text,
    """        <UiTreeRow
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
        </UiTreeRow>""",
    """        <UiTreeRow
          selected={browserSource === 'favorites'}
          className={`content-tree-row ${browserSource === 'favorites' ? 'active' : ''}`}
          onClick={() => selectVirtualSource('favorites')}
        >
          <span aria-hidden="true" style={{ width: 13 }} />
          <Star className="entity-icon entity-icon-light" size={14} fill="currentColor" aria-hidden="true" />
          <span>Favorites ({assetVirtualViewForKind(virtualViews, 'favorites').assetIds.length})</span>
        </UiTreeRow>
        <UiTreeRow
          selected={browserSource === 'recent'}
          className={`content-tree-row ${browserSource === 'recent' ? 'active' : ''}`}
          onClick={() => selectVirtualSource('recent')}
        >
          <span aria-hidden="true" style={{ width: 13 }} />
          <Clock3 className="entity-icon" size={14} aria-hidden="true" />
          <span>Recent ({assetVirtualViewForKind(virtualViews, 'recent').assetIds.length})</span>
        </UiTreeRow>
        <UiTreeRow
          selected={browserSource === 'downloads'}
          className={`content-tree-row ${browserSource === 'downloads' ? 'active' : ''}`}
          onClick={() => selectVirtualSource('downloads')}
        >
          <span aria-hidden="true" style={{ width: 13 }} />
          <Download className="entity-icon" size={14} aria-hidden="true" />
          <span>Downloads ({assetVirtualViewForKind(virtualViews, 'downloads').assetIds.length})</span>
        </UiTreeRow>
        {search.trim() && (
          <UiTreeRow
            selected={browserSource === 'search-results'}
            className={`content-tree-row ${browserSource === 'search-results' ? 'active' : ''}`}
            onClick={() => selectVirtualSource('search-results')}
          >
            <span aria-hidden="true" style={{ width: 13 }} />
            <Search className="entity-icon" size={14} aria-hidden="true" />
            <span>Search Results ({searchResultIds.length})</span>
          </UiTreeRow>
        )}""",
    'virtual navigation',
)
text = replace(
    text,
    "        <RemoteAssetBrowser source={activeOnlineSource} />",
    "        <RemoteAssetBrowser source={activeOnlineSource} onImportedFiles={setPendingDownloadedPaths} />",
    'remote callback',
)
text = replace(
    text,
    "                  onChange={(event) => setSearch(event.target.value)}",
    "                  onChange={(event) => changeSearch(event.target.value)}",
    'search handler',
)
text = replace(
    text,
    """                      asset={asset}
                      favorite={favorites.has(favoriteId(asset))}
                      selected={selection.has(asset.id)}
                      thumbnailProvider={thumbnailProvider}
                      onActivate={() => activateAsset(asset)}
                      onFavorite={() => toggleFavorite(asset)}
                      onReimport={() => asset.guid && onAssetAction('asset.reimport', asset.guid)}""",
    """                      asset={asset}
                      favorite={assetVirtualViewContains(virtualViews, 'favorites', asset.id)}
                      selected={selection.has(asset.id)}
                      thumbnailProvider={thumbnailProvider}
                      onActivate={() => activateAsset(asset)}
                      onFavorite={() => toggleFavorite(asset)}
                      onRemoveFromView={
                        activeVirtualView &&
                        getAssetVirtualViewActions(activeVirtualView, {
                          assetId: asset.id,
                          writable: !asset.readOnly,
                        }).some(({ action, enabled }) => action === 'remove-from-view' && enabled)
                          ? () => removeFromActiveVirtualView(asset)
                          : undefined
                      }
                      removeFromViewLabel={
                        activeVirtualView?.kind === 'downloads' ? 'Remove from Downloads' : undefined
                      }
                      onReimport={() => asset.guid && onAssetAction('asset.reimport', asset.guid)}""",
    'card virtual actions',
)
text = replace(
    text,
    """                  {browserSource === 'favorites'
                    ? 'No favorite assets yet. Star an asset to add it here.'
                    : 'No assets match this folder and filter.'}""",
    """                  {browserSource === 'favorites'
                    ? 'No favorite assets yet. Star an asset to add it here.'
                    : browserSource === 'recent'
                      ? 'No recently used assets yet.'
                      : browserSource === 'downloads'
                        ? 'No downloaded assets yet.'
                        : browserSource === 'search-results'
                          ? 'No assets match this search.'
                          : 'No assets match this folder and filter.'}""",
    'empty state',
)
panel.write_text(text)

card = Path('editor/src/renderer/src/content/ContentAssetCard.tsx')
text = card.read_text()
text = replace(text, "import { ChevronDown, ChevronRight, Star } from 'lucide-react';", "import { ChevronDown, ChevronRight, Star, X } from 'lucide-react';", 'card icon')
text = replace(
    text,
    """  onFavorite,
  onReimport,
  onSelect,
  expandable = false,""",
    """  onFavorite,
  onRemoveFromView,
  removeFromViewLabel,
  onReimport,
  onSelect,
  expandable = false,""",
    'card destructure',
)
text = replace(
    text,
    """  onFavorite: () => void;
  onReimport: () => void;
  onSelect: (additive: boolean) => void;
  expandable?: boolean;""",
    """  onFavorite: () => void;
  onRemoveFromView?: () => void;
  removeFromViewLabel?: string;
  onReimport: () => void;
  onSelect: (additive: boolean) => void;
  expandable?: boolean;""",
    'card props',
)
text = replace(
    text,
    """          <button aria-label="Favorite" className={favorite ? 'active' : ''} onClick={onFavorite}>
            <Star size={12} />
          </button>
          {asset.guid && !asset.readOnly && <button onClick={onReimport}>Reimport</button>}""",
    """          <button aria-label="Favorite" className={favorite ? 'active' : ''} onClick={onFavorite}>
            <Star size={12} />
          </button>
          {onRemoveFromView && removeFromViewLabel && (
            <button aria-label={removeFromViewLabel} title={removeFromViewLabel} onClick={onRemoveFromView}>
              <X size={12} />
            </button>
          )}
          {asset.guid && !asset.readOnly && <button onClick={onReimport}>Reimport</button>}""",
    'card actions',
)
card.write_text(text)

remote = Path('editor/src/renderer/src/content/RemoteAssetBrowser.tsx')
text = remote.read_text()
text = replace(text, """type Props = {
  source: ArcAssetSourceDescriptor;
};""", """type Props = {
  source: ArcAssetSourceDescriptor;
  onImportedFiles?: (paths: readonly string[]) => void;
};""", 'remote props')
text = replace(text, "export function RemoteAssetBrowser({ source }: Props) {", "export function RemoteAssetBrowser({ source, onImportedFiles }: Props) {", 'remote signature')
text = replace(
    text,
    """      .then((imported) => {
        setImportMessage(
          `Imported ${imported.importedFiles.length} files · ${imported.cacheHits} cache hits · ${imported.downloadedFiles} downloaded`,
        );
      })""",
    """      .then((imported) => {
        if (imported.succeeded) onImportedFiles?.(imported.importedFiles);
        setImportMessage(
          `Imported ${imported.importedFiles.length} files · ${imported.cacheHits} cache hits · ${imported.downloadedFiles} downloaded`,
        );
      })""",
    'remote completion',
)
remote.write_text(text)

remote_test = Path('editor/src/renderer/src/content/RemoteAssetBrowser.test.tsx')
text = remote_test.read_text()
text = replace(
    text,
    """    const view = render(<RemoteAssetBrowser source={source} />);
    await waitFor(() => expect(search).toHaveBeenCalledWith('polyhaven', expect.objectContaining({ limit: 160 })));""",
    """    const onImportedFiles = vi.fn();
    const view = render(<RemoteAssetBrowser source={source} onImportedFiles={onImportedFiles} />);
    await waitFor(() => expect(search).toHaveBeenCalledWith('polyhaven', expect.objectContaining({ limit: 160 })));""",
    'remote test render',
)
text = replace(
    text,
    "    expect(await view.findByText('Imported 2 files · 0 cache hits · 2 downloaded')).toBeInTheDocument();",
    """    expect(await view.findByText('Imported 2 files · 0 cache hits · 2 downloaded')).toBeInTheDocument();
    expect(onImportedFiles).toHaveBeenCalledWith(['rock.gltf', 'rock_diff.png']);""",
    'remote test assertion',
)
remote_test.write_text(text)

panel_test = Path('editor/src/renderer/src/content/ContentBrowserPanel.test.tsx')
text = panel_test.read_text()
text = replace(
    text,
    """  it('shows starred assets in the top-level Favorites folder', () => {
    localStorage.setItem('arc.content.favorites', JSON.stringify(['rock-guid']));
    const view = renderBrowser();

    fireEvent.click(view.getByRole('button', { name: 'Favorites' }));

    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    expect(view.queryByText('Sky')).not.toBeInTheDocument();
    expect(view.queryByText('Engine Sky Texture')).not.toBeInTheDocument();
  });""",
    """  it('migrates legacy Favorites and exposes all durable virtual views', async () => {
    localStorage.setItem('arc.content.favorites', JSON.stringify(['rock-guid']));
    const view = renderBrowser();

    const favorites = await view.findByRole('button', { name: 'Favorites (1)' });
    expect(view.getByRole('button', { name: 'Recent (0)' })).toBeInTheDocument();
    expect(view.getByRole('button', { name: 'Downloads (0)' })).toBeInTheDocument();
    fireEvent.click(favorites);

    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    expect(view.queryByText('Sky')).not.toBeInTheDocument();
    expect(view.queryByText('Engine Sky Texture')).not.toBeInTheDocument();
    expect(localStorage.getItem('arc.content.favorites')).toBeNull();
  });

  it('records selected assets in the persistent Recent virtual view', async () => {
    const view = renderBrowser();
    fireEvent.click(view.getByText('Hero Rock'));

    const recent = await view.findByRole('button', { name: 'Recent (1)' });
    fireEvent.click(recent);

    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    expect(view.queryByText('Sky')).not.toBeInTheDocument();
  });

  it('turns a live query into a transient cross-scope Search Results view', async () => {
    const view = renderBrowser();
    fireEvent.change(view.getByLabelText('Search assets'), { target: { value: 'engine sky' } });

    expect(await view.findByRole('button', { name: 'Search Results (1)' })).toBeInTheDocument();
    expect(view.getByText('Engine Sky Texture')).toBeInTheDocument();
    expect(view.queryByText('Hero Rock')).not.toBeInTheDocument();
    expect(localStorage.getItem('arc.content.virtualViews.v1:D:/Test')).not.toContain('search-results');
  });

  it('uses the virtual-view action policy to remove downloaded assets without deleting them', async () => {
    localStorage.setItem(
      'arc.content.virtualViews.v1:D:/Test',
      JSON.stringify({
        version: 1,
        views: [
          { kind: 'favorites', assetIds: [] },
          { kind: 'recent', assetIds: [] },
          { kind: 'downloads', assetIds: ['rock'] },
        ],
      }),
    );
    const view = renderBrowser();

    fireEvent.click(await view.findByRole('button', { name: 'Downloads (1)' }));
    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    fireEvent.click(view.getByRole('button', { name: 'Remove from Downloads' }));

    await waitFor(() => expect(view.queryByText('Hero Rock')).not.toBeInTheDocument());
    expect(view.getByRole('button', { name: 'Downloads (0)' })).toBeInTheDocument();
  });""",
    'panel virtual tests',
)
panel_test.write_text(text)

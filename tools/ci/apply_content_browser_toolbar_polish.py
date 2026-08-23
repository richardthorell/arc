from pathlib import Path


def replace_once(path: str, old: str, new: str) -> None:
    file_path = Path(path)
    text = file_path.read_text(encoding="utf-8")
    if old not in text:
        raise RuntimeError("expected source block not found in %s" % path)
    file_path.write_text(text.replace(old, new, 1), encoding="utf-8")


panel = "editor/src/renderer/src/content/ContentBrowserPanel.tsx"
css = "editor/src/renderer/src/app/workbench.css"
tests = "editor/src/renderer/src/content/ContentBrowserPanel.test.tsx"

replace_once(
    panel,
    "import { UiTreeRow } from '../ui';",
    "import { UiButton, UiIconButton, UiSearchInput, UiSelect, UiTreeRow } from '../ui';",
)

replace_once(
    panel,
    "const folderKey = (source: LocalBrowserSource, path: string) => `${source}:${normalizedPath(path)}`;\n",
    """const folderKey = (source: LocalBrowserSource, path: string) => `${source}:${normalizedPath(path)}`;
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
""",
)

replace_once(
    panel,
    "  const [browserSource, setBrowserSource] = useState('project');\n",
    "  const [browserSource, setBrowserSource] = useState('project');\n  const [treeWidth, setTreeWidth] = useState(readContentTreeWidth);\n",
)

replace_once(
    panel,
    """  const projectRootExpanded = expandedFolders.has(folderKey('project', ''));
  const builtinRootExpanded = expandedFolders.has(folderKey('builtin', ''));

  return (
""",
    """  const setContentTreeWidth = (nextWidth: number) => {
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
""",
)

replace_once(
    panel,
    """      className="content-browser-v2"
      onClick={() => createContextMenu && setCreateContextMenu(null)}
""",
    """      className="content-browser-v2"
      style={{ '--content-tree-width': `${treeWidth}px` } as React.CSSProperties}
      onClick={() => createContextMenu && setCreateContextMenu(null)}
""",
)

replace_once(
    panel,
    """      </aside>
      {activeOnlineSource ? (
""",
    """      </aside>
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
""",
)

old_toolbar = """            <header className="content-browser-v2-toolbar">
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
              <button onClick={() => onCommand('file.importScene')} title="Import scene or model asset">
                Import
              </button>
              <nav>
                <button
                  onClick={() => {
                    if (browserSource === 'project' || browserSource === 'builtin') {
                      navigateFolder(browserSource, '');
                    }
                  }}
                >
                  {sourceTitle}
                </button>
                {crumbs.map((crumb, index) => (
                  <span key={`${crumb}-${index}`}>
                    <ChevronRight size={12} />
                    <button
                      onClick={() =>
                        navigateFolder(
                          browserSource === 'builtin' ? 'builtin' : 'project',
                          crumbs.slice(0, index + 1).join('/'),
                        )
                      }
                    >
                      {crumb}
                    </button>
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
"""
new_toolbar = """            <header className="content-browser-v2-toolbar">
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
"""
replace_once(panel, old_toolbar, new_toolbar)

replace_once(
    css,
    """  grid-template-columns: 190px minmax(320px, 1fr) minmax(0, 0fr);
  height: 100%;
  min-height: 0;
  background: #111820;
  transition: grid-template-columns 140ms ease;
}

.content-browser-v2:has(.content-asset-details) {
  grid-template-columns: 190px minmax(320px, 1fr) 240px;
}
""",
    """  grid-template-columns: var(--content-tree-width, 190px) 5px minmax(320px, 1fr) minmax(0, 0fr);
  height: 100%;
  min-height: 0;
  background: #111820;
}

.content-browser-v2:has(.content-asset-details) {
  grid-template-columns: var(--content-tree-width, 190px) 5px minmax(320px, 1fr) 240px;
}
""",
)

replace_once(
    css,
    """  overflow: auto;
  border-right: 1px solid #2a3540;
  background: linear-gradient(180deg, #19232c, #121920);
}

.content-folder-tree {
  padding: 8px 5px;
}
""",
    """  overflow: auto;
  background: linear-gradient(180deg, #19232c, #121920);
}

.content-folder-tree {
  padding: 8px 5px;
}

.content-folder-resizer {
  position: relative;
  z-index: 5;
  width: 5px;
  min-width: 5px;
  height: 100%;
  padding: 0;
  border: 0;
  outline: 0;
  background: #111820;
  cursor: col-resize;
  touch-action: none;
}

.content-folder-resizer::before {
  position: absolute;
  inset: 0 auto 0 2px;
  width: 1px;
  background: #2a3540;
  content: '';
  transition: background 90ms ease;
}

.content-folder-resizer:hover::before,
.content-folder-resizer:focus-visible::before {
  background: var(--arc-color-accent);
}
""",
)

replace_once(
    css,
    """  gap: 5px;
  padding: 0 7px;
  overflow: hidden;
  border-bottom: 1px solid #2a3540;
""",
    """  position: relative;
  z-index: 20;
  gap: 5px;
  padding: 0 7px;
  overflow: visible;
  border-bottom: 1px solid #2a3540;
""",
)

replace_once(
    css,
    """.content-browser-v2-toolbar button,
.content-browser-v2-toolbar select,
.content-browser-v2-toolbar input,
.content-asset-details > button {
  min-height: 26px;
  border: 1px solid #32404d;
  border-radius: 3px;
  color: #c9d5df;
  background: #111920;
}

.content-browser-v2-toolbar button.active {
  border-color: #378bd0;
  background: #1d4d75;
}

.content-browser-v2-toolbar nav,
.content-browser-v2-toolbar nav span,
.content-browser-v2-toolbar label {
""",
    """.content-asset-details > button {
  min-height: 26px;
  border: 1px solid #32404d;
  border-radius: 3px;
  color: #c9d5df;
  background: #111920;
}

.content-browser-v2-toolbar nav,
.content-browser-v2-toolbar nav span,
.content-browser-search {
""",
)

replace_once(
    css,
    """.content-browser-v2-toolbar nav button {
  border: 0;
  background: transparent;
  white-space: nowrap;
}

.content-browser-v2-toolbar label {
  gap: 5px;
  padding-left: 7px;
  border: 1px solid #32404d;
  border-radius: 3px;
  background: #0f171e;
}

.content-browser-v2-toolbar label input {
  width: 130px;
  border: 0;
  outline: 0;
}
""",
    """.content-browser-v2-toolbar nav .ui-button {
  flex: 0 0 auto;
  height: var(--arc-button-height);
  padding-inline: var(--arc-space-3);
  white-space: nowrap;
}

.content-browser-search {
  position: relative;
  flex: 0 1 230px;
  min-width: 150px;
}

.content-browser-search > svg {
  position: absolute;
  z-index: 1;
  left: 10px;
  color: var(--arc-color-text-dim);
  pointer-events: none;
}

.content-browser-search .ui-search-input {
  width: 100%;
}

.content-toolbar-select {
  flex: 0 0 auto;
}

.content-toolbar-select-type,
.content-toolbar-select-state {
  width: 108px;
}

.content-toolbar-select-sort {
  width: 92px;
}

.content-browser-v2-toolbar > .ui-button,
.content-create-wrap > .ui-button {
  flex: 0 0 auto;
}
""",
)

replace_once(
    tests,
    """  it('filters registry assets and emits GUID drag payloads', () => {
""",
    """  it('uses shared ARC controls in the Content Browser toolbar', () => {
    const view = renderBrowser();

    expect(view.getByRole('button', { name: /Create/ })).toHaveClass('ui-button', 'ui-button-toolbar');
    expect(view.getByRole('button', { name: 'Import' })).toHaveClass('ui-button', 'ui-button-toolbar');
    expect(view.getByLabelText('Search assets')).toHaveClass('ui-text-input', 'ui-search-input');
    expect(view.getByRole('combobox', { name: 'Asset type' })).toHaveClass('ui-select-trigger');
    expect(view.getByRole('combobox', { name: 'Asset state' })).toHaveClass('ui-select-trigger');
    expect(view.getByRole('combobox', { name: 'Sort assets' })).toHaveClass('ui-select-trigger');
    expect(view.getByLabelText('Grid view')).toHaveClass('ui-icon-button');
    expect(view.getByLabelText('List view')).toHaveClass('ui-icon-button');
  });

  it('persists an adjustable Content Browser tree width', () => {
    localStorage.setItem('arc.content.treeWidth', '244');
    const view = renderBrowser();
    const separator = view.getByRole('separator', { name: 'Resize content folder tree' });

    expect(separator).toHaveAttribute('aria-valuenow', '244');
    fireEvent.keyDown(separator, { key: 'ArrowRight' });
    expect(separator).toHaveAttribute('aria-valuenow', '256');
    expect(localStorage.getItem('arc.content.treeWidth')).toBe('256');

    fireEvent.doubleClick(separator);
    expect(separator).toHaveAttribute('aria-valuenow', '190');
    expect(localStorage.getItem('arc.content.treeWidth')).toBe('190');
  });

  it('filters registry assets and emits GUID drag payloads', () => {
""",
)

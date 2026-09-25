import { useEffect, useMemo, useRef, useState } from 'react';

import { allCommands } from '../app/commandRegistry';
import { dispatchWorkbenchCommand } from '../app/commandDispatcher';
import type { AssetItem, SceneEntity } from '../services/editorHostTypes';
import { UiDrawerPanel, UiSearchHeader, UiSearchList } from '../ui';
import { AssetSearchEntity, CommandSearchEntity, type SearchEntity } from './SearchEntity';
import { subscribeSearchDrawerRequests, type SearchDrawerMode } from './searchDrawerRoute';

import './SearchPanel.css';

const resultLimit = 200;

export function SearchPanel({
  assets,
  onSelectAsset,
}: {
  entities: SceneEntity[];
  assets: AssetItem[];
  onSelectEntity: (id: string) => void;
  onSelectAsset: (id: string) => void;
}) {
  const [mode, setMode] = useState<SearchDrawerMode>('assets');
  const [query, setQuery] = useState('');
  const [focusRequest, setFocusRequest] = useState(0);
  const inputRef = useRef<HTMLInputElement | null>(null);

  const assetEntities = useMemo(() => assets.map((asset) => new AssetSearchEntity(asset)), [assets]);
  const commandEntities = useMemo(() => allCommands.map((command) => new CommandSearchEntity(command)), []);
  const activeEntities = mode === 'assets' ? assetEntities : commandEntities;
  const results = useMemo(() => activeEntities.filter((entity) => entity.matches(query)), [activeEntities, query]);
  const visibleResults = results.slice(0, resultLimit);

  useEffect(
    () =>
      subscribeSearchDrawerRequests((nextMode) => {
        setMode(nextMode);
        setFocusRequest((request) => request + 1);
      }),
    [],
  );

  useEffect(() => {
    if (focusRequest === 0) return;
    inputRef.current?.focus();
    inputRef.current?.select();
  }, [focusRequest, mode]);

  const focusMode = (nextMode: SearchDrawerMode) => {
    setMode(nextMode);
    setFocusRequest((request) => request + 1);
  };

  const activate = (entity: SearchEntity) => {
    if (entity instanceof AssetSearchEntity) {
      onSelectAsset(entity.asset.id);
      return;
    }
    if (!(entity instanceof CommandSearchEntity)) return;
    if (entity.command.id === 'view.assetSearch') {
      focusMode('assets');
      return;
    }
    if (entity.command.id === 'view.commandPalette') {
      focusMode('commands');
      return;
    }
    dispatchWorkbenchCommand(entity.command.id);
  };

  return (
    <UiDrawerPanel className="search-panel">
      <UiSearchHeader
        inputRef={inputRef}
        mode={mode}
        modes={[
          { id: 'assets', label: 'Assets', count: assetEntities.length },
          { id: 'commands', label: 'Commands', count: commandEntities.length },
        ]}
        placeholder={mode === 'assets' ? 'Search assets…' : 'Search commands…'}
        query={query}
        resultCount={results.length}
        searchLabel={mode === 'assets' ? 'Search assets' : 'Search commands'}
        title="Search"
        onModeChange={(nextMode) => setMode(nextMode as SearchDrawerMode)}
        onQueryChange={setQuery}
      />
      <UiSearchList
        ariaLabel={mode === 'assets' ? 'Asset search results' : 'Command search results'}
        emptyMessage={mode === 'assets' ? 'No matching assets' : 'No matching commands'}
        items={visibleResults}
        onActivate={activate}
      />
      {results.length > visibleResults.length && (
        <footer className="search-panel-limit">
          Showing the first {visibleResults.length} of {results.length} results
        </footer>
      )}
    </UiDrawerPanel>
  );
}

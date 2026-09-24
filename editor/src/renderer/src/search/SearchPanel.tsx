import { useMemo, useState } from 'react';

import { allCommands } from '../app/commandRegistry';
import type { CommandContext, CommandId } from '../app/workbenchTypes';
import type { AssetItem } from '../services/editorHostTypes';
import { UiDrawerPanel, UiSearchHeader, UiSearchList } from '../ui';
import { AssetSearchEntity, CommandSearchEntity, type SearchEntity } from './SearchEntity';

import './SearchPanel.css';

type SearchMode = 'assets' | 'commands';

const resultLimit = 200;

export function SearchPanel({
  assets,
  commandContext,
  onSelectAsset,
  onCommand,
}: {
  assets: AssetItem[];
  commandContext: CommandContext;
  onSelectAsset: (id: string) => void;
  onCommand: (command: CommandId) => void;
}) {
  const [mode, setMode] = useState<SearchMode>('assets');
  const [query, setQuery] = useState('');

  const assetEntities = useMemo(() => assets.map((asset) => new AssetSearchEntity(asset)), [assets]);
  const commandEntities = useMemo(
    () => allCommands.map((command) => new CommandSearchEntity(command, commandContext)),
    [commandContext],
  );
  const activeEntities = mode === 'assets' ? assetEntities : commandEntities;
  const results = useMemo(() => activeEntities.filter((entity) => entity.matches(query)), [activeEntities, query]);
  const visibleResults = results.slice(0, resultLimit);

  const activate = (entity: SearchEntity) => {
    if (entity instanceof AssetSearchEntity) {
      onSelectAsset(entity.asset.id);
      return;
    }
    if (entity instanceof CommandSearchEntity) onCommand(entity.command.id);
  };

  return (
    <UiDrawerPanel className="search-panel">
      <UiSearchHeader
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
        onModeChange={(nextMode) => setMode(nextMode as SearchMode)}
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

import { useMemo, useState } from 'react';

import { UiSearchHeader, UiSearchList } from '../ui';
import type { UiSearchListItem } from '../ui';

const assetResults: readonly UiSearchListItem[] = [
  {
    id: 'asset:wood',
    variant: 'asset',
    title: 'M_Warm_Wood.arcmat',
    subtitle: 'Content/Materials/M_Warm_Wood.arcmat',
    meta: 'material',
    state: 'ready',
  },
  {
    id: 'asset:crate',
    variant: 'asset',
    title: 'SM_Shipping_Crate.glb',
    subtitle: 'Content/Models/SM_Shipping_Crate.glb',
    meta: 'scene',
    state: 'importing',
  },
  {
    id: 'asset:albedo',
    variant: 'asset',
    title: 'T_Mountain_Albedo.png',
    subtitle: 'Content/Textures/T_Mountain_Albedo.png',
    meta: 'texture',
    state: 'stale',
  },
];

const commandResults: readonly UiSearchListItem[] = [
  {
    id: 'command:save',
    variant: 'command',
    title: 'Save Scene',
    subtitle: 'Save the active scene.',
    meta: 'File',
    shortcut: 'Ctrl+S',
  },
  {
    id: 'command:frame',
    variant: 'command',
    title: 'Frame Selected',
    subtitle: 'Focus the active viewport on the current selection.',
    meta: 'Viewport',
    shortcut: 'F',
  },
  {
    id: 'command:undo',
    variant: 'command',
    title: 'Undo',
    subtitle: 'Undo the last scene edit.',
    meta: 'Edit',
    shortcut: 'Ctrl+Z',
    disabled: true,
    disabledReason: 'There is nothing to undo',
  },
];

export function UiLabSearchShowcase() {
  const [mode, setMode] = useState<'assets' | 'commands'>('assets');
  const [query, setQuery] = useState('');
  const source = mode === 'assets' ? assetResults : commandResults;
  const results = useMemo(() => {
    const normalized = query.trim().toLocaleLowerCase();
    if (!normalized) return source;
    return source.filter((item) =>
      `${item.title} ${item.subtitle ?? ''} ${item.meta ?? ''} ${item.state ?? ''}`
        .toLocaleLowerCase()
        .includes(normalized),
    );
  }, [query, source]);

  return (
    <div className="ui-lab-search-showcase">
      <UiSearchHeader
        mode={mode}
        modes={[
          { id: 'assets', label: 'Assets', count: assetResults.length },
          { id: 'commands', label: 'Commands', count: commandResults.length },
        ]}
        placeholder={mode === 'assets' ? 'Search assets…' : 'Search commands…'}
        query={query}
        resultCount={results.length}
        searchLabel="UI Lab search"
        onModeChange={(nextMode) => setMode(nextMode as typeof mode)}
        onQueryChange={setQuery}
      />
      <UiSearchList items={results} onActivate={() => undefined} />
    </div>
  );
}

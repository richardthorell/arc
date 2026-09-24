// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { registerWorkbenchCommandHandler } from '../app/commandDispatcher';
import type { AssetItem } from '../services/editorHostTypes';
import { SearchPanel } from './SearchPanel';

const assets: AssetItem[] = [
  {
    id: 'material-wood',
    name: 'M_Warm_Wood.arcmat',
    path: 'Content/Materials/M_Warm_Wood.arcmat',
    kind: 'material',
    status: 'ready',
  },
  {
    id: 'texture-rock',
    name: 'T_Rock_Albedo.png',
    path: 'Content/Textures/T_Rock_Albedo.png',
    kind: 'texture',
    status: 'stale',
  },
];

let unregister: (() => void) | undefined;

afterEach(() => {
  unregister?.();
  unregister = undefined;
  cleanup();
});

describe('SearchPanel', () => {
  it('starts in asset mode and filters asset results', () => {
    const onSelectAsset = vi.fn();
    render(
      <SearchPanel assets={assets} entities={[]} onSelectAsset={onSelectAsset} onSelectEntity={() => undefined} />,
    );

    expect(screen.getByRole('tab', { name: /Assets/ })).toHaveAttribute('aria-selected', 'true');
    fireEvent.change(screen.getByRole('searchbox', { name: 'Search assets' }), { target: { value: 'wood' } });
    expect(screen.getByText('M_Warm_Wood.arcmat')).toBeInTheDocument();
    expect(screen.queryByText('T_Rock_Albedo.png')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('listitem', { name: /M_Warm_Wood/ }));
    expect(onSelectAsset).toHaveBeenCalledWith('material-wood');
  });

  it('switches to commands and runs a matching workbench command', () => {
    const handler = vi.fn();
    unregister = registerWorkbenchCommandHandler(handler);
    render(
      <SearchPanel assets={assets} entities={[]} onSelectAsset={() => undefined} onSelectEntity={() => undefined} />,
    );

    fireEvent.click(screen.getByRole('tab', { name: /Commands/ }));
    fireEvent.change(screen.getByRole('searchbox', { name: 'Search commands' }), { target: { value: 'save scene' } });
    fireEvent.click(screen.getByRole('listitem', { name: /Save Scene/ }));

    expect(handler).toHaveBeenCalledWith('file.save');
  });
});

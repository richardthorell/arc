// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { TerrainToolsPanel } from './TerrainToolsPanel';
import type { TerrainToolState } from './TerrainToolsPanel';

const terrain = {
  enabled: true,
  size: 180,
  resolution: 257,
  chunkQuads: 128,
  receiveShadows: true,
  contentRevision: 1,
  brushTool: 'sculpt' as const,
  brushRadius: 6,
  brushStrength: 0.25,
  brushFalloff: 1,
  activeLayer: 0,
  layers: ['Grass', 'Dirt', 'Rock', 'Sand'].map((name) => ({ name, baseColorPath: `textures/${name}.png` })),
};

const initialState: TerrainToolState = {
  entity: { index: 4, generation: 1 },
  active: true,
  hoverVisible: false,
  tool: 'sculpt',
  radius: 6,
  strength: 0.25,
  falloff: 1,
  activeLayer: 0,
};

afterEach(cleanup);

describe('TerrainToolsPanel', () => {
  it('keeps brush controls outside the Inspector and sends host-authoritative tool changes', async () => {
    const command = vi.fn(async (_type: string, payload: unknown) => ({
      succeeded: true,
      payload: { ...initialState, ...(payload as Partial<TerrainToolState>) },
    }));
    const onStateChange = vi.fn();
    const { rerender } = render(<TerrainToolsPanel assets={[]} command={command} onStateChange={onStateChange}
      state={initialState} terrain={terrain} />);

    await userEvent.click(screen.getByRole('tab', { name: /Paint/ }));
    await waitFor(() => expect(command).toHaveBeenCalledWith('terrain.setBrush',
      expect.objectContaining({ entity: initialState.entity, tool: 'paint' })));

    const paintState = { ...initialState, tool: 'paint' as const };
    onStateChange.mockClear();
    command.mockClear();
    rerender(<TerrainToolsPanel assets={[]} command={command} onStateChange={onStateChange}
      state={paintState} terrain={terrain} />);
    await userEvent.click(screen.getByLabelText('Paint Rock'));
    await waitFor(() => expect(command).toHaveBeenCalledWith('terrain.setBrush',
      expect.objectContaining({ tool: 'paint', activeLayer: 2 })));
  });

  it('updates brush ranges through the terrain tool contract', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true, payload: initialState });
    render(<TerrainToolsPanel assets={[]} command={command} onStateChange={() => undefined}
      state={initialState} terrain={terrain} />);

    const radius = screen.getByLabelText('Radius');
    fireEvent.change(radius, { target: { value: '6.25' } });
    await waitFor(() => expect(command).toHaveBeenCalledWith('terrain.setBrush',
      expect.objectContaining({ radius: 6.25 })));
  });
});

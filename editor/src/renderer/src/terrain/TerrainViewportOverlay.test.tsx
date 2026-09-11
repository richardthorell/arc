// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { TerrainViewportOverlay } from './TerrainViewportOverlay';
import type { TerrainToolState } from './TerrainViewportOverlay';

const terrain = {
  enabled: true,
  size: 180,
  resolution: 257,
  chunkQuads: 128,
  patchQuads: 32,
  maximumHierarchyDepth: 0,
  geometricErrorMultiplier: 1,
  receiveShadows: true,
  castShadows: true,
  shadowLodBias: 0,
  maximumShadowDistance: 0,
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

describe('TerrainViewportOverlay', () => {
  it('keeps contextual sculpt and paint interaction in the viewport', async () => {
    const command = vi.fn(async (_type: string, payload: unknown) => ({
      succeeded: true,
      payload: { ...initialState, ...(payload as Partial<TerrainToolState>) },
    }));
    const onStateChange = vi.fn();
    const { rerender } = render(
      <TerrainViewportOverlay
        assets={[]}
        command={command}
        onStateChange={onStateChange}
        state={initialState}
        terrain={terrain}
      />,
    );

    await userEvent.click(screen.getByRole('tab', { name: /Paint/ }));
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith(
        'terrain.setBrush',
        expect.objectContaining({ entity: initialState.entity, tool: 'paint' }),
      ),
    );

    command.mockClear();
    rerender(
      <TerrainViewportOverlay
        assets={[]}
        command={command}
        onStateChange={onStateChange}
        state={{ ...initialState, tool: 'paint' }}
        terrain={terrain}
      />,
    );
    await userEvent.click(screen.getByLabelText('Paint Rock'));
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith(
        'terrain.setBrush',
        expect.objectContaining({ tool: 'paint', activeLayer: 2 }),
      ),
    );
  });

  it('sends brush range changes through the existing host-authoritative brush contract', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true, payload: initialState });
    render(
      <TerrainViewportOverlay
        assets={[]}
        command={command}
        onStateChange={() => undefined}
        state={initialState}
        terrain={terrain}
      />,
    );

    fireEvent.change(screen.getByLabelText('Radius'), { target: { value: '9.5' } });
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('terrain.setBrush', expect.objectContaining({ radius: 9.5 })),
    );
  });
});

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { TerrainStackPanel } from './TerrainStackPanel';
import type { TerrainModifierStackSnapshot } from './TerrainStackPanel';

const entity = { index: 7, generation: 2 };
const sculpt = {
  id: '11111111-1111-4111-8111-111111111111',
  name: 'Large Forms',
  type: 'sculpt' as const,
  typeId: 'arc.terrain.sculpt_layer.v1',
  enabled: true,
  regionPayloads: 2,
};
const paint = {
  id: '22222222-2222-4222-8222-222222222222',
  name: 'Ground',
  type: 'paint' as const,
  typeId: 'arc.terrain.paint_layer.v1',
  enabled: true,
  regionPayloads: 1,
};

const snapshot: TerrainModifierStackSnapshot = {
  entity,
  assetBacked: true,
  readOnly: false,
  assetPath: 'Content/Terrain/Valley.terrain',
  authoringRevision: 4,
  activeModifier: sculpt.id,
  modifiers: [sculpt, paint],
  rebuild: {
    state: 'idle',
    authoringRevision: 4,
    dirtyRegions: 0,
    geometryRegions: 0,
    attributeRegions: 0,
    error: '',
  },
};

afterEach(cleanup);

describe('TerrainStackPanel', () => {
  it('loads the asset-owned stack and sends stable-id modifier operations', async () => {
    const command = vi.fn(async (_type: string, payload: unknown) => {
      const operation = (payload as { operation: string }).operation;
      if (operation === 'add_sculpt') {
        return {
          succeeded: true,
          payload: {
            ...snapshot,
            authoringRevision: 5,
            modifiers: [
              ...snapshot.modifiers,
              {
                ...sculpt,
                id: '33333333-3333-4333-8333-333333333333',
                name: 'Sculpt Layer',
                regionPayloads: 0,
              },
            ],
          },
        };
      }
      return { succeeded: true, payload: snapshot };
    });

    render(<TerrainStackPanel command={command} entity={entity} />);

    expect(await screen.findByText('Large Forms')).toBeInTheDocument();
    expect(screen.getByText('Ground')).toBeInTheDocument();
    expect(screen.getByText('Base Source')).toBeInTheDocument();
    expect(await screen.findByLabelText('Modifier name')).toHaveClass('ui-text-input');
    expect(command).toHaveBeenCalledWith('terrain.modifierStack', { entity, operation: 'inspect' });

    await userEvent.click(screen.getByRole('button', { name: /Sculpt/ }));
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('terrain.modifierStack', {
        entity,
        operation: 'add_sculpt',
      }),
    );
  });

  it('uses modifier stable ids when selecting the sculpt target', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true, payload: { ...snapshot, activeModifier: paint.id } });
    render(<TerrainStackPanel command={command} entity={entity} />);

    await screen.findByText('Ground');
    await userEvent.click(screen.getByRole('option', { name: /Ground/ }));

    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('terrain.modifierStack', {
        entity,
        operation: 'select',
        modifier: paint.id,
      }),
    );
  });

  it('uses modifier stable ids when toggling visibility', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true, payload: snapshot });
    render(<TerrainStackPanel command={command} entity={entity} />);

    await screen.findByText('Large Forms');
    await userEvent.click(screen.getByRole('button', { name: 'Disable Large Forms' }));

    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('terrain.modifierStack', {
        entity,
        operation: 'set_enabled',
        modifier: sculpt.id,
        enabled: false,
      }),
    );
  });

  it('duplicates the selected modifier and supports direct drag reordering', async () => {
    const duplicated = { ...sculpt, id: '33333333-3333-4333-8333-333333333333', name: 'Large Forms Copy' };
    const command = vi.fn(async (_type: string, payload: unknown) => {
      const operation = (payload as { operation: string }).operation;
      return {
        succeeded: true,
        payload:
          operation === 'duplicate'
            ? { ...snapshot, activeModifier: duplicated.id, modifiers: [sculpt, duplicated, paint] }
            : snapshot,
      };
    });
    render(<TerrainStackPanel command={command} entity={entity} />);

    await screen.findByText('Large Forms');
    await userEvent.click(await screen.findByRole('button', { name: 'Duplicate Modifier' }));
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('terrain.modifierStack', {
        entity,
        operation: 'duplicate',
        modifier: sculpt.id,
      }),
    );

    const source = screen.getByRole('option', { name: /Large Forms Copy/ });
    const target = screen.getByRole('option', { name: /Ground/ });
    const data = new Map<string, string>();
    const dataTransfer = {
      effectAllowed: 'none',
      setData: (type: string, value: string) => data.set(type, value),
      getData: (type: string) => data.get(type) ?? '',
    };
    fireEvent.dragStart(source, { dataTransfer });
    fireEvent.dragOver(target, { dataTransfer });
    fireEvent.drop(target, { dataTransfer });
    await waitFor(() =>
      expect(command).toHaveBeenCalledWith('terrain.modifierStack', {
        entity,
        operation: 'move',
        modifier: duplicated.id,
        index: 2,
      }),
    );
  });

  it('reports incremental rebuild domains and persistent failures', async () => {
    const command = vi.fn().mockResolvedValue({
      succeeded: true,
      payload: {
        ...snapshot,
        rebuild: {
          state: 'failed',
          authoringRevision: 7,
          dirtyRegions: 3,
          geometryRegions: 2,
          attributeRegions: 1,
          error: 'Heightmap source is unavailable',
        },
      },
    });
    render(<TerrainStackPanel command={command} entity={entity} />);

    expect(await screen.findByRole('alert')).toHaveTextContent('Terrain rebuild failed');
    expect(screen.getByRole('alert')).toHaveTextContent('2 geometry · 1 attribute');
    expect(screen.getByRole('alert')).toHaveTextContent('Heightmap source is unavailable');
  });
});

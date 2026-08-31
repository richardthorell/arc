import { describe, expect, it, vi } from 'vitest';

import { assignDroppedMaterialToViewport, instantiateDroppedMeshInViewport } from './viewportAssetDrop';

describe('assignDroppedMaterialToViewport', () => {
  it('waits for the viewport pick frame and assigns the material to the picked entity', async () => {
    const command = vi.fn(async (type: string) => {
      if (type === 'viewport.pick') return { succeeded: true, frameRevision: 10 };
      if (type === 'entity.setMaterial') return { succeeded: true, frameRevision: 11 };
      return { succeeded: true };
    });
    const query = vi
      .fn()
      .mockResolvedValueOnce({
        succeeded: true,
        frameRevision: 10,
        payload: { entity: { index: 3, generation: 1 }, selectionCount: 1 },
      })
      .mockResolvedValueOnce({
        succeeded: true,
        frameRevision: 11,
        payload: { entity: { index: 7, generation: 2 }, selectionCount: 1 },
      });

    const result = await assignDroppedMaterialToViewport(
      { command, query },
      { viewportId: 'viewport-1', x: 320, y: 240, path: 'Content/Materials/Hero.arcmat' },
      { sleep: vi.fn().mockResolvedValue(undefined), intervalMs: 0, attempts: 3 },
    );

    expect(result).toEqual({ succeeded: true, entity: { index: 7, generation: 2 } });
    expect(command).toHaveBeenNthCalledWith(1, 'viewport.pick', { viewportId: 'viewport-1', x: 320, y: 240 });
    expect(command).toHaveBeenNthCalledWith(2, 'entity.setMaterial', {
      entity: { index: 7, generation: 2 },
      path: 'Content/Materials/Hero.arcmat',
    });
  });

  it('does not assign a material when the pick resolves to empty space', async () => {
    const command = vi.fn().mockResolvedValue({ succeeded: true, frameRevision: 4 });
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      frameRevision: 5,
      payload: { entity: { index: 0xffffffff, generation: 0 }, selectionCount: 0 },
    });

    const result = await assignDroppedMaterialToViewport(
      { command, query },
      { viewportId: 'viewport-1', x: 10, y: 12, path: 'Content/Materials/Hero.arcmat' },
      { attempts: 1 },
    );

    expect(result).toEqual({ succeeded: false, error: 'No scene object at the drop position' });
    expect(command).toHaveBeenCalledTimes(1);
  });
});

describe('instantiateDroppedMeshInViewport', () => {
  it('creates and places an existing mesh asset at the viewport drop point', async () => {
    const entity = { index: 8, generation: 2 };
    const command = vi
      .fn()
      .mockResolvedValueOnce({ succeeded: true, payload: { worldPosition: [1, 2, 3] } })
      .mockResolvedValueOnce({ succeeded: true, payload: { entity } })
      .mockResolvedValue({ succeeded: true });
    const result = await instantiateDroppedMeshInViewport(
      { command, query: vi.fn() },
      { viewportId: 'viewport-1', x: 10, y: 20, path: 'Content/Models/crate.obj' },
    );
    expect(result).toEqual({ succeeded: true, entity });
    expect(command).toHaveBeenNthCalledWith(1, 'viewport.pick', { viewportId: 'viewport-1', x: 10, y: 20 });
    expect(command).toHaveBeenNthCalledWith(2, 'entity.create', { kind: 'empty' });
    expect(command).toHaveBeenNthCalledWith(3, 'entity.setMaterial', {
      entity,
      path: '__arc_mesh__/Content/Models/crate.obj',
    });
    expect(command).toHaveBeenNthCalledWith(4, 'entity.setTransform', {
      entity,
      transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    });
    expect(command).toHaveBeenNthCalledWith(5, 'entity.rename', { entity, name: 'crate' });
  });
});

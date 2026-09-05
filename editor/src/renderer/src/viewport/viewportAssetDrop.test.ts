import { describe, expect, it, vi } from 'vitest';

import {
  assignDroppedMaterialToViewport,
  instantiateDroppedMeshInViewport,
  viewportMeshDropIntent,
} from './viewportAssetDrop';

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

describe('viewportMeshDropIntent', () => {
  it('maps plain, Ctrl, and Shift drops to explicit model operations', () => {
    expect(viewportMeshDropIntent({})).toBe('create');
    expect(viewportMeshDropIntent({ control: true })).toBe('createChild');
    expect(viewportMeshDropIntent({ shift: true })).toBe('replace');
    expect(viewportMeshDropIntent({ control: true, shift: true })).toBe('replace');
  });
});

describe('instantiateDroppedMeshInViewport', () => {
  const resolvedPickQuery = (entity = { index: 4, generation: 1 }) =>
    vi.fn().mockResolvedValue({
      succeeded: true,
      frameRevision: 11,
      payload: { entity, selectionCount: 1, name: 'Target' },
    });

  it('creates a root model entity with Default Phong and places it at the drop point', async () => {
    const entity = { index: 8, generation: 2 };
    const command = vi
      .fn()
      .mockResolvedValueOnce({ succeeded: true, frameRevision: 10, payload: { worldPosition: [1, 2, 3] } })
      .mockResolvedValueOnce({ succeeded: true, payload: { entity } })
      .mockResolvedValue({ succeeded: true });

    const result = await instantiateDroppedMeshInViewport(
      { command, query: resolvedPickQuery() },
      { viewportId: 'viewport-1', x: 10, y: 20, path: 'Content/Models/crate.obj' },
      { attempts: 1 },
    );

    expect(result).toEqual({ succeeded: true, entity, intent: 'create' });
    expect(command).toHaveBeenNthCalledWith(1, 'viewport.pick', { viewportId: 'viewport-1', x: 10, y: 20 });
    // A primitive shell supplies MeshRenderer + Default Phong; assigning the model clears procedural metadata.
    expect(command).toHaveBeenNthCalledWith(2, 'entity.create', { kind: 'cube' });
    expect(command).toHaveBeenNthCalledWith(3, 'entity.setMaterial', {
      entity,
      path: '__arc_mesh__/Content/Models/crate.obj',
    });
    expect(command).toHaveBeenNthCalledWith(4, 'entity.setTransform', {
      entity,
      transform: { position: [1, 2, 3], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    });
    expect(command).toHaveBeenNthCalledWith(5, 'entity.rename', { entity, name: 'crate' });
    expect(command).not.toHaveBeenCalledWith('entity.reparent', expect.anything());
  });

  it('Ctrl+drop creates a child and preserves its drop-space world transform', async () => {
    const target = { index: 5, generation: 3 };
    const entity = { index: 9, generation: 1 };
    const command = vi
      .fn()
      .mockResolvedValueOnce({ succeeded: true, frameRevision: 20, payload: { worldPosition: [4, 0, -2] } })
      .mockResolvedValueOnce({ succeeded: true, payload: { entity } })
      .mockResolvedValue({ succeeded: true });
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      frameRevision: 21,
      payload: { entity: target, selectionCount: 1, name: 'Table' },
    });

    const result = await instantiateDroppedMeshInViewport(
      { command, query },
      { viewportId: 'viewport-1', x: 15, y: 25, path: 'Content/Models/cup.glb', control: true },
      { attempts: 1 },
    );

    expect(result).toEqual({ succeeded: true, entity, intent: 'createChild' });
    expect(command).toHaveBeenCalledWith('entity.setTransform', {
      entity,
      transform: { position: [4, 0, -2], rotation: [0, 0, 0, 1], scale: [1, 1, 1] },
    });
    expect(command).toHaveBeenCalledWith('entity.reparent', {
      entity,
      parent: target,
      preserveWorld: true,
    });
  });

  it('Ctrl+drop onto empty space creates at the scene root', async () => {
    const entity = { index: 10, generation: 1 };
    const command = vi
      .fn()
      .mockResolvedValueOnce({ succeeded: true, frameRevision: 30, payload: { worldPosition: [0, 0, 4] } })
      .mockResolvedValueOnce({ succeeded: true, payload: { entity } })
      .mockResolvedValue({ succeeded: true });
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      frameRevision: 31,
      payload: { entity: { index: 0xffffffff, generation: 0 }, selectionCount: 0 },
    });

    const result = await instantiateDroppedMeshInViewport(
      { command, query },
      { viewportId: 'viewport-1', x: 30, y: 40, path: 'Content/Models/lamp.fbx', control: true },
      { attempts: 1 },
    );

    expect(result).toEqual({ succeeded: true, entity, intent: 'createChild' });
    expect(command).not.toHaveBeenCalledWith('entity.reparent', expect.anything());
  });

  it('Shift+drop replaces only the target mesh without creating or changing its material', async () => {
    const target = { index: 11, generation: 4 };
    const command = vi.fn().mockResolvedValueOnce({ succeeded: true, frameRevision: 40 }).mockResolvedValue({
      succeeded: true,
    });
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      frameRevision: 41,
      payload: { entity: target, selectionCount: 1, name: 'Chair' },
    });

    const result = await instantiateDroppedMeshInViewport(
      { command, query },
      { viewportId: 'viewport-1', x: 50, y: 60, path: 'Content/Models/chair.glb', shift: true },
      { attempts: 1 },
    );

    expect(result).toEqual({ succeeded: true, entity: target, intent: 'replace' });
    expect(command).toHaveBeenNthCalledWith(2, 'entity.setMaterial', {
      entity: target,
      path: '__arc_mesh__/Content/Models/chair.glb',
    });
    expect(command).not.toHaveBeenCalledWith('entity.create', expect.anything());
    expect(command).not.toHaveBeenCalledWith('entity.setTransform', expect.anything());
    expect(command).not.toHaveBeenCalledWith('entity.rename', expect.anything());
  });

  it('rejects Shift+drop over empty space', async () => {
    const command = vi.fn().mockResolvedValueOnce({ succeeded: true, frameRevision: 50 });
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      frameRevision: 51,
      payload: { entity: { index: 0xffffffff, generation: 0 }, selectionCount: 0 },
    });

    const result = await instantiateDroppedMeshInViewport(
      { command, query },
      { viewportId: 'viewport-1', x: 70, y: 80, path: 'Content/Models/chair.glb', shift: true },
      { attempts: 1 },
    );

    expect(result).toEqual({ succeeded: false, error: 'Shift+drop requires a mesh target' });
    expect(command).toHaveBeenCalledTimes(1);
  });
});

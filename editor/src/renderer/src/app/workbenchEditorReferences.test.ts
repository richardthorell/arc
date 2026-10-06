import { describe, expect, it, vi } from 'vitest';

import type { ProjectSnapshot } from '../services/editorHostTypes';
import { createWorkbenchEditorReferenceController } from './workbenchEditorReferences';

const project: ProjectSnapshot = {
  name: 'Test',
  root: 'D:/Test',
  assetRoot: 'Content',
  activeScene: 'Content/Scenes/Main.arcscene',
  scene: [
    {
      id: '12:3',
      guid: 'player-guid',
      name: 'Player',
      kind: 'mesh',
      active: true,
    },
  ],
  assets: [
    {
      id: 'material-guid',
      guid: 'material-guid',
      name: 'brushed_metal.arcmat',
      title: 'Brushed Metal',
      path: 'Materials/brushed_metal.arcmat',
      kind: 'material',
      status: 'ready',
    },
  ],
  console: [],
  renderStats: {
    fps: 0,
    frameTimeMs: 0,
    drawCalls: 0,
    triangles: 0,
    visibleEntities: 0,
    lights: 0,
    gpuMemoryMb: 0,
  },
};

describe('createWorkbenchEditorReferenceController', () => {
  it('resolves and activates entity GUIDs through workbench selection', async () => {
    const selectEntity = vi.fn();
    const focusSelectedEntity = vi.fn();
    const controller = createWorkbenchEditorReferenceController({
      getProject: () => project,
      selectEntity,
      focusSelectedEntity,
      selectAsset: vi.fn(),
      openAsset: vi.fn(),
    });
    const reference = { kind: 'entity' as const, id: 'player-guid' };

    await expect(Promise.resolve(controller.resolve(reference))).resolves.toMatchObject({
      label: 'Player',
      subtitle: 'Mesh',
    });
    await controller.activate(reference);
    await controller.focus?.(reference);

    expect(selectEntity).toHaveBeenCalledWith('12:3');
    expect(focusSelectedEntity).toHaveBeenCalledTimes(1);
  });

  it('loads asset thumbnails through the ARC resource registry', async () => {
    const resources = {
      read: vi.fn(async (uri: string) => ({
        uri: {} as never,
        mediaType: 'image/png',
        dataUrl: 'data:image/png;base64,material-thumb',
      })),
    };
    const controller = createWorkbenchEditorReferenceController({
      getProject: () => project,
      selectEntity: vi.fn(),
      focusSelectedEntity: vi.fn(),
      selectAsset: vi.fn(),
      openAsset: vi.fn(),
      resources,
    });
    const reference = { kind: 'asset' as const, id: 'material-guid' };

    await expect(Promise.resolve(controller.resolve(reference))).resolves.toMatchObject({
      label: 'Brushed Metal',
      subtitle: 'Material',
      thumbnailUrl: 'data:image/png;base64,material-thumb',
    });
    expect(resources.read).toHaveBeenCalledWith('arc://asset/material-guid/thumbnail?size=64');
  });

  it('resolves asset GUIDs and opens the asset only for focus', async () => {
    const selectAsset = vi.fn();
    const openAsset = vi.fn();
    const controller = createWorkbenchEditorReferenceController({
      getProject: () => project,
      selectEntity: vi.fn(),
      focusSelectedEntity: vi.fn(),
      selectAsset,
      openAsset,
    });
    const reference = { kind: 'asset' as const, id: 'material-guid' };

    await expect(Promise.resolve(controller.resolve(reference))).resolves.toMatchObject({
      label: 'Brushed Metal',
      subtitle: 'Material',
    });
    await controller.activate(reference);
    expect(selectAsset).toHaveBeenCalledTimes(1);
    expect(openAsset).not.toHaveBeenCalled();

    await controller.focus?.(reference);
    expect(selectAsset).toHaveBeenCalledTimes(2);
    expect(openAsset).toHaveBeenCalledWith(project.assets[0]);
  });
});

import { describe, expect, it, vi } from 'vitest';

import { createArcAssetResourceHandler } from './arcAssetResources';
import { ArcResourceRegistry } from './arcResourceRegistry';

const assets = [
  {
    guid: 'material-guid',
    title: 'Brushed Metal',
    name: 'brushed_metal.arcmat',
    path: 'Materials/brushed_metal.arcmat',
    kind: 'material',
    scope: 'project',
    state: 'ready',
    generation: 7,
  },
];

describe('ARC asset resources', () => {
  it('resolves project asset metadata by stable GUID without exposing filesystem roots', async () => {
    const registry = new ArcResourceRegistry();
    registry.register(
      createArcAssetResourceHandler({
        listAssets: async () => assets,
        loadThumbnail: async () => null,
      }),
    );

    await expect(registry.resolve('arc://asset/material-guid')).resolves.toMatchObject({
      label: 'Brushed Metal',
      subtitle: 'Material',
      generation: 7,
      metadata: {
        guid: 'material-guid',
        path: 'Materials/brushed_metal.arcmat',
        scope: 'project',
        state: 'ready',
      },
    });
  });

  it('loads thumbnail subresources through the same registered asset provider', async () => {
    const loadThumbnail = vi.fn(async (path: string, maxSize: number) => ({
      path,
      width: maxSize,
      height: maxSize,
      dataUrl: 'data:image/png;base64,thumbnail',
    }));
    const registry = new ArcResourceRegistry();
    registry.register(
      createArcAssetResourceHandler({
        listAssets: async () => assets,
        loadThumbnail,
      }),
    );

    await expect(registry.read('arc://asset/material-guid/thumbnail?size=96')).resolves.toMatchObject({
      mediaType: 'image/png',
      dataUrl: 'data:image/png;base64,thumbnail',
      generation: 7,
      metadata: {
        width: 96,
        height: 96,
        path: 'Materials/brushed_metal.arcmat',
      },
    });
    expect(loadThumbnail).toHaveBeenCalledWith('Materials/brushed_metal.arcmat', 96);
  });

  it('rejects unknown subresources and invalid or unsupported thumbnail parameters', async () => {
    const loadThumbnail = vi.fn();
    const registry = new ArcResourceRegistry();
    registry.register(
      createArcAssetResourceHandler({
        listAssets: async () => assets,
        loadThumbnail,
      }),
    );

    await expect(registry.read('arc://asset/material-guid/source')).resolves.toBeNull();
    await expect(registry.read('arc://asset/material-guid/thumbnail?size=8')).resolves.toBeNull();
    await expect(registry.read('arc://asset/material-guid/thumbnail?fit=crop')).resolves.toBeNull();
    expect(loadThumbnail).not.toHaveBeenCalled();
  });
});

import { describe, expect, it, vi } from 'vitest';

import { ensureExternalAssetImported } from './externalAssetImportLifecycle';

describe('external asset import lifecycle', () => {
  it('requests the initial import when a newly discovered asset is stale', async () => {
    const reimportAsset = vi.fn().mockResolvedValue({ succeeded: true });
    const queryAssets = vi
      .fn()
      .mockResolvedValueOnce([])
      .mockResolvedValueOnce([{ guid: 'texture-guid', path: 'Content/Bricks.jpg', state: 'stale' as const }])
      .mockResolvedValueOnce([{ guid: 'texture-guid', path: 'Content/Bricks.jpg', state: 'importing' as const }])
      .mockResolvedValueOnce([{ guid: 'texture-guid', path: 'Content/Bricks.jpg', state: 'ready' as const }]);

    const asset = await ensureExternalAssetImported(
      'Content/Bricks.jpg',
      'texture',
      { queryAssets, reimportAsset },
      { sleep: async () => undefined },
    );

    expect(asset.state).toBe('ready');
    expect(reimportAsset).toHaveBeenCalledTimes(1);
    expect(reimportAsset).toHaveBeenCalledWith('texture-guid');
  });

  it('does not reimport an asset that became ready before discovery polling observed it', async () => {
    const reimportAsset = vi.fn();
    const queryAssets = vi.fn().mockResolvedValue([
      { guid: 'texture-guid', path: 'Content/Ready.png', state: 'ready' as const },
    ]);

    await ensureExternalAssetImported(
      'content\\ready.png',
      'texture',
      { queryAssets, reimportAsset },
      { sleep: async () => undefined },
    );

    expect(reimportAsset).not.toHaveBeenCalled();
  });

  it('surfaces native import failures', async () => {
    const reimportAsset = vi.fn();
    const queryAssets = vi.fn().mockResolvedValue([
      {
        guid: 'texture-guid',
        path: 'Content/Broken.png',
        state: 'failed' as const,
        diagnostic: 'Texture decode failed',
      },
    ]);

    await expect(
      ensureExternalAssetImported(
        'Content/Broken.png',
        'texture',
        { queryAssets, reimportAsset },
        { sleep: async () => undefined },
      ),
    ).rejects.toThrow('Texture decode failed');

    expect(reimportAsset).not.toHaveBeenCalled();
  });
});

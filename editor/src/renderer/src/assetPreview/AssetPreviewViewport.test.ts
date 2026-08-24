import { describe, expect, it } from 'vitest';

import { assetPreviewViewportId, serializeAssetPreviewViewportLifecycle } from './AssetPreviewViewport';

describe('assetPreviewViewportId', () => {
  it('uses a reserved material preview surface namespace', () => {
    expect(assetPreviewViewportId('material', 'AABBCCDD-0011-2233-4455-66778899AABB')).toBe(
      'asset-preview-material-aabbccdd-0011-2233-4455-66778899aabb',
    );
  });

  it('keeps shader preview surfaces distinct from material previews', () => {
    const guid = '11111111-2222-3333-4444-555555555555';
    expect(assetPreviewViewportId('shader', guid)).toBe(`asset-preview-shader-${guid}`);
    expect(assetPreviewViewportId('shader', guid)).not.toBe(assetPreviewViewportId('material', guid));
  });
});

describe('serializeAssetPreviewViewportLifecycle', () => {
  it('finishes the previous detach before recreating the same preview viewport', async () => {
    const viewportId = 'asset-preview-material-lifecycle-test';
    const calls: string[] = [];
    let markDetachStarted: (() => void) | undefined;
    const detachStarted = new Promise<void>((resolve) => {
      markDetachStarted = resolve;
    });
    let releaseDetach: (() => void) | undefined;
    const detachBlocked = new Promise<void>((resolve) => {
      releaseDetach = resolve;
    });

    const detach = serializeAssetPreviewViewportLifecycle(viewportId, async () => {
      calls.push('detach:start');
      markDetachStarted?.();
      await detachBlocked;
      calls.push('detach:end');
    });
    const create = serializeAssetPreviewViewportLifecycle(viewportId, async () => {
      calls.push('create');
    });

    await detachStarted;
    expect(calls).toEqual(['detach:start']);

    releaseDetach?.();
    await Promise.all([detach, create]);
    expect(calls).toEqual(['detach:start', 'detach:end', 'create']);
  });
});

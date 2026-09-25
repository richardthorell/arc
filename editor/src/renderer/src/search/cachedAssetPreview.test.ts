// @vitest-environment jsdom

import { afterEach, describe, expect, it, vi } from 'vitest';

import { loadCachedAssetPreview } from './cachedAssetPreview';

const originalArc = window.arc;

afterEach(() => {
  Object.defineProperty(window, 'arc', { configurable: true, value: originalArc });
});

describe('loadCachedAssetPreview', () => {
  it('uses the host cache-only thumbnail lookup', async () => {
    const query = vi.fn().mockResolvedValue({
      succeeded: true,
      payload: { dataUrl: 'data:image/png;base64,cached' },
    });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: { host: { query } },
    });

    await expect(loadCachedAssetPreview('Content/Textures/Cached.png', 7)).resolves.toBe(
      'data:image/png;base64,cached',
    );
    expect(query).toHaveBeenCalledWith('asset.thumbnail', {
      path: 'Content/Textures/Cached.png',
      maxSize: 0,
    });
  });

  it('returns null when no cached thumbnail exists', async () => {
    const query = vi.fn().mockResolvedValue({ succeeded: false });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: { host: { query } },
    });

    await expect(loadCachedAssetPreview('Content/Textures/NotCached.png', 1)).resolves.toBeNull();
    expect(query).toHaveBeenCalledTimes(1);
  });
});

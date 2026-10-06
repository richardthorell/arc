import { describe, expect, it, vi } from 'vitest';

import { ArcResourceRegistry } from './arcResourceRegistry';

describe('ArcResourceRegistry', () => {
  it('registers arbitrary resource kinds without changing registry code', async () => {
    const resolve = vi.fn(async (uri) => ({ uri, label: `Widget ${uri.id}` }));
    const read = vi.fn(async (uri) => ({
      uri,
      mediaType: 'text/plain',
      text: `details:${uri.id}`,
    }));
    const registry = new ArcResourceRegistry();
    registry.register({ kind: 'widget', resolve, read });

    await expect(registry.resolve('arc://widget/widget-42')).resolves.toMatchObject({ label: 'Widget widget-42' });
    await expect(registry.read('arc://widget/widget-42/details')).resolves.toMatchObject({
      mediaType: 'text/plain',
      text: 'details:widget-42',
    });
    expect(resolve).toHaveBeenCalledTimes(1);
    expect(read).toHaveBeenCalledTimes(1);
  });

  it('rejects duplicate handlers and leaves unknown or malformed resources unresolved', async () => {
    const registry = new ArcResourceRegistry();
    const unregister = registry.register({ kind: 'asset', resolve: () => null });

    expect(() => registry.register({ kind: 'asset', resolve: () => null })).toThrow(/already registered/);
    await expect(registry.resolve('arc://unknown/value')).resolves.toBeNull();
    await expect(registry.read('not-a-uri')).resolves.toBeNull();

    unregister();
    expect(registry.has('asset')).toBe(false);
  });
});

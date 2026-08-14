import { describe, expect, it, vi } from 'vitest';

import { PolyHavenAssetSource } from './polyHavenAssetSource';

const catalog = {
  sunset: {
    name: 'City Sunset',
    description: 'Warm rooftop sunset',
    category: 'Outdoor/City',
    tags: ['city', 'sunset'],
    thumbnail_url: 'https://cdn.example/sunset.png',
    date_published: 1_700_000_000,
    type: 0,
  },
  concrete: {
    name: 'Concrete Floor',
    description: 'Weathered concrete',
    category: 'Concrete/Floors',
    tags: ['concrete', 'rough'],
    type: 1,
  },
  football: {
    name: 'Dirty Football',
    description: 'Weathered soccer ball',
    category: 'Leisure/Sports',
    tags: ['soccer', 'ball'],
    polycount: 37_486,
    type: 2,
  },
};

describe('PolyHavenAssetSource', () => {
  it('normalizes, filters and caches the Poly Haven catalog', async () => {
    const fetchJson = vi.fn().mockResolvedValue(catalog);
    const source = new PolyHavenAssetSource({ fetchJson, userAgent: 'ARC-Editor/test' });

    const result = await source.search({ text: 'weathered', kinds: ['model'] });
    expect(result.source.id).toBe('polyhaven');
    expect(result.source.attribution).toBe('Powered by Poly Haven');
    expect(result.total).toBe(1);
    expect(result.assets[0]).toMatchObject({
      id: 'football',
      name: 'Dirty Football',
      kind: 'model',
      license: 'CC0',
    });

    await source.search({ text: 'city' });
    expect(fetchJson).toHaveBeenCalledTimes(1);
    expect(fetchJson).toHaveBeenCalledWith('https://api.polyhaven.com/assets', {
      Accept: 'application/json',
      'User-Agent': 'ARC-Editor/test',
    });
  });

  it('returns a normalized asset by provider id', async () => {
    const source = new PolyHavenAssetSource({ fetchJson: vi.fn().mockResolvedValue(catalog) });
    await expect(source.getAsset('concrete')).resolves.toMatchObject({
      id: 'concrete',
      kind: 'texture',
      category: 'Concrete/Floors',
    });
    await expect(source.getAsset('missing')).resolves.toBeNull();
  });

  it('flattens nested Poly Haven file responses into a download manifest', async () => {
    const fetchJson = vi.fn().mockResolvedValue({
      blend: {
        '4k': {
          blend: {
            url: 'https://dl.example/asset.blend',
            size: 1200,
            md5: 'deadbeef',
          },
        },
      },
      textures: {
        diffuse: {
          url: 'https://dl.example/diffuse.jpg',
          size: 600,
          sha256: 'abc123',
        },
      },
    });
    const source = new PolyHavenAssetSource({ fetchJson, userAgent: 'ARC-Editor/test' });

    const manifest = await source.getDownloadManifest('dirty football');
    expect(fetchJson).toHaveBeenCalledWith('https://api.polyhaven.com/files/dirty%20football', {
      Accept: 'application/json',
      'User-Agent': 'ARC-Editor/test',
    });
    expect(manifest).toEqual({
      sourceId: 'polyhaven',
      assetId: 'dirty football',
      files: [
        {
          logicalPath: 'blend/4k/blend',
          url: 'https://dl.example/asset.blend',
          sizeBytes: 1200,
          checksum: { algorithm: 'md5', value: 'deadbeef' },
        },
        {
          logicalPath: 'textures/diffuse',
          url: 'https://dl.example/diffuse.jpg',
          sizeBytes: 600,
          checksum: { algorithm: 'sha256', value: 'abc123' },
        },
      ],
    });
  });
});

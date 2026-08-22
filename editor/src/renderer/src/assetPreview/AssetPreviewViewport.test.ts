import { describe, expect, it } from 'vitest';

import { assetPreviewViewportId } from './AssetPreviewViewport';

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

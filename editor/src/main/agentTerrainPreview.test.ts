import { describe, expect, it } from 'vitest';

import { previewTerrainModifier, previewTerrainStroke } from './agentTerrainPreview';

const revision = 'a'.repeat(64);

const stroke = {
  terrainAsset: 'Content/World/Main.arcterrain',
  expectedRevision: revision,
  layerId: 'sculpt.base',
  kind: 'sculpt' as const,
  region: { minX: 4, minY: 8, maxX: 12, maxY: 16 },
  strength: 0.25,
  radius: 2,
};

describe('terrain agent previews', () => {
  it('reports the exact affected region without changing the request', () => {
    const request = { ...stroke, region: { ...stroke.region } };
    const before = structuredClone(request);

    expect(previewTerrainStroke(request, 128)).toMatchObject({
      kind: 'stroke',
      affectedRegion: { minX: 4, minY: 8, maxX: 12, maxY: 16 },
      affectedArea: 64,
      risk: 'bounded',
      requiresApproval: false,
    });
    expect(request).toEqual(before);
  });

  it('marks broad edits for approval using caller-owned policy', () => {
    expect(previewTerrainStroke(stroke, 32)).toMatchObject({
      affectedArea: 64,
      risk: 'broad',
      requiresApproval: true,
    });
  });

  it('rejects invalid strokes before producing an approval payload', () => {
    expect(() => previewTerrainStroke({ ...stroke, expectedRevision: 'stale' }, 128)).toThrow(
      'expectedRevision must be a SHA-256 revision',
    );
  });

  it('rejects invalid broad-edit policy thresholds', () => {
    expect(() => previewTerrainStroke(stroke, 0)).toThrow('broadEditArea');
  });

  it('previews modifier ordering without inventing affected terrain regions', () => {
    expect(
      previewTerrainModifier({
        terrainAsset: 'Content/World/Main.arcterrain',
        expectedRevision: revision,
        modifierId: 'road.main',
        insertAfterId: 'noise.base',
      }),
    ).toEqual({
      kind: 'modifier-order',
      terrainAsset: 'Content/World/Main.arcterrain',
      expectedRevision: revision,
      modifierId: 'road.main',
      insertAfterId: 'noise.base',
      risk: 'bounded',
      requiresApproval: false,
      summary: 'move modifier road.main after noise.base',
    });
  });
});

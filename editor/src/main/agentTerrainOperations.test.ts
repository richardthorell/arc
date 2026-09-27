import { describe, expect, it } from 'vitest';
import { validateTerrainModifier, validateTerrainRegion, validateTerrainStroke } from './agentTerrainOperations';

const revision = 'a'.repeat(64);

describe('agent terrain operation contracts', () => {
  it('accepts a bounded sculpt stroke owned by a terrain revision', () => {
    expect(
      validateTerrainStroke({
        terrainAsset: 'Content/World/Main.arcterrain',
        expectedRevision: revision,
        layerId: 'sculpt.base',
        kind: 'sculpt',
        region: { minX: 0, minY: 0, maxX: 16, maxY: 16 },
        strength: 0.25,
        radius: 4,
      }),
    ).toEqual({ valid: true, diagnostics: [] });
  });

  it('requires paint strokes to name an attribute channel', () => {
    const result = validateTerrainStroke({
      terrainAsset: 'Content/World/Main.arcterrain',
      expectedRevision: revision,
      layerId: 'paint.biome',
      kind: 'paint',
      region: { minX: 0, minY: 0, maxX: 1, maxY: 1 },
      strength: 0.5,
      radius: 1,
    });
    expect(result.valid).toBe(false);
    expect(result.diagnostics).toContain('paint strokes require a stable channel identifier');
  });

  it('rejects unbounded, malformed, or project-escaping requests', () => {
    const result = validateTerrainStroke({
      terrainAsset: '../Outside.arcterrain',
      expectedRevision: 'stale',
      layerId: '',
      kind: 'sculpt',
      region: { minX: 5, minY: 0, maxX: 2, maxY: 1 },
      strength: 2,
      radius: 0,
    });
    expect(result.valid).toBe(false);
    expect(result.diagnostics.length).toBeGreaterThanOrEqual(5);
  });

  it('rejects non-finite and empty terrain regions', () => {
    expect(validateTerrainRegion({ minX: 0, minY: 0, maxX: Number.POSITIVE_INFINITY, maxY: 1 }).valid).toBe(false);
    expect(validateTerrainRegion({ minX: 2, minY: 2, maxX: 2, maxY: 3 }).valid).toBe(false);
  });

  it('validates revision-safe modifier ordering requests', () => {
    expect(
      validateTerrainModifier({
        terrainAsset: 'Content/World/Main.arcterrain',
        expectedRevision: revision,
        modifierId: 'road.main',
        insertAfterId: 'noise.base',
      }).valid,
    ).toBe(true);

    const selfOrder = validateTerrainModifier({
      terrainAsset: 'Content/World/Main.arcterrain',
      expectedRevision: revision,
      modifierId: 'road.main',
      insertAfterId: 'road.main',
    });
    expect(selfOrder.valid).toBe(false);
    expect(selfOrder.diagnostics).toContain('a modifier cannot be ordered after itself');
  });
});

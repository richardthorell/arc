import { describe, expect, it } from 'vitest';
import { createAssetImportRecipe, createImportedAssetProvenance } from './assetProvenance';
import { parseAssetProvenanceSidecar, serializeAssetProvenanceSidecar } from './assetProvenancePersistence';

describe('asset provenance persistence', () => {
  it('round-trips complete provenance without introducing ARC asset identity', () => {
    const provenance = createImportedAssetProvenance(
      { sourceId: 'polyhaven', id: 'rock_01', license: 'CC0' },
      {
        importedAt: '2026-10-02T18:00:00.000Z',
        sourceUrl: 'https://example.test/rock_01',
        sourceRevision: 'revision-7',
        sourceHash: 'sha256:abc123',
        recipe: createAssetImportRecipe(
          { logicalPaths: ['4k/rock.glb', '4k/albedo.jpg'] },
          { generateTangents: true, scale: 1 },
        ),
      },
    );

    const serialized = serializeAssetProvenanceSidecar(provenance);

    expect(serialized).not.toContain('assetId');
    expect(parseAssetProvenanceSidecar(serialized)).toEqual(provenance);
  });

  it('keeps legacy provenance without a recipe readable', () => {
    const contents = JSON.stringify({
      version: 1,
      provenance: {
        sourceId: 'legacy-provider',
        sourceAssetId: 'legacy_asset',
        importedAt: '2026-01-01T00:00:00.000Z',
        license: 'legacy-license',
      },
    });

    expect(parseAssetProvenanceSidecar(contents)).toEqual({
      sourceId: 'legacy-provider',
      sourceAssetId: 'legacy_asset',
      importedAt: '2026-01-01T00:00:00.000Z',
      license: 'legacy-license',
      recipe: undefined,
    });
  });

  it('rejects malformed, unsupported, and incomplete sidecars', () => {
    expect(parseAssetProvenanceSidecar('{broken')).toBeNull();
    expect(parseAssetProvenanceSidecar(JSON.stringify({ version: 2, provenance: {} }))).toBeNull();
    expect(
      parseAssetProvenanceSidecar(
        JSON.stringify({
          version: 1,
          provenance: {
            sourceId: 'polyhaven',
            sourceAssetId: 'rock_01',
            importedAt: '2026-10-02T18:00:00.000Z',
            license: 'CC0',
            recipe: { version: 1, logicalPaths: [42] },
          },
        }),
      ),
    ).toBeNull();
  });
});

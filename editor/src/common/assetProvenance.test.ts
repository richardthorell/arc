import { describe, expect, it } from 'vitest';
import {
  createAssetImportRecipe,
  createAssetProvenanceMetadata,
  createImportedAssetProvenance,
  createReimportRequest,
} from './assetProvenance';

describe('asset provenance', () => {
  it('normalizes recipes deterministically without changing selected files', () => {
    const recipe = createAssetImportRecipe(
      { logicalPaths: ['textures\\normal.png', './mesh.glb', 'mesh.glb', ' textures/albedo.png '] },
      { scale: 1, generateTangents: true },
    );

    expect(recipe).toEqual({
      version: 1,
      logicalPaths: ['mesh.glb', 'textures/albedo.png', 'textures/normal.png'],
      options: { generateTangents: true, scale: 1 },
    });
  });

  it('records source identity independently from the ARC asset identity', () => {
    const recipe = createAssetImportRecipe({ logicalPaths: ['mesh.glb'] });
    const provenance = createImportedAssetProvenance(
      { sourceId: 'polyhaven', id: 'rock_01', license: 'CC0' },
      {
        importedAt: '2026-09-28T20:00:00.000Z',
        sourceUrl: 'https://example.test/rock_01',
        sourceRevision: 'revision-7',
        sourceHash: 'sha256:abc123',
        recipe,
      },
    );

    expect(provenance).toEqual({
      sourceId: 'polyhaven',
      sourceAssetId: 'rock_01',
      importedAt: '2026-09-28T20:00:00.000Z',
      license: 'CC0',
      sourceUrl: 'https://example.test/rock_01',
      sourceRevision: 'revision-7',
      sourceHash: 'sha256:abc123',
      recipe: { version: 1, logicalPaths: ['mesh.glb'], options: undefined },
    });
  });

  it('reconstructs reimport selection from the recorded recipe rather than current UI defaults', () => {
    const provenance = createImportedAssetProvenance(
      { sourceId: 'polyhaven', id: 'rock_01', license: 'CC0' },
      {
        importedAt: '2026-09-28T20:00:00.000Z',
        recipe: createAssetImportRecipe({ logicalPaths: ['4k/rock.glb', '4k/albedo.jpg'] }),
      },
    );

    expect(createReimportRequest(provenance)).toEqual({
      sourceId: 'polyhaven',
      assetId: 'rock_01',
      logicalPaths: ['4k/albedo.jpg', '4k/rock.glb'],
      destinationScope: 'project',
    });
  });

  it('keeps legacy provenance readable when no recipe was recorded', () => {
    expect(
      createReimportRequest({
        sourceId: 'polyhaven',
        sourceAssetId: 'legacy_asset',
        importedAt: '2026-01-01T00:00:00.000Z',
        license: 'CC0',
      }),
    ).toBeNull();
  });

  it('projects complete provenance into stable metadata for editor presentation', () => {
    const provenance = createImportedAssetProvenance(
      { sourceId: 'polyhaven', id: 'rock_01', license: 'CC0' },
      {
        importedAt: '2026-09-28T20:00:00.000Z',
        sourceUrl: 'https://example.test/rock_01',
        sourceRevision: 'revision-7',
        sourceHash: 'sha256:abc123',
        recipe: createAssetImportRecipe(
          { logicalPaths: ['4k/rock.glb', '4k/albedo.jpg'] },
          { scale: 1, generateTangents: true },
        ),
      },
    );

    expect(createAssetProvenanceMetadata(provenance)).toEqual({
      provider: 'polyhaven',
      providerAssetId: 'rock_01',
      importedAt: '2026-09-28T20:00:00.000Z',
      licenseAtImport: 'CC0',
      sourceUrl: 'https://example.test/rock_01',
      sourceRevision: 'revision-7',
      sourceHash: 'sha256:abc123',
      recipeVersion: 1,
      selectedFiles: ['4k/albedo.jpg', '4k/rock.glb'],
      importOptions: [
        { key: 'generateTangents', value: true },
        { key: 'scale', value: 1 },
      ],
    });
  });

  it('projects legacy provenance without inventing recipe metadata', () => {
    expect(
      createAssetProvenanceMetadata({
        sourceId: 'legacy-provider',
        sourceAssetId: 'legacy_asset',
        importedAt: '2026-01-01T00:00:00.000Z',
        license: 'legacy-license',
      }),
    ).toEqual({
      provider: 'legacy-provider',
      providerAssetId: 'legacy_asset',
      importedAt: '2026-01-01T00:00:00.000Z',
      licenseAtImport: 'legacy-license',
      sourceUrl: undefined,
      sourceRevision: undefined,
      sourceHash: undefined,
      recipeVersion: undefined,
      selectedFiles: [],
      importOptions: [],
    });
  });
});

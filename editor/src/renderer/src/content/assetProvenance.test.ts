import { describe, expect, it } from 'vitest';

import { createAssetProvenanceMetadata, createAssetReimportRequest } from './assetProvenance';

describe('asset provenance', () => {
  const source = {
    provider: 'poly-haven',
    providerAssetId: 'wood-floor-01',
    sourceUrl: 'https://example.invalid/wood-floor-01',
    sourceRevision: 'revision-7',
    sourceHash: 'sha256:abc123',
    licenseAtImport: 'CC0',
  };

  it('captures source identity and the selected import recipe', () => {
    const metadata = createAssetProvenanceMetadata(source, {
      importer: 'texture',
      variant: '4k-jpg',
      options: { generateMips: true, maxSize: 4096, colorSpace: 'srgb' },
    });

    expect(metadata).toEqual({
      version: 1,
      source,
      recipe: {
        importer: 'texture',
        variant: '4k-jpg',
        options: { colorSpace: 'srgb', generateMips: true, maxSize: 4096 },
      },
    });
  });

  it('builds re-import from recorded metadata while preserving ARC asset identity', () => {
    const metadata = createAssetProvenanceMetadata(source, {
      importer: 'texture',
      variant: '2k-exr',
      options: { maxSize: 2048 },
    });

    const request = createAssetReimportRequest('asset-guid-123', metadata);

    expect(request.assetId).toBe('asset-guid-123');
    expect(request.source).toEqual(source);
    expect(request.recipe).toEqual(metadata.recipe);
  });

  it('does not allow current UI defaults to replace the recorded recipe', () => {
    const metadata = createAssetProvenanceMetadata(source, {
      importer: 'texture',
      variant: '1k-jpg',
      options: { maxSize: 1024 },
    });
    const currentUiDefaults = { variant: '8k-exr', options: { maxSize: 8192 } };

    const request = createAssetReimportRequest('asset-guid-123', metadata);

    expect(request.recipe.variant).toBe('1k-jpg');
    expect(request.recipe.options.maxSize).toBe(1024);
    expect(request.recipe).not.toMatchObject(currentUiDefaults);
  });

  it('rejects incomplete provenance instead of creating non-reproducible metadata', () => {
    expect(() =>
      createAssetProvenanceMetadata(
        { ...source, providerAssetId: '   ' },
        { importer: 'texture', variant: '4k-jpg', options: {} },
      ),
    ).toThrow('provider asset ID');
  });
});

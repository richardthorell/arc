import { describe, expect, it } from 'vitest';

import type { ArcAssetDownloadManifest, ArcRemoteAsset } from '../../common/assetSourceTypes';
import { buildImportProvenance } from './importProvenance';

const asset: ArcRemoteAsset = {
  id: 'studio_small_09',
  sourceId: 'polyhaven',
  name: 'Studio Small 09',
  description: '',
  kind: 'hdri',
  category: 'studio',
  tags: [],
  license: 'CC0',
  metadata: {
    filesHash: 'revision-42',
    sourceUrl: 'https://polyhaven.com/a/studio_small_09',
  },
};

const manifest: ArcAssetDownloadManifest = {
  sourceId: 'polyhaven',
  assetId: 'studio_small_09',
  files: [
    { logicalPath: 'hdri/2k.hdr', url: 'https://example.test/2k.hdr' },
    { logicalPath: 'hdri/4k.hdr', url: 'https://example.test/4k.hdr' },
  ],
};

describe('import provenance', () => {
  it('records stable source identity, revision, license, and a deterministic import recipe', () => {
    expect(
      buildImportProvenance({
        asset,
        manifest,
        logicalPaths: ['hdri/4k.hdr', 'hdri/2k.hdr', 'hdri/4k.hdr'],
        importedAt: '2026-10-05T09:00:00.000Z',
        options: { colorSpace: 'linear', generateMips: true },
      }),
    ).toEqual({
      sourceId: 'polyhaven',
      sourceAssetId: 'studio_small_09',
      importedAt: '2026-10-05T09:00:00.000Z',
      license: 'CC0',
      sourceUrl: 'https://polyhaven.com/a/studio_small_09',
      sourceRevision: 'revision-42',
      recipe: {
        version: 1,
        logicalPaths: ['hdri/2k.hdr', 'hdri/4k.hdr'],
        options: { colorSpace: 'linear', generateMips: true },
      },
    });
  });

  it('rejects mismatched provider identities', () => {
    expect(() =>
      buildImportProvenance({
        asset,
        manifest: { ...manifest, assetId: 'other' },
        logicalPaths: ['hdri/2k.hdr'],
        importedAt: '2026-10-05T09:00:00.000Z',
      }),
    ).toThrow('asset and download manifest identities must match');
  });

  it('rejects source files that are not present in the resolved manifest', () => {
    expect(() =>
      buildImportProvenance({
        asset,
        manifest,
        logicalPaths: ['hdri/8k.hdr'],
        importedAt: '2026-10-05T09:00:00.000Z',
      }),
    ).toThrow("unknown source file 'hdri/8k.hdr'");
  });
});

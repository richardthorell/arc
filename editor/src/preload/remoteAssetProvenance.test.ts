import { describe, expect, it } from 'vitest';

import type { ArcAssetDownloadFile, ArcRemoteAsset } from '../common/assetSourceTypes';
import {
  createRemoteImportProvenance,
  createRemoteImportProvenanceSidecar,
  createRemoteManifestHash,
  serializeRemoteImportProvenanceSidecar,
} from './remoteAssetProvenance';

const asset: ArcRemoteAsset = {
  id: 'rock_01',
  sourceId: 'polyhaven',
  name: 'Rock',
  description: '',
  kind: 'model',
  category: 'rocks',
  tags: [],
  license: 'CC0',
  metadata: { filesHash: 'provider-revision-7' },
};

const files: ArcAssetDownloadFile[] = [
  {
    logicalPath: '4k/rock.glb',
    url: 'https://example.test/rock.glb',
    sizeBytes: 1024,
    checksum: { algorithm: 'sha256', value: 'ABCDEF' },
  },
  {
    logicalPath: '4k/albedo.jpg',
    url: 'https://example.test/albedo.jpg',
    sizeBytes: 512,
    checksum: { algorithm: 'md5', value: '1234ABCD' },
  },
];

describe('remote asset provenance', () => {
  it('captures provider metadata and the exact deterministic import recipe', () => {
    expect(
      createRemoteImportProvenance(asset, { logicalPaths: ['4k/rock.glb', '4k/albedo.jpg'] }, files, {
        importedAt: '2026-10-03T10:00:00.000Z',
        sourceHomepage: 'https://polyhaven.com/',
        importOptions: { scale: 1, generateTangents: true },
      }),
    ).toEqual({
      sourceId: 'polyhaven',
      sourceAssetId: 'rock_01',
      importedAt: '2026-10-03T10:00:00.000Z',
      license: 'CC0',
      sourceUrl: 'https://polyhaven.com/a/rock_01',
      sourceRevision: 'provider-revision-7',
      sourceHash: createRemoteManifestHash(files),
      recipe: {
        version: 1,
        logicalPaths: ['4k/albedo.jpg', '4k/rock.glb'],
        options: { generateTangents: true, scale: 1 },
      },
    });
  });

  it('produces the same source hash regardless of manifest ordering or checksum case', () => {
    const reordered = [{ ...files[1] }, { ...files[0], checksum: { algorithm: 'sha256' as const, value: 'abcdef' } }];
    expect(createRemoteManifestHash(reordered)).toBe(createRemoteManifestHash(files));
  });

  it('changes the source hash when selected source content changes', () => {
    const changed = files.map((file, index) =>
      index === 0 ? { ...file, checksum: { algorithm: 'sha256' as const, value: 'different' } } : file,
    );
    expect(createRemoteManifestHash(changed)).not.toBe(createRemoteManifestHash(files));
  });

  it('builds a versioned sidecar without duplicating recipe state', () => {
    const provenance = createRemoteImportProvenance(asset, { logicalPaths: ['4k/rock.glb'] }, [files[0]], {
      importedAt: '2026-10-03T10:00:00.000Z',
      sourceHomepage: 'https://polyhaven.com',
    });

    const sidecar = createRemoteImportProvenanceSidecar(
      provenance,
      ['Content/External/polyhaven/rock_01/rock.glb'],
      ['asset-guid-1'],
    );

    expect(sidecar).toEqual({
      version: 1,
      provenance,
      importedFiles: ['Content/External/polyhaven/rock_01/rock.glb'],
      importedAssetIds: ['asset-guid-1'],
    });
    expect(sidecar).not.toHaveProperty('logicalPaths');
    expect(sidecar.provenance.recipe?.logicalPaths).toEqual(['4k/rock.glb']);
  });

  it('serializes the sidecar as stable human-readable JSON ready for persistence', () => {
    const provenance = createRemoteImportProvenance(asset, { logicalPaths: ['4k/rock.glb'] }, [files[0]], {
      importedAt: '2026-10-03T10:00:00.000Z',
      sourceHomepage: 'https://polyhaven.com',
    });
    const sidecar = createRemoteImportProvenanceSidecar(
      provenance,
      ['Content/External/polyhaven/rock_01/rock.glb'],
      ['asset-guid-1'],
    );

    const serialized = serializeRemoteImportProvenanceSidecar(sidecar);

    expect(serialized.endsWith('\n')).toBe(true);
    expect(JSON.parse(serialized)).toEqual(sidecar);
    expect(serialized).toContain('"version": 1');
    expect(serialized).toContain('"sourceHash": "sha256:');
    expect(serialized).toContain('"recipe": {');
  });
});

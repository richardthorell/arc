import path from 'node:path';
import { describe, expect, it } from 'vitest';

import type { ArcRemoteAsset } from '../common/assetSourceTypes';
import {
  buildRemoteImportProvenanceRecord,
  createAssetSourceBridge,
  remoteDestinationPath,
  remoteFileName,
} from './assetSourceBridge';

const file = {
  logicalPath: '../../gltf/2k/gltf/include/../0',
  url: 'https://dl.example/assets/rock%20diff.png',
};

describe('asset source bridge paths', () => {
  it('uses safe file names and keeps model image dependencies under textures', () => {
    expect(remoteFileName(file)).toBe('rock_diff.png');
    const relative = remoteDestinationPath('../rock', file, 'model').replaceAll(path.sep, '/');
    expect(relative).toBe('_rock/textures/rock_diff.png');
    expect(relative).not.toContain('../');
  });
});

describe('asset source bridge import operations', () => {
  it('allocates stable operation identities and only cancels active imports', () => {
    const bridge = createAssetSourceBridge(async () => {
      throw new Error('unexpected invoke');
    });

    expect(bridge.createImportOperation()).toBe(1);
    expect(bridge.createImportOperation()).toBe(2);
    expect(bridge.cancelImport(1)).toBe(false);
  });
});

describe('asset source bridge provenance publication', () => {
  it('persists the canonical sidecar for the exact resolved import selection', () => {
    const asset: ArcRemoteAsset = {
      id: 'rock_01',
      sourceId: 'polyhaven',
      name: 'Rock',
      description: '',
      kind: 'model',
      category: 'rocks',
      tags: [],
      license: 'CC0',
      metadata: { filesHash: 'revision-7' },
    };
    const selectedFiles = [
      {
        logicalPath: '4k/rock.glb',
        url: 'https://example.test/rock.glb',
        checksum: { algorithm: 'sha256' as const, value: 'ABCDEF' },
      },
      {
        logicalPath: '4k/albedo.jpg',
        url: 'https://example.test/albedo.jpg',
        checksum: { algorithm: 'md5' as const, value: '1234' },
      },
    ];

    const { provenance, serializedSidecar } = buildRemoteImportProvenanceRecord({
      asset,
      selectedFiles,
      importedFiles: ['Content/External/polyhaven/rock_01/rock.glb'],
      importedAssetIds: ['asset-guid-1'],
      importedAt: '2026-10-05T20:00:00.000Z',
      sourceHomepage: 'https://polyhaven.com/',
    });
    const sidecar = JSON.parse(serializedSidecar) as Record<string, unknown>;

    expect(provenance.recipe?.logicalPaths).toEqual(['4k/albedo.jpg', '4k/rock.glb']);
    expect(provenance.sourceHash).toMatch(/^sha256:/);
    expect(provenance.sourceUrl).toBe('https://polyhaven.com/a/rock_01');
    expect(sidecar).toMatchObject({
      version: 1,
      importedFiles: ['Content/External/polyhaven/rock_01/rock.glb'],
      importedAssetIds: ['asset-guid-1'],
      provenance: {
        sourceId: 'polyhaven',
        sourceAssetId: 'rock_01',
        sourceHash: provenance.sourceHash,
        recipe: {
          version: 1,
          logicalPaths: ['4k/albedo.jpg', '4k/rock.glb'],
        },
      },
    });
    expect(sidecar).not.toHaveProperty('logicalPaths');
    expect(serializedSidecar.endsWith('\n')).toBe(true);
  });
});

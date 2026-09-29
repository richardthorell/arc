import { describe, expect, it } from 'vitest';

import type { ArcAssetDownloadManifest } from '../../../common/assetSourceTypes';
import { createRemoteAssetImportPlan } from './remoteAssetImportPlan';

const manifest = (): ArcAssetDownloadManifest => ({
  sourceId: 'poly-haven',
  assetId: 'studio-small',
  files: [
    { logicalPath: 'textures/studio.hdr', url: 'https://example.test/studio.hdr', sizeBytes: 12 },
    { logicalPath: 'textures/studio.exr', url: 'https://example.test/studio.exr', sizeBytes: 20 },
  ],
});

describe('createRemoteAssetImportPlan', () => {
  it('preserves selection order and collapses duplicate paths', () => {
    const plan = createRemoteAssetImportPlan(manifest(), [
      'textures/studio.exr',
      './textures/studio.hdr',
      'textures/studio.exr',
    ]);

    expect(plan.request).toEqual({
      sourceId: 'poly-haven',
      assetId: 'studio-small',
      logicalPaths: ['textures/studio.exr', 'textures/studio.hdr'],
      destinationScope: 'project',
    });
    expect(plan.totalBytes).toBe(32);
  });

  it('rejects files that were not offered by the fetched manifest', () => {
    expect(() => createRemoteAssetImportPlan(manifest(), ['textures/other.exr'])).toThrow(
      /not present in the manifest/,
    );
  });

  it('rejects unsafe provider paths before import', () => {
    const unsafe = manifest();
    unsafe.files[0] = { ...unsafe.files[0], logicalPath: '../escape.hdr' };
    expect(() => createRemoteAssetImportPlan(unsafe, ['../escape.hdr'])).toThrow(/unsafe path/);
  });

  it('rejects duplicate normalized manifest paths', () => {
    const duplicate = manifest();
    duplicate.files.push({ logicalPath: './textures/studio.hdr', url: 'https://example.test/duplicate' });
    expect(() => createRemoteAssetImportPlan(duplicate, ['textures/studio.hdr'])).toThrow(/duplicate path/);
  });

  it('requires a non-empty selection and only reports total bytes when all sizes are known', () => {
    expect(() => createRemoteAssetImportPlan(manifest(), [])).toThrow(/at least one/);
    const unknownSize = manifest();
    delete unknownSize.files[1].sizeBytes;
    expect(createRemoteAssetImportPlan(unknownSize, ['textures/studio.exr']).totalBytes).toBeUndefined();
  });
});

import { describe, expect, it } from 'vitest';

import type { ArcAssetDownloadManifest } from '../../../common/assetSourceTypes';
import {
  manifestFormats,
  manifestResolutions,
  preferredFormat,
  preferredResolution,
  selectManifestFiles,
} from './remoteAssetVariants';

const manifest: ArcAssetDownloadManifest = {
  sourceId: 'polyhaven',
  assetId: 'rock',
  files: [
    { logicalPath: 'gltf/2k/gltf', url: 'https://cdn.example/rock.gltf', sizeBytes: 10 },
    { logicalPath: 'gltf/2k/gltf/include/0', url: 'https://cdn.example/rock_diff.png', sizeBytes: 20 },
    { logicalPath: 'gltf/4k/gltf', url: 'https://cdn.example/rock.gltf?4k', sizeBytes: 30 },
    { logicalPath: 'blend/2k/blend', url: 'https://cdn.example/rock.blend', sizeBytes: 40 },
  ],
};

describe('remote asset variants', () => {
  it('discovers resolution and format choices with sensible defaults', () => {
    expect(manifestResolutions(manifest)).toEqual(['2k', '4k']);
    expect(manifestFormats(manifest)).toEqual(['blend', 'gltf', 'png']);
    expect(preferredResolution(manifestResolutions(manifest))).toBe('2k');
    expect(preferredFormat(manifestFormats(manifest), 'model')).toBe('gltf');
  });

  it('keeps dependency files nested under the selected variant', () => {
    const files = selectManifestFiles(manifest, '2k', 'gltf');
    expect(files.map((file) => file.logicalPath)).toEqual(['gltf/2k/gltf', 'gltf/2k/gltf/include/0']);
  });
});

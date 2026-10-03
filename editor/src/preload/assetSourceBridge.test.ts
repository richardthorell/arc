import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { createAssetSourceBridge, remoteDestinationPath, remoteFileName } from './assetSourceBridge';

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

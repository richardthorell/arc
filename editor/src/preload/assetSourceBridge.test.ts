import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { remoteDestinationPath, remoteFileName } from './assetSourceBridge';

const file = {
  logicalPath: '../../gltf/2k/gltf/include/../0',
  url: 'https://dl.example/assets/rock%20diff.png',
};

describe('asset source bridge paths', () => {
  it('uses the download filename while sanitizing manifest path segments', () => {
    expect(remoteFileName(file)).toBe('rock_diff.png');
    const relative = remoteDestinationPath('../rock', file).replaceAll(path.sep, '/');
    expect(relative).toBe('_rock/gltf/2k/gltf/include/0/rock_diff.png');
    expect(relative).not.toContain('../');
  });
});

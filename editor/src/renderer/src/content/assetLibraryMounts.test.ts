import { describe, expect, it } from 'vitest';

import { assetLibraryMountForScope, buildAssetLibraryMounts } from './assetLibraryMounts';

describe('asset library mounts', () => {
  it('keeps logical scope ordering and writability independent of physical roots', () => {
    const mounts = buildAssetLibraryMounts({
      organization: 'Z:\\Shared\\Arc\\',
      user: '/Users/richard/.arc/assets/',
      project: 'C:\\Projects\\Game\\Content\\',
      builtin: 'C:\\Arc\\Engine\\Assets\\',
    });

    expect(mounts).toEqual([
      { scope: 'builtin', root: 'C:/Arc/Engine/Assets', writable: false },
      { scope: 'project', root: 'C:/Projects/Game/Content', writable: true },
      { scope: 'user', root: '/Users/richard/.arc/assets', writable: true },
      { scope: 'organization', root: 'Z:/Shared/Arc', writable: false },
    ]);
  });

  it('omits unavailable optional mounts instead of aliasing their identity', () => {
    const mounts = buildAssetLibraryMounts({ project: '/project/Content', user: '   ' });

    expect(mounts).toEqual([{ scope: 'project', root: '/project/Content', writable: true }]);
    expect(assetLibraryMountForScope(mounts, 'user')).toBeNull();
    expect(assetLibraryMountForScope(mounts, 'organization')).toBeNull();
  });

  it('resolves the configured mount by logical scope', () => {
    const mounts = buildAssetLibraryMounts({ project: '/project', user: '/user' });

    expect(assetLibraryMountForScope(mounts, 'user')).toEqual({
      scope: 'user',
      root: '/user',
      writable: true,
    });
  });
});

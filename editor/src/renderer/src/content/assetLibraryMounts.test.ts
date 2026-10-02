import { describe, expect, it } from 'vitest';

import { assetLibraryMountForScope, buildAssetLibraryMounts, resolveAssetLibraryMountPath } from './assetLibraryMounts';

describe('asset library mounts', () => {
  it('keeps logical scope ordering and writability independent of physical roots', () => {
    const mounts = buildAssetLibraryMounts({
      organization: 'Z:\\Shared\\Arc\\',
      user: '/Users/example/.arc/assets/',
      project: 'C:\\Projects\\Game\\Content\\',
      builtin: 'C:\\Arc\\Engine\\Assets\\',
    });

    expect(mounts).toEqual([
      { scope: 'builtin', root: 'C:/Arc/Engine/Assets', writable: false },
      { scope: 'project', root: 'C:/Projects/Game/Content', writable: true },
      { scope: 'user', root: '/Users/example/.arc/assets', writable: true },
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

  it('resolves relative asset paths through their logical mount', () => {
    const mounts = buildAssetLibraryMounts({
      project: 'C:\\Projects\\Game\\Content',
      user: '/Users/example/.arc/assets',
    });

    expect(resolveAssetLibraryMountPath(mounts, 'project', 'Materials\\Metal.arcasset')).toBe(
      'C:/Projects/Game/Content/Materials/Metal.arcasset',
    );
    expect(resolveAssetLibraryMountPath(mounts, 'user', './Textures/Noise.arcasset')).toBe(
      '/Users/example/.arc/assets/Textures/Noise.arcasset',
    );
  });

  it('does not alias missing scopes or allow a relative path to escape its mount', () => {
    const mounts = buildAssetLibraryMounts({ project: '/project/Content' });

    expect(resolveAssetLibraryMountPath(mounts, 'user', 'Textures/Noise.arcasset')).toBeNull();
    expect(resolveAssetLibraryMountPath(mounts, 'project', '../Shared/Secret.arcasset')).toBeNull();
    expect(resolveAssetLibraryMountPath(mounts, 'project', '/absolute/asset.arcasset')).toBeNull();
    expect(resolveAssetLibraryMountPath(mounts, 'project', 'C:\\Other\\asset.arcasset')).toBeNull();
  });
});

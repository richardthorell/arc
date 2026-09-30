import { describe, expect, it } from 'vitest';

import { resolveAssetLibraryMutationTarget } from './assetLibraryMutationTarget';

describe('asset library mutation target', () => {
  it('preserves writable logical scopes without exposing physical storage paths', () => {
    expect(resolveAssetLibraryMutationTarget('project', 'Characters/Heroes')).toEqual({
      scope: 'project',
      relativeFolder: 'Characters/Heroes',
    });
    expect(resolveAssetLibraryMutationTarget('user', 'Brushes\\Natural')).toEqual({
      scope: 'user',
      relativeFolder: 'Brushes/Natural',
    });
  });

  it('routes read-only and unknown scopes through the shared project fallback', () => {
    expect(resolveAssetLibraryMutationTarget('builtin', 'Materials')).toEqual({
      scope: 'project',
      relativeFolder: 'Materials',
    });
    expect(resolveAssetLibraryMutationTarget('organization', 'Studio/Shared')).toEqual({
      scope: 'project',
      relativeFolder: 'Studio/Shared',
    });
    expect(resolveAssetLibraryMutationTarget('legacy-mount', 'Legacy')).toEqual({
      scope: 'project',
      relativeFolder: 'Legacy',
    });
  });

  it('normalizes logical folders deterministically', () => {
    expect(resolveAssetLibraryMutationTarget('user', ' /Brushes//Natural/ ')).toEqual({
      scope: 'user',
      relativeFolder: 'Brushes/Natural',
    });
    expect(resolveAssetLibraryMutationTarget('project')).toEqual({ scope: 'project', relativeFolder: '' });
  });

  it('rejects traversal instead of allowing a logical scope to escape its mount', () => {
    expect(() => resolveAssetLibraryMutationTarget('user', '../Project')).toThrow(/logical relative path/i);
    expect(() => resolveAssetLibraryMutationTarget('project', 'Characters/./Hero')).toThrow(/logical relative path/i);
  });
});

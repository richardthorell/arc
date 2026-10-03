import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { projectAssetLogicalMounts, resolveProjectAssetMountRoots } from './projectAssetMountRoots';

describe('resolveProjectAssetMountRoots', () => {
  it('resolves explicitly configured logical mount roots without inventing optional scopes', () => {
    const root = path.resolve('workspace/project');
    const mounts = resolveProjectAssetMountRoots({
      projectRoot: root,
      projectAssetRoots: ['GameContent'],
      builtinAssetsRoot: path.resolve('engine/assets'),
      userAssetsRoot: path.resolve('user/assets'),
      organizationAssetsRoot: path.resolve('organization/assets'),
    });

    expect(mounts).toEqual({
      builtinRoot: path.resolve('engine/assets'),
      projectRoot: path.join(root, 'GameContent'),
      userRoot: path.resolve('user/assets'),
      organizationRoot: path.resolve('organization/assets'),
    });
  });

  it('keeps optional mounts unavailable when the host has not configured them', () => {
    const root = path.resolve('workspace/project');
    expect(
      resolveProjectAssetMountRoots({
        projectRoot: root,
        projectAssetRoots: [],
      }),
    ).toEqual({
      builtinRoot: '',
      projectRoot: path.join(root, 'Content'),
      userRoot: '',
      organizationRoot: '',
    });
  });

  it('rejects a project asset root that escapes the active project', () => {
    expect(() =>
      resolveProjectAssetMountRoots({
        projectRoot: path.resolve('workspace/project'),
        projectAssetRoots: ['../shared'],
      }),
    ).toThrow('must remain inside the active project');
  });
});

describe('projectAssetLogicalMounts', () => {
  it('projects resolved host roots onto the production logical-mount contract', () => {
    expect(
      projectAssetLogicalMounts({
        builtinRoot: path.resolve('engine/assets'),
        projectRoot: path.resolve('workspace/project/Content'),
        userRoot: path.resolve('user/assets'),
        organizationRoot: path.resolve('organization/assets'),
      }),
    ).toEqual({
      builtin: path.resolve('engine/assets'),
      project: path.resolve('workspace/project/Content'),
      user: path.resolve('user/assets'),
      organization: path.resolve('organization/assets'),
    });
  });

  it('omits unavailable optional roots while keeping Project authoritative', () => {
    const projectRoot = path.resolve('workspace/project/Content');
    expect(
      projectAssetLogicalMounts({
        builtinRoot: '',
        projectRoot,
        userRoot: '',
        organizationRoot: '',
      }),
    ).toEqual({ project: projectRoot });
  });
});

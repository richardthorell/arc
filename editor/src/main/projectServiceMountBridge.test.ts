import path from 'node:path';

import { describe, expect, it, vi } from 'vitest';

import type { ArcProjectCandidate } from '../common/projectTypes';
import { ProjectService } from './projectService';

const project = (root: string, assetRoots: string[] = ['Content']): ArcProjectCandidate =>
  ({
    projectRoot: root,
    descriptor: { assetRoots },
  }) as ArcProjectCandidate;

const service = (overrides: { builtinAssetsRoot?: string; userAssetsRoot?: string; organizationAssetsRoot?: string } = {}) =>
  new ProjectService({
    userDataPath: path.resolve('tmp/project-service-mount-bridge'),
    currentEngineVersion: '1.0.0',
    currentEditorPath: 'arc-editor',
    projectToolPath: '',
    templatesRoot: path.resolve('missing/templates'),
    host: { connected: true, error: '', command: async () => ({ succeeded: true }) },
    ...overrides,
  });

describe('ProjectService logical asset mount bridge', () => {
  it('resolves all configured logical mounts through the shared host resolver', () => {
    const root = path.resolve('workspace/project');
    const instance = service({
      builtinAssetsRoot: path.resolve('engine/assets'),
      userAssetsRoot: path.resolve('user/assets'),
      organizationAssetsRoot: path.resolve('organization/assets'),
    });

    expect(instance.assetMounts(project(root, ['GameContent']))).toEqual({
      builtin: path.resolve('engine/assets'),
      project: path.join(root, 'GameContent'),
      user: path.resolve('user/assets'),
      organization: path.resolve('organization/assets'),
    });
  });

  it('publishes the active project mounts on the production project snapshot bridge', () => {
    const root = path.resolve('workspace/project');
    const instance = service({ builtinAssetsRoot: path.resolve('engine/assets') });
    vi.spyOn(instance, 'active').mockReturnValue(project(root));

    expect(instance.snapshot().mounts).toEqual({
      builtin: path.resolve('engine/assets'),
      project: path.join(root, 'Content'),
    });
  });

  it('does not invent optional mounts when the host has not configured them', () => {
    const root = path.resolve('workspace/project');
    const instance = service();

    expect(instance.assetMounts(project(root))).toEqual({ project: path.join(root, 'Content') });
  });
});

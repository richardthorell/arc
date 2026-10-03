import { describe, expect, it } from 'vitest';

import type { AssetItem, ProjectSnapshot } from '../services/editorHostTypes';
import { assetLibraryScopeViewsForProject } from './assetLibraryScopeSnapshot';

const asset = (id: string, scope: AssetItem['scope']): AssetItem =>
  ({ id, guid: id, name: id, path: `${scope}/${id}.arcasset`, kind: 'material', status: 'ready', scope }) as AssetItem;

const project = (overrides: Partial<ProjectSnapshot> = {}): ProjectSnapshot => ({
  name: 'Scope Test',
  root: '/project',
  assetRoot: '/project/Content',
  activeScene: '',
  scene: [],
  assets: [],
  console: [],
  renderStats: {
    fps: 0,
    frameTimeMs: 0,
    drawCalls: 0,
    triangles: 0,
    visibleEntities: 0,
    lights: 0,
    gpuMemoryMb: 0,
  },
  ...overrides,
});

describe('asset library project scope adapter', () => {
  it('keeps Project available through the existing assetRoot compatibility path', () => {
    const views = assetLibraryScopeViewsForProject(project({ assets: [asset('project-id', 'project')] }));

    expect(views.find((view) => view.scope === 'project')).toMatchObject({
      available: true,
      writable: true,
      assetIds: ['project-id'],
    });
  });

  it('only exposes optional scopes when the host explicitly configures their mounts', () => {
    const assets = [asset('builtin-id', 'builtin'), asset('user-id', 'user'), asset('org-id', 'organization')];
    const views = assetLibraryScopeViewsForProject(
      project({ assets, mounts: { builtin: '/engine/assets', user: '/home/assets' } }),
    );

    expect(views.find((view) => view.scope === 'builtin')).toMatchObject({
      available: true,
      writable: false,
      assetIds: ['builtin-id'],
    });
    expect(views.find((view) => view.scope === 'user')).toMatchObject({
      available: true,
      writable: true,
      assetIds: ['user-id'],
    });
    expect(views.find((view) => view.scope === 'organization')).toMatchObject({
      available: false,
      writable: false,
      assetIds: [],
    });
  });

  it('does not derive optional mount availability from registry contents', () => {
    const views = assetLibraryScopeViewsForProject(project({ assets: [asset('org-id', 'organization')] }));

    expect(views.find((view) => view.scope === 'organization')).toMatchObject({ available: false, assetIds: [] });
  });

  it('keeps stable IDs unchanged when physical host roots move', () => {
    const assets = [asset('stable-user-guid', 'user')];
    const first = assetLibraryScopeViewsForProject(project({ assets, mounts: { user: '/first/user/root' } }));
    const second = assetLibraryScopeViewsForProject(project({ assets, mounts: { user: '/second/user/root' } }));

    expect(first.find((view) => view.scope === 'user')?.assetIds).toEqual(['stable-user-guid']);
    expect(second.find((view) => view.scope === 'user')?.assetIds).toEqual(['stable-user-guid']);
  });
});

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ProjectSnapshot } from '../services/editorHostTypes';
import { ContentBrowserPanel } from './ContentBrowserPanel';

const baseProject: ProjectSnapshot = {
  name: 'Scopes',
  root: 'D:/Scopes',
  assetRoot: 'D:/Scopes/Content',
  mounts: {
    builtin: 'C:/ARC/assets',
    project: 'D:/Scopes/Content',
    user: 'C:/Users/Test/ARC/assets',
    organization: 'Z:/Studio/ARC/assets',
  },
  activeScene: '',
  scene: [],
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
  assets: [
    {
      id: 'project-rock',
      guid: 'project-rock',
      name: 'Project Rock',
      path: 'Content/Props/rock.glb',
      kind: 'mesh' as const,
      status: 'ready' as const,
      scope: 'project' as const,
    },
    {
      id: 'builtin-grid',
      guid: 'builtin-grid',
      name: 'Built-in Grid',
      path: 'Engine/Materials/grid.arcmat',
      kind: 'material' as const,
      status: 'ready' as const,
      scope: 'builtin' as const,
      readOnly: true,
    },
    {
      id: 'user-brick',
      guid: 'user-brick',
      name: 'User Brick',
      path: 'User/Materials/brick.arcmat',
      kind: 'material' as const,
      status: 'ready' as const,
      scope: 'user' as const,
    },
    {
      id: 'org-logo',
      guid: 'org-logo',
      name: 'Organization Logo',
      path: 'Organization/Textures/logo.ktx',
      kind: 'texture' as const,
      status: 'ready' as const,
      scope: 'organization' as const,
      readOnly: true,
    },
  ],
};

afterEach(cleanup);
beforeEach(() => {
  localStorage.clear();
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      assetSources: { list: vi.fn().mockResolvedValue([]) },
      projects: {
        createAsset: vi.fn(),
        readText: vi.fn().mockRejectedValue(new Error('missing metadata')),
        writeText: vi.fn(),
      },
    },
  });
});

const renderBrowser = (project: ProjectSnapshot = baseProject) =>
  render(
    <ContentBrowserPanel
      project={project}
      cache={null}
      selectedAssetId={null}
      onSelectAsset={vi.fn()}
      onCommand={vi.fn()}
      onInstantiatePrefab={vi.fn()}
      onAssetAction={vi.fn()}
      thumbnailProvider={vi.fn().mockResolvedValue(null)}
    />,
  );

describe('ContentBrowserPanel logical scopes', () => {
  it('shows every configured logical mount with explicit access policy', () => {
    const view = renderBrowser();

    expect(view.getByRole('button', { name: 'Built-in Read only' })).toBeInTheDocument();
    expect(view.getByRole('button', { name: 'Project Writable' })).toBeInTheDocument();
    expect(view.getByRole('button', { name: 'User Writable' })).toBeInTheDocument();
    expect(view.getByRole('button', { name: 'Organization Read only' })).toBeInTheDocument();
  });

  it('switches logical scopes without changing ARC asset identity', () => {
    const onSelectAsset = vi.fn();
    const view = render(
      <ContentBrowserPanel
        project={baseProject}
        cache={null}
        selectedAssetId={null}
        onSelectAsset={onSelectAsset}
        onCommand={vi.fn()}
        onInstantiatePrefab={vi.fn()}
        onAssetAction={vi.fn()}
        thumbnailProvider={vi.fn().mockResolvedValue(null)}
      />,
    );

    fireEvent.click(view.getByRole('button', { name: 'User Writable' }));
    expect(view.getByText('User Brick')).toBeInTheDocument();
    expect(view.queryByText('Project Rock')).not.toBeInTheDocument();
    fireEvent.click(view.getByText('User Brick'));
    expect(onSelectAsset).toHaveBeenCalledWith('user-brick');
  });

  it('keeps configured empty scopes visible and omits unconfigured scopes', () => {
    const view = renderBrowser({
      ...baseProject,
      mounts: { project: 'D:/Scopes/Content', user: 'C:/Users/Test/ARC/assets' },
      assets: baseProject.assets.filter((asset) => asset.scope === 'project'),
    });

    expect(view.getByRole('button', { name: 'User Writable' })).toBeInTheDocument();
    expect(view.queryByRole('button', { name: /Built-in/ })).not.toBeInTheDocument();
    expect(view.queryByRole('button', { name: /Organization/ })).not.toBeInTheDocument();
  });
});

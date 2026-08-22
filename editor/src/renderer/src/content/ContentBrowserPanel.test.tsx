// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { ContentBrowserPanel } from './ContentBrowserPanel';

const writeText = vi.fn().mockResolvedValue(undefined);

afterEach(cleanup);
beforeEach(() => {
  writeText.mockClear();
  localStorage.clear();
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      assetSources: { list: vi.fn().mockResolvedValue([]) },
      projects: { writeText },
    },
  });
});

const project = {
  name: 'Test',
  root: 'D:/Test',
  assetRoot: 'D:/Test/Content',
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
      id: 'rock',
      guid: 'rock-guid',
      name: 'Hero Rock',
      path: 'Content/Props/hero.glb',
      kind: 'mesh' as const,
      status: 'ready' as const,
    },
    {
      id: 'sky',
      guid: 'sky-guid',
      name: 'Sky',
      path: 'Content/Environment/sky.hdr',
      kind: 'texture' as const,
      status: 'stale' as const,
    },
    {
      id: 'engine-sky-texture',
      guid: 'engine-sky-texture-guid',
      name: 'Engine Sky Texture',
      path: 'Engine/Environment/Textures/sky.ktx',
      scope: 'builtin' as const,
      readOnly: true,
      kind: 'texture' as const,
      status: 'ready' as const,
    },
    {
      id: 'engine-material-texture',
      guid: 'engine-material-texture-guid',
      name: 'Engine Material Texture',
      path: 'Engine/Materials/Textures/default.ktx',
      scope: 'builtin' as const,
      readOnly: true,
      kind: 'texture' as const,
      status: 'ready' as const,
    },
  ],
};

const renderBrowser = (onCommand = vi.fn()) =>
  render(
    <ContentBrowserPanel
      project={project}
      cache={null}
      selectedAssetId={null}
      onSelectAsset={vi.fn()}
      onCommand={onCommand}
      onInstantiatePrefab={vi.fn()}
      onAssetAction={vi.fn()}
      thumbnailProvider={vi.fn().mockResolvedValue(null)}
    />,
  );

describe('ContentBrowserPanel', () => {
  it('filters registry assets and emits GUID drag payloads', () => {
    const view = renderBrowser();
    fireEvent.change(view.getByLabelText('Search assets'), { target: { value: 'rock' } });
    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    expect(view.queryByText('Sky')).not.toBeInTheDocument();
    const transfer = { setData: vi.fn(), effectAllowed: '' };
    fireEvent.dragStart(view.getByText('Hero Rock').closest('.content-asset')!, { dataTransfer: transfer });
    expect(transfer.setData).toHaveBeenCalledWith('application/x-arc-asset', expect.stringContaining('rock-guid'));
  });

  it('supports folder navigation and list view', () => {
    const view = renderBrowser();
    const propsFolder = view.getByRole('button', { name: 'Props' });
    expect(propsFolder).toHaveClass('ui-tree-row');
    expect(propsFolder.querySelector('.entity-icon-folder')).toBeInTheDocument();

    fireEvent.click(propsFolder);
    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    expect(view.queryByText('Sky')).not.toBeInTheDocument();
    fireEvent.click(view.getByLabelText('List view'));
    expect(view.getByRole('listbox')).toHaveClass('list');
  });

  it('shows starred assets in the top-level Favorites folder', () => {
    localStorage.setItem('arc.content.favorites', JSON.stringify(['rock-guid']));
    const view = renderBrowser();

    fireEvent.click(view.getByRole('button', { name: 'Favorites' }));

    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    expect(view.queryByText('Sky')).not.toBeInTheDocument();
    expect(view.queryByText('Engine Sky Texture')).not.toBeInTheDocument();
  });

  it('renders engine folders as a nested tree without flattening duplicate leaf names', () => {
    const view = renderBrowser();
    fireEvent.click(view.getByRole('button', { name: 'Engine' }));

    const environmentFolders = view.getAllByRole('button', { name: 'Environment' });
    const engineEnvironment = environmentFolders.find((button) => button.hasAttribute('aria-expanded'));
    expect(engineEnvironment).toBeDefined();
    expect(view.getByRole('button', { name: 'Materials' })).toBeInTheDocument();
    expect(view.queryByRole('button', { name: 'Textures' })).not.toBeInTheDocument();

    fireEvent.click(engineEnvironment!);
    const environmentTextures = view.getByRole('button', { name: 'Textures' });
    fireEvent.click(environmentTextures);
    expect(view.getByText('Engine Sky Texture')).toBeInTheDocument();
    expect(view.queryByText('Engine Material Texture')).not.toBeInTheDocument();

    fireEvent.click(view.getByRole('button', { name: 'Materials' }));
    const textureFolders = view.getAllByRole('button', { name: 'Textures' });
    expect(textureFolders).toHaveLength(2);

    fireEvent.click(textureFolders[1]);
    expect(view.getByText('Engine Material Texture')).toBeInTheDocument();
    expect(view.queryByText('Engine Sky Texture')).not.toBeInTheDocument();
  });

  it('creates a PBR material in the active Content folder', async () => {
    const view = renderBrowser();
    fireEvent.click(view.getByRole('button', { name: 'Props' }));
    fireEvent.click(view.getByRole('button', { name: /Create/ }));
    fireEvent.click(view.getByRole('menuitem', { name: /Material/ }));
    fireEvent.change(view.getByLabelText('Asset name'), { target: { value: 'Rock Material' } });
    fireEvent.click(view.getByRole('button', { name: 'Create Material' }));

    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    const [path, text] = writeText.mock.calls[0] as [string, string];
    expect(path).toBe('Content/Props/Rock Material.arcmat');
    const asset = JSON.parse(text);
    expect(asset.shader).toBe('arc/default_phong');
    expect(asset.domain).toBe('surface');
    expect(asset.graph.nodes.some((node: { type: string }) => node.type === 'output')).toBe(true);
    expect(asset.graph.connections).toHaveLength(3);
  });

  it('creates a compute shader with the native .slang extension', async () => {
    const view = renderBrowser();
    fireEvent.click(view.getByRole('button', { name: /Create/ }));
    fireEvent.click(view.getByRole('menuitem', { name: /Shader/ }));
    fireEvent.change(view.getByLabelText('Asset name'), { target: { value: 'Cull Tiles' } });
    fireEvent.change(view.getByLabelText('Shader template'), { target: { value: 'compute' } });
    fireEvent.click(view.getByRole('button', { name: 'Create Shader' }));

    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(writeText.mock.calls[0][0]).toBe('Content/Cull Tiles.slang');
    expect(writeText.mock.calls[0][1]).toContain('[numthreads(8, 8, 1)]');
  });

  it('offers creation from the empty-space context menu', () => {
    const view = renderBrowser();
    fireEvent.contextMenu(view.getByRole('listbox'), { clientX: 120, clientY: 180 });
    expect(view.getByRole('menu', { name: 'Create asset' })).toBeInTheDocument();
    expect(view.getByRole('menuitem', { name: /Material/ })).toBeInTheDocument();
    expect(view.getByRole('menuitem', { name: /Shader/ })).toBeInTheDocument();
  });

  it('keeps create and import available while browsing Engine assets', async () => {
    const onCommand = vi.fn();
    const view = renderBrowser(onCommand);
    fireEvent.click(view.getByRole('button', { name: 'Engine' }));

    const create = view.getByRole('button', { name: /Create/ });
    const importButton = view.getByRole('button', { name: 'Import' });
    expect(create).not.toBeDisabled();
    expect(importButton).not.toBeDisabled();

    fireEvent.click(importButton);
    expect(onCommand).toHaveBeenCalledWith('file.importScene');

    fireEvent.click(create);
    fireEvent.click(view.getByRole('menuitem', { name: /Material/ }));
    fireEvent.change(view.getByLabelText('Asset name'), { target: { value: 'Engine View Material' } });
    fireEvent.click(view.getByRole('button', { name: 'Create Material' }));

    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    expect(writeText.mock.calls[0][0]).toBe('Content/Engine View Material.arcmat');
  });
});
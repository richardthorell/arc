// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { resetEditorDocuments } from '../editors/editorDocuments';
import type { ProjectSnapshot } from '../services/editorHostTypes';
import { ContentBrowserPanel } from './ContentBrowserPanel';

const writeText = vi.fn().mockResolvedValue(undefined);

const project: ProjectSnapshot = {
  name: 'Flow Test',
  root: 'D:/FlowTest',
  assetRoot: 'D:/FlowTest/Content',
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
};

beforeEach(() => {
  writeText.mockClear();
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: {
      assetSources: { list: vi.fn().mockResolvedValue([]) },
      projects: { writeText },
    },
  });
});

afterEach(() => {
  cleanup();
  resetEditorDocuments();
  Reflect.deleteProperty(window, 'arc');
});

describe('Content Browser Flow creation', () => {
  it('creates a Flow gameplay graph from the Create menu', async () => {
    const view = render(
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

    fireEvent.click(view.getByRole('button', { name: /Create/ }));
    expect(view.getByRole('menuitem', { name: /Flow Graph/ })).toBeInTheDocument();
    fireEvent.click(view.getByRole('menuitem', { name: /Flow Graph/ }));
    expect(view.getByRole('dialog', { name: 'Create Flow Graph' })).toBeInTheDocument();

    fireEvent.change(view.getByLabelText('Asset name'), { target: { value: 'Player Controller' } });
    fireEvent.click(view.getByRole('button', { name: 'Create Flow Graph' }));

    await waitFor(() => expect(writeText).toHaveBeenCalledTimes(1));
    const [path, text] = writeText.mock.calls[0] as [string, string];
    expect(path).toBe('Content/Player Controller.arcflow');
    const asset = JSON.parse(text);
    expect(asset).toMatchObject({ version: 1, assetType: 'flow', name: 'Player Controller' });
    expect(asset.graph.nodes).toHaveLength(1);
    expect(asset.graph.nodes[0].type).toBe('beginPlay');
  });

  it('offers Flow Graph as an asset filter', () => {
    const view = render(
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

    fireEvent.click(view.getByRole('combobox', { name: 'Asset type' }));
    expect(view.getByRole('option', { name: 'Flow Graph' })).toBeInTheDocument();
  });
});

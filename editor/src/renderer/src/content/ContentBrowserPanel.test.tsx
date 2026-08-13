// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ContentBrowserPanel } from './ContentBrowserPanel';

afterEach(cleanup);

const project = {
  name: 'Test', root: 'D:/Test', assetRoot: 'D:/Test/Content', activeScene: '', scene: [], console: [],
  renderStats: { fps: 0, frameTimeMs: 0, drawCalls: 0, triangles: 0, visibleEntities: 0, lights: 0, gpuMemoryMb: 0 },
  assets: [
    { id: 'rock', guid: 'rock-guid', name: 'Hero Rock', path: 'Content/Props/hero.glb', kind: 'mesh' as const, status: 'ready' as const },
    { id: 'sky', guid: 'sky-guid', name: 'Sky', path: 'Content/Environment/sky.hdr', kind: 'texture' as const, status: 'stale' as const },
  ],
};

describe('ContentBrowserPanel', () => {
  it('filters registry assets and emits GUID drag payloads', () => {
    const view = render(<ContentBrowserPanel project={project} cache={null} selectedAssetId={null} onSelectAsset={vi.fn()} onCommand={vi.fn()} onInstantiatePrefab={vi.fn()} onAssetAction={vi.fn()} thumbnailProvider={vi.fn().mockResolvedValue(null)}/>);
    fireEvent.change(view.getByLabelText('Search assets'), { target: { value: 'rock' } });
    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    expect(view.queryByText('Sky')).not.toBeInTheDocument();
    const transfer = { setData: vi.fn(), effectAllowed: '' };
    fireEvent.dragStart(view.getByText('Hero Rock').closest('button')!, { dataTransfer: transfer });
    expect(transfer.setData).toHaveBeenCalledWith('application/x-arc-asset', expect.stringContaining('rock-guid'));
  });

  it('supports folder navigation and list view', () => {
    const view = render(<ContentBrowserPanel project={project} cache={null} selectedAssetId={null} onSelectAsset={vi.fn()} onCommand={vi.fn()} onInstantiatePrefab={vi.fn()} onAssetAction={vi.fn()} thumbnailProvider={vi.fn().mockResolvedValue(null)}/>);
    fireEvent.click(view.getByRole('button', { name: 'Props' }));
    expect(view.getByText('Hero Rock')).toBeInTheDocument();
    expect(view.queryByText('Sky')).not.toBeInTheDocument();
    fireEvent.click(view.getByLabelText('List view'));
    expect(view.getByRole('listbox')).toHaveClass('list');
  });
});

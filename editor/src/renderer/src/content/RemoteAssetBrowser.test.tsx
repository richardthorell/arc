// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { ArcAssetSourceDescriptor } from '../../../common/assetSourceTypes';
import { RemoteAssetBrowser } from './RemoteAssetBrowser';

const source: ArcAssetSourceDescriptor = {
  id: 'polyhaven',
  displayName: 'Poly Haven',
  homepage: 'https://polyhaven.com',
  attribution: 'Powered by Poly Haven',
  licenseSummary: 'CC0',
  capabilities: { search: true, downloadManifest: true },
};

const search = vi.fn();
const manifest = vi.fn();
const importToProject = vi.fn();

beforeEach(() => {
  search.mockReset().mockResolvedValue({
    source,
    total: 1,
    assets: [{
      id: 'rock', sourceId: 'polyhaven', name: 'Granite Rock', description: 'A scanned rock.', kind: 'model',
      category: 'Nature/Rocks', tags: ['rock'], thumbnailUrl: 'https://cdn.example/rock.jpg', license: 'CC0',
      attribution: 'Powered by Poly Haven', metadata: {},
    }],
  });
  manifest.mockReset().mockResolvedValue({
    sourceId: 'polyhaven', assetId: 'rock', files: [
      { logicalPath: 'gltf/2k/gltf', url: 'https://cdn.example/rock.gltf', sizeBytes: 10 },
      { logicalPath: 'gltf/2k/gltf/include/0', url: 'https://cdn.example/rock_diff.png', sizeBytes: 20 },
      { logicalPath: 'blend/2k/blend', url: 'https://cdn.example/rock.blend', sizeBytes: 40 },
    ],
  });
  importToProject.mockReset().mockImplementation(async (_request, onProgress) => {
    onProgress?.({ phase: 'complete', completedFiles: 2, totalFiles: 2, completedBytes: 30, totalBytes: 30 });
    return {
      succeeded: true,
      destinationRoot: 'Content/External/polyhaven/rock',
      importedFiles: ['rock.gltf', 'rock_diff.png'],
      cacheHits: 0,
      downloadedFiles: 2,
      provenance: { sourceId: 'polyhaven', sourceAssetId: 'rock', importedAt: new Date().toISOString(), license: 'CC0' },
    };
  });
  Object.defineProperty(window, 'arc', {
    configurable: true,
    value: { assetSources: { search, manifest, importToProject } },
  });
});

afterEach(cleanup);

describe('RemoteAssetBrowser', () => {
  it('browses provider assets, selects a variant, and imports its dependencies', async () => {
    const view = render(<RemoteAssetBrowser source={source} />);
    await waitFor(() => expect(search).toHaveBeenCalledWith('polyhaven', expect.objectContaining({ limit: 160 })));
    const assetButton = await view.findByRole('button', { name: /Granite Rock/ });
    expect(view.getByText('Powered by Poly Haven')).toBeInTheDocument();

    fireEvent.click(assetButton);
    await waitFor(() => expect(manifest).toHaveBeenCalledWith('polyhaven', 'rock'));
    await waitFor(() => expect(view.getByLabelText('Remote asset resolution')).toHaveValue('2k'));
    expect(view.getByLabelText('Remote asset format')).toHaveValue('gltf');

    fireEvent.click(view.getByRole('button', { name: 'Import to Project' }));
    await waitFor(() => expect(importToProject).toHaveBeenCalledTimes(1));
    expect(importToProject.mock.calls[0][0]).toEqual({
      sourceId: 'polyhaven',
      assetId: 'rock',
      logicalPaths: ['gltf/2k/gltf', 'gltf/2k/gltf/include/0'],
      destinationScope: 'project',
    });
    expect(await view.findByText('Imported 2 files · 0 cache hits · 2 downloaded')).toBeInTheDocument();
  });
});
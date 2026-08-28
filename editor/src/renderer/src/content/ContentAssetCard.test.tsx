// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import { assetDisplayName, assetFileExtension, ContentAssetCard } from './ContentAssetCard';

const mesh: AssetItem = {
  id: 'cabin',
  guid: 'cabin-guid',
  name: 'SM_Cabin.glb',
  path: 'Content/Architecture/SM_Cabin.glb',
  kind: 'mesh',
  status: 'ready',
  sourceBytes: 2_621_440,
  vertexCount: 18432,
  triangleCount: 12288,
};

const texture: AssetItem = {
  id: 'albedo',
  guid: 'albedo-guid',
  name: 'Cabin_Albedo.PNG',
  path: 'Content/Architecture/Cabin_Albedo.PNG',
  kind: 'texture',
  status: 'ready',
  width: 2048,
  height: 1024,
  mipLevels: 12,
};

const renderCard = (asset: AssetItem) =>
  render(
    <ContentAssetCard
      asset={asset}
      favorite={false}
      selected={false}
      thumbnailProvider={vi.fn().mockResolvedValue(null)}
      onActivate={vi.fn()}
      onFavorite={vi.fn()}
      onReimport={vi.fn()}
      onSelect={vi.fn()}
    />,
  );

afterEach(cleanup);

describe('ContentAssetCard', () => {
  it('uses a one-line extension-free display name and asset type', () => {
    const view = renderCard(mesh);

    expect(assetDisplayName(mesh)).toBe('SM_Cabin');
    expect(assetFileExtension(mesh)).toBe('glb');
    expect(view.getByTitle('SM_Cabin')).toHaveClass('content-asset-name');
    expect(view.getByText('Mesh')).toBeVisible();
    expect(view.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it('shows general and mesh-specific information in the shared floating surface', () => {
    const view = renderCard(mesh);
    fireEvent.mouseEnter(view.getByRole('option'));
    const tooltip = view.getByRole('tooltip');

    expect(tooltip).toHaveClass('menu-dropdown', 'ui-floating-surface', 'content-asset-hover');
    expect(tooltip).toHaveTextContent('2.50 MiB');
    expect(tooltip).toHaveTextContent('.glb');
    expect(tooltip).toHaveTextContent('Content/Architecture/SM_Cabin.glb');
    expect(tooltip).toHaveTextContent('18,432');
    expect(tooltip).toHaveTextContent('12,288');
  });

  it('shows texture dimensions and mip levels when registry metadata is available', () => {
    const view = renderCard(texture);
    fireEvent.mouseEnter(view.getByRole('option'));
    const tooltip = view.getByRole('tooltip');

    expect(tooltip).toHaveTextContent('2048 × 1024');
    expect(tooltip).toHaveTextContent('12');
  });
});

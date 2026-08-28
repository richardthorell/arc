// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { act, cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import { assetDisplayName, assetFileExtension, assetHoverPosition, ContentAssetCard } from './ContentAssetCard';

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

const revealTooltip = async (view: ReturnType<typeof render>) => {
  vi.useFakeTimers();
  fireEvent.mouseEnter(view.getByRole('option'), { clientX: 120, clientY: 90 });
  expect(view.queryByRole('tooltip')).not.toBeInTheDocument();
  await act(async () => {
    await vi.advanceTimersByTimeAsync(350);
  });
  return view.getByRole('tooltip');
};

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

describe('ContentAssetCard', () => {
  it('uses a one-line extension-free display name and asset type', () => {
    const view = renderCard(mesh);

    expect(assetDisplayName(mesh)).toBe('SM_Cabin');
    expect(assetFileExtension(mesh)).toBe('glb');
    expect(view.getByTitle('SM_Cabin')).toHaveClass('content-asset-name');
    expect(view.getByText('Mesh')).toBeVisible();
    expect(view.queryByRole('tooltip')).not.toBeInTheDocument();
  });

  it('delays the hover surface and portals it outside the asset card', async () => {
    const view = renderCard(mesh);
    const tooltip = await revealTooltip(view);

    expect(tooltip).toHaveClass('menu-dropdown', 'ui-floating-surface', 'content-asset-hover');
    expect(tooltip.closest('.content-asset')).toBeNull();
    expect(tooltip.parentElement).toHaveClass('content-asset-hover-portal');
    expect(tooltip).toHaveTextContent('2.50 MiB');
    expect(tooltip).toHaveTextContent('.glb');
    expect(tooltip).toHaveTextContent('Content/Architecture/SM_Cabin.glb');
    expect(tooltip).toHaveTextContent('18,432');
    expect(tooltip).toHaveTextContent('12,288');
  });

  it('shows texture dimensions and mip levels when registry metadata is available', async () => {
    const view = renderCard(texture);
    const tooltip = await revealTooltip(view);

    expect(tooltip).toHaveTextContent('2048 × 1024');
    expect(tooltip).toHaveTextContent('12');
  });

  it('places hover details down-right when there is room and up-right near the bottom edge', () => {
    expect(assetHoverPosition({ x: 100, y: 100 }, { width: 276, height: 180 }, { width: 1200, height: 800 })).toEqual({
      left: 114,
      top: 114,
    });
    expect(assetHoverPosition({ x: 100, y: 760 }, { width: 276, height: 180 }, { width: 1200, height: 800 })).toEqual({
      left: 114,
      top: 566,
    });
  });
});

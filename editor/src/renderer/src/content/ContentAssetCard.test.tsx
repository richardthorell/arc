// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { act, cleanup, fireEvent, render } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import {
  assetDisplayName,
  assetFileExtension,
  assetHoverPosition,
  assetSpecificHoverDetails,
  ContentAssetCard,
} from './ContentAssetCard';

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

const materialWithTextureDefaults: AssetItem = {
  id: 'material',
  name: 'M_Wood.arcmat',
  path: 'Content/Materials/M_Wood.arcmat',
  kind: 'material',
  status: 'ready',
  width: 0,
  height: 0,
  mipLevels: 0,
  streamingMode: 'resident',
  materialShader: 'default_phong',
  materialParameterCount: 6,
  materialTextureCount: 2,
};

const model: AssetItem = {
  id: 'model',
  name: 'Cabin.glb',
  path: 'Content/Models/Cabin.glb',
  kind: 'scene',
  status: 'ready',
  width: 0,
  height: 0,
  vertexCount: 18432,
  triangleCount: 12288,
  meshCount: 3,
  materialSlotCount: 2,
  nodeCount: 5,
  animationCount: 0,
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

  it('shows texture metadata only for texture-like asset types', async () => {
    const textureView = renderCard(texture);
    const textureTooltip = await revealTooltip(textureView);

    expect(textureTooltip).toHaveTextContent('2048 × 1024');
    expect(textureTooltip).toHaveTextContent('12');
    cleanup();
    vi.useRealTimers();

    const materialView = renderCard(materialWithTextureDefaults);
    const materialTooltip = await revealTooltip(materialView);
    expect(materialTooltip).not.toHaveTextContent('Resolution');
    expect(materialTooltip).not.toHaveTextContent('Mip levels');
    expect(materialTooltip).not.toHaveTextContent('Streaming');
    expect(materialTooltip).toHaveTextContent('default_phong');
  });

  it('defines useful per-type details for every current asset presentation kind', () => {
    expect(assetSpecificHoverDetails(model)).toEqual([
      { label: 'Meshes', value: '3' },
      { label: 'Vertices', value: '18,432' },
      { label: 'Triangles', value: '12,288' },
      { label: 'Material slots', value: '2' },
      { label: 'Nodes', value: '5' },
      { label: 'Animations', value: '0' },
    ]);
    expect(
      assetSpecificHoverDetails({
        id: 'environment',
        name: 'Studio.hdr',
        path: 'Content/Environment/Studio.hdr',
        kind: 'environment',
        status: 'ready',
        width: 4096,
        height: 2048,
        textureFormat: 'RGBA16F',
        mipLevels: 13,
      }),
    ).toEqual([
      { label: 'Resolution', value: '4096 × 2048' },
      { label: 'Format', value: 'RGBA16F' },
      { label: 'Mip levels', value: '13' },
    ]);
    expect(
      assetSpecificHoverDetails({
        id: 'shader',
        name: 'Surface.arcshader',
        path: 'Content/Shaders/Surface.arcshader',
        kind: 'shader',
        status: 'ready',
        shaderStages: ['Vertex', 'Fragment'],
        shaderEntryPoints: ['vs_main', 'fs_main'],
        shaderCompileStatus: 'Compiled',
        shaderVariantCount: 4,
      }),
    ).toHaveLength(4);
    expect(
      assetSpecificHoverDetails({
        id: 'prefab',
        name: 'Cabin.arcprefab',
        path: 'Content/Prefabs/Cabin.arcprefab',
        kind: 'prefab',
        status: 'ready',
        entityCount: 8,
        componentCount: 21,
        nestedPrefabCount: 1,
        rootEntityName: 'Cabin',
      }),
    ).toHaveLength(4);
    expect(
      assetSpecificHoverDetails({
        id: 'scene',
        name: 'Village.arcscene',
        path: 'Content/Scenes/Village.arcscene',
        kind: 'scene',
        status: 'ready',
        entityCount: 42,
        meshCount: 12,
        cameraCount: 2,
        lightCount: 5,
      }),
    ).toHaveLength(4);
    expect(
      assetSpecificHoverDetails({
        id: 'folder',
        name: 'Props',
        path: 'Content/Props',
        kind: 'folder',
        status: 'ready',
        itemCount: 17,
      }),
    ).toEqual([{ label: 'Items', value: '17' }]);
    expect(
      assetSpecificHoverDetails({
        id: 'unknown',
        name: 'Data.bin',
        path: 'Content/Data.bin',
        kind: 'unknown',
        status: 'ready',
        width: 2048,
        height: 2048,
      }),
    ).toEqual([]);
  });

  it('keeps engine implementation metadata out of the user-facing hover', async () => {
    const view = renderCard({
      ...mesh,
      importerId: 'ufbx-importer-v2',
      residency: 'device',
      readOnly: true,
    });
    const tooltip = await revealTooltip(view);

    expect(tooltip).not.toHaveTextContent('Importer');
    expect(tooltip).not.toHaveTextContent('ufbx-importer-v2');
    expect(tooltip).not.toHaveTextContent('Residency');
    expect(tooltip).not.toHaveTextContent('Engine · Read-only');
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

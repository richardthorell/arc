import { describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import {
  assetDragType,
  assetPresentationIcon,
  assetPresentationKind,
  assetPresentationLabel,
  assetPresentationStatus,
  isContentBrowserAssetVisible,
} from './assetPresentation';

const asset = (path: string, kind: AssetItem['kind'] = 'scene') => ({ kind, path });

describe('model asset presentation', () => {
  for (const extension of ['fbx', 'obj', 'glb', 'gltf']) {
    it(`presents .${extension} imported scenes as models`, () => {
      const value = asset(`Content/Models/robot.${extension}`);
      expect(assetPresentationKind(value)).toBe('model');
      expect(assetPresentationLabel(value)).toBe('Model');
      expect(assetPresentationIcon(value)).toBe('mesh');
      expect(assetDragType(value)).toBe('mesh');
    });
  }

  it('keeps native ARC scenes as scenes', () => {
    const value = asset('Content/Scenes/demo.arcscene');
    expect(assetPresentationKind(value)).toBe('scene');
    expect(assetPresentationLabel(value)).toBe('Scene');
  });

  it('presents native mesh assets as models while preserving mesh drag semantics', () => {
    const value = asset('Content/Meshes/cube.arcmesh', 'mesh');
    expect(assetPresentationKind(value)).toBe('model');
    expect(assetPresentationLabel(value)).toBe('Model');
    expect(assetPresentationIcon(value)).toBe('mesh');
    expect(assetDragType(value)).toBe('mesh');
  });

  it('presents native Water presets as dedicated authoring assets', () => {
    const value = asset('builtin/water/presets/open_ocean.arcwater', 'water');
    expect(assetPresentationKind(value)).toBe('water');
    expect(assetPresentationLabel(value)).toBe('Water Preset');
    expect(assetPresentationIcon(value)).toBe('settings');
    expect(assetDragType(value)).toBe('water');
  });
});

describe('Engine Content Browser manifest', () => {
  it('exposes only explicitly curated built-in assets', () => {
    expect(
      isContentBrowserAssetVisible({ scope: 'builtin', path: 'builtin/materials/glass.arcmat' }),
    ).toBe(true);
    expect(
      isContentBrowserAssetVisible({ scope: 'builtin', path: 'Engine/Water/Presets/open_ocean.arcwater' }),
    ).toBe(true);
    expect(
      isContentBrowserAssetVisible({
        scope: 'builtin',
        path: 'builtin/textures/editor/default_checker_floor.png',
      }),
    ).toBe(false);
    expect(
      isContentBrowserAssetVisible({
        scope: 'builtin',
        path: 'Engine/environments/material_preview_studio_4k.exr',
      }),
    ).toBe(false);

    expect(
      isContentBrowserAssetVisible({
        scope: 'builtin',
        path: 'builtin/textures/terrain/aerial_grass_rock/aerial_grass_rock_ao_1k.jpg',
      }),
    ).toBe(false);
    expect(
      isContentBrowserAssetVisible({ scope: 'builtin', path: 'builtin/shaders/include/arc_pbr.glsl' }),
    ).toBe(false);
    expect(
      isContentBrowserAssetVisible({ scope: 'builtin', path: 'builtin/fixtures/persistence_fixture.arcscene' }),
    ).toBe(false);
  });

  it('never applies the Engine manifest to normal project assets', () => {
    expect(
      isContentBrowserAssetVisible({ scope: 'project', path: 'Content/Textures/terrain.png' }),
    ).toBe(true);
  });
});

describe('built-in source asset presentation', () => {
  it('distinguishes engine shader source from assignable material assets', () => {
    expect(
      assetPresentationLabel({
        kind: 'shader',
        path: 'builtin/shaders/default_unlit.frag',
        scope: 'builtin',
        readOnly: true,
      }),
    ).toBe('Engine Shader Source');
    expect(assetPresentationLabel({ kind: 'shader', path: 'Content/Shaders/custom.frag' })).toBe('Shader');
  });

  it('presents immutable built-in materials and shaders as source instead of stale', () => {
    const common = {
      state: 'stale' as const,
      scope: 'builtin' as const,
      readOnly: true,
      residency: 'source' as const,
      hasLastGood: false,
    };
    expect(assetPresentationStatus({ ...common, kind: 'material' })).toBe('source');
    expect(assetPresentationStatus({ ...common, kind: 'shader' })).toBe('source');
  });

  it('keeps project assets and import-dependent built-ins stale', () => {
    expect(
      assetPresentationStatus({
        kind: 'material',
        state: 'stale',
        scope: 'project',
        readOnly: false,
        residency: 'source',
        hasLastGood: false,
      }),
    ).toBe('stale');
    expect(
      assetPresentationStatus({
        kind: 'texture',
        state: 'stale',
        scope: 'builtin',
        readOnly: true,
        residency: 'source',
        hasLastGood: false,
      }),
    ).toBe('stale');
  });
});

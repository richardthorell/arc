import { describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import {
  assetDragType,
  assetPresentationIcon,
  assetPresentationKind,
  assetPresentationLabel,
  assetPresentationStatus,
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

  it('presents Material Instances as assignable material-family assets', () => {
    const value = asset('Content/Materials/Floor.arcmatinst', 'materialInstance');
    expect(assetPresentationKind(value)).toBe('materialInstance');
    expect(assetPresentationLabel(value)).toBe('Material Instance');
    expect(assetPresentationIcon(value)).toBe('material');
    expect(assetDragType(value)).toBe('materialInstance');
  });

  it('presents native Water presets as dedicated authoring assets', () => {
    const value = asset('builtin/water/presets/open_ocean.arcwater', 'water');
    expect(assetPresentationKind(value)).toBe('water');
    expect(assetPresentationLabel(value)).toBe('Water Preset');
    expect(assetPresentationIcon(value)).toBe('settings');
    expect(assetDragType(value)).toBe('water');
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

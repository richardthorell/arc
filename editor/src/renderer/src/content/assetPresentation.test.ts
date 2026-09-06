import { describe, expect, it } from 'vitest';

import type { AssetItem } from '../services/editorHostTypes';
import {
  assetDragType,
  assetPresentationIcon,
  assetPresentationKind,
  assetPresentationLabel,
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
});

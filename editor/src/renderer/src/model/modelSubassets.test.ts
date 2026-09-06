import { describe, expect, it } from 'vitest';
import type { AssetItem } from '../services/editorHostTypes';
import { buildModelSubassets, skeletonCompatibility, skeletonHierarchyDepth } from './modelSubassets';

const model = (patch: Partial<AssetItem> = {}): AssetItem => ({
  id: 'hand',
  name: 'Hand.fbx',
  path: 'Content/Hand.fbx',
  kind: 'scene',
  status: 'ready',
  meshCount: 2,
  skeletonBoneCount: 3,
  skeletonJoints: [
    { index: 0, name: 'Root', parent: -1 },
    { index: 1, name: 'Palm', parent: 0 },
    { index: 2, name: 'Finger', parent: 1 },
  ],
  ...patch,
});

describe('model sub-assets', () => {
  it('keeps mesh and skeleton resources nested under the model', () => {
    expect(buildModelSubassets(model()).map((item) => item.kind)).toEqual(['mesh', 'mesh', 'skeleton']);
  });
  it('computes skeleton compatibility conservatively', () => {
    expect(skeletonCompatibility(model(), model({ id: 'other' }))).toBe('compatible');
    expect(skeletonCompatibility(model(), model({ skeletonJoints: undefined, skeletonBoneCount: 3 }))).toBe('partial');
    expect(skeletonCompatibility(model(), model({ skeletonBoneCount: 4 }))).toBe('incompatible');
  });
  it('derives hierarchy depth from stable joint indices', () => {
    expect(skeletonHierarchyDepth(model())).toBe(3);
  });
});

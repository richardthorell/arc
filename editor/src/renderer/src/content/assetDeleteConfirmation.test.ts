import { describe, expect, it } from 'vitest';
import { buildAssetDependencyIndex } from './assetDependencyOperations';
import { describeAssetDeleteConfirmation } from './assetDeleteConfirmation';

describe('asset delete confirmation', () => {
  it('requires confirmation for a safe non-empty delete', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
    ]);

    expect(describeAssetDeleteConfirmation(index, ['material-a', 'texture-a'])).toEqual({
      assetIds: ['material-a', 'texture-a'],
      internalReferenceCount: 1,
      blockingReferences: [],
      blockingAssetIds: [],
      requiresConfirmation: true,
      blocked: false,
    });
  });

  it('reports deterministic external blockers without duplicating assets', () => {
    const index = buildAssetDependencyIndex([
      { sourceAssetId: 'scene-b', targetAssetId: 'material-a', kind: 'material' },
      { sourceAssetId: 'scene-a', targetAssetId: 'material-a', kind: 'material' },
      { sourceAssetId: 'scene-a', targetAssetId: 'texture-a', kind: 'texture' },
      { sourceAssetId: 'material-a', targetAssetId: 'texture-a', kind: 'texture' },
    ]);

    const confirmation = describeAssetDeleteConfirmation(index, ['texture-a', 'material-a']);

    expect(confirmation.assetIds).toEqual(['material-a', 'texture-a']);
    expect(confirmation.internalReferenceCount).toBe(1);
    expect(confirmation.blockingAssetIds).toEqual(['scene-a', 'scene-b']);
    expect(confirmation.blockingReferences).toHaveLength(3);
    expect(confirmation.requiresConfirmation).toBe(false);
    expect(confirmation.blocked).toBe(true);
  });

  it('does not present an empty selection as blocked or confirmable', () => {
    expect(describeAssetDeleteConfirmation(buildAssetDependencyIndex([]), [])).toEqual({
      assetIds: [],
      internalReferenceCount: 0,
      blockingReferences: [],
      blockingAssetIds: [],
      requiresConfirmation: false,
      blocked: false,
    });
  });
});

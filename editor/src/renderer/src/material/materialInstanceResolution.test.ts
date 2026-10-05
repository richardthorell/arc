import { describe, expect, it } from 'vitest';
import {
  resetMaterialInstanceOverride,
  resolveMaterialInstance,
  type MaterialInstanceAsset,
} from './materialInstanceResolution';

const instance: MaterialInstanceAsset = {
  id: 'instance:paint-red',
  parentMaterialId: 'material:paint',
  overrides: [{ parameterId: 'roughness-id', value: 0.2 }],
};

describe('material instance resolution', () => {
  it('preserves parent identity and resolves overrides by stable parameter id', () => {
    const resolved = resolveMaterialInstance(instance, [
      { id: 'base-color-id', name: 'Base Color', value: [1, 1, 1, 1] },
      { id: 'roughness-id', name: 'Surface Roughness', value: 0.7 },
    ]);

    expect(resolved.parentMaterialId).toBe('material:paint');
    expect(resolved.parameters).toEqual([
      { id: 'base-color-id', name: 'Base Color', value: [1, 1, 1, 1], inherited: true },
      { id: 'roughness-id', name: 'Surface Roughness', value: 0.2, inherited: false },
    ]);
  });

  it('survives parent parameter renames because names are presentation only', () => {
    const resolved = resolveMaterialInstance(instance, [
      { id: 'roughness-id', name: 'Perceptual Roughness', value: 0.7 },
    ]);

    expect(resolved.parameters[0]).toEqual({
      id: 'roughness-id',
      name: 'Perceptual Roughness',
      value: 0.2,
      inherited: false,
    });
  });

  it('reports stale overrides deterministically and does not promote them to parameters', () => {
    const resolved = resolveMaterialInstance(
      {
        ...instance,
        overrides: [
          { parameterId: 'removed-z', value: 1 },
          { parameterId: 'roughness-id', value: 0.4 },
          { parameterId: 'removed-a', value: 2 },
        ],
      },
      [{ id: 'roughness-id', name: 'Roughness', value: 0.7 }],
    );

    expect(resolved.staleOverrideIds).toEqual(['removed-a', 'removed-z']);
    expect(resolved.parameters).toHaveLength(1);
  });

  it('resets an override without mutating the source asset', () => {
    const reset = resetMaterialInstanceOverride(instance, 'roughness-id');

    expect(reset.overrides).toEqual([]);
    expect(instance.overrides).toHaveLength(1);
    expect(resolveMaterialInstance(reset, [{ id: 'roughness-id', name: 'Roughness', value: 0.7 }]).parameters[0])
      .toMatchObject({ value: 0.7, inherited: true });
  });

  it('defensively copies vector-like values', () => {
    const parentValue = [1, 1, 1, 1];
    const overrideValue = [0.8, 0.2, 0.1, 1];
    const resolved = resolveMaterialInstance(
      { ...instance, overrides: [{ parameterId: 'base-color-id', value: overrideValue }] },
      [{ id: 'base-color-id', name: 'Base Color', value: parentValue }],
    );

    expect(resolved.parameters[0]?.value).not.toBe(overrideValue);
    expect(resolved.parameters[0]?.value).toEqual(overrideValue);
  });
});

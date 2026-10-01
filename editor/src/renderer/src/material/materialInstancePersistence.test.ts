import { describe, expect, it } from 'vitest';

import {
  deserializeMaterialInstanceAsset,
  MATERIAL_INSTANCE_ASSET_VERSION,
  serializeMaterialInstanceAsset,
} from './materialInstancePersistence';

describe('material instance persistence', () => {
  it('round-trips parent identity and stable parameter overrides', () => {
    const serialized = serializeMaterialInstanceAsset({
      parentMaterialId: 'materials/paint',
      overrides: [
        { parameterId: 'roughness', value: 0.2 },
        { parameterId: 'tint', value: [1, 0.5, 0.25, 1] },
      ],
    });

    expect(JSON.parse(serialized).version).toBe(MATERIAL_INSTANCE_ASSET_VERSION);
    expect(deserializeMaterialInstanceAsset(serialized)).toEqual({
      parentMaterialId: 'materials/paint',
      overrides: [
        { parameterId: 'roughness', value: 0.2 },
        { parameterId: 'tint', value: [1, 0.5, 0.25, 1] },
      ],
    });
  });

  it('does not persist inherited parent defaults', () => {
    const serialized = serializeMaterialInstanceAsset({
      parentMaterialId: ' parent-id ',
      overrides: [{ parameterId: ' roughness ', value: 0.4 }],
    });

    expect(JSON.parse(serialized)).toEqual({
      version: 1,
      parentMaterialId: 'parent-id',
      overrides: [{ parameterId: 'roughness', value: 0.4 }],
    });
  });

  it('rejects malformed, duplicate, and unsupported persisted state', () => {
    expect(deserializeMaterialInstanceAsset('not json')).toBeNull();
    expect(deserializeMaterialInstanceAsset('{"version":2,"parentMaterialId":"parent","overrides":[]}')).toBeNull();
    expect(
      deserializeMaterialInstanceAsset(
        '{"version":1,"parentMaterialId":"parent","overrides":[{"parameterId":"x","value":1},{"parameterId":"x","value":2}]}',
      ),
    ).toBeNull();
  });

  it('rejects invalid authored identity before serialization', () => {
    expect(() => serializeMaterialInstanceAsset({ parentMaterialId: ' ', overrides: [] })).toThrow();
    expect(() =>
      serializeMaterialInstanceAsset({
        parentMaterialId: 'parent',
        overrides: [
          { parameterId: 'x', value: 1 },
          { parameterId: ' x ', value: 2 },
        ],
      }),
    ).toThrow('Duplicate material instance override: x');
  });
});

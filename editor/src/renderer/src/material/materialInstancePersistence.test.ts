import { describe, expect, it } from 'vitest';

import {
  deserializeMaterialInstanceAsset,
  MATERIAL_INSTANCE_ASSET_VERSION,
  materialFunctionSlotParameterId,
  materialParameterId,
  serializeMaterialInstanceAsset,
} from './materialInstancePersistence';

const parent = {
  guid: '11111111-1111-1111-1111-111111111111',
  pathHint: 'materials/standard_lit.arcmat',
};

describe('material instance persistence', () => {
  it('round-trips parent identity, parameters, and Function Slot overrides', () => {
    const serialized = serializeMaterialInstanceAsset({
      version: MATERIAL_INSTANCE_ASSET_VERSION,
      name: 'Floor',
      parent,
      parameterOverrides: [
        { parameterId: '101', value: 0.2 },
        { parameterId: '202', value: [1, 0.5, 0.25, 1] },
      ],
      functionOverrides: [
        {
          slotId: 'base-color-source',
          function: {
            guid: '22222222-2222-2222-2222-222222222222',
            pathHint: 'material_functions/checker.arcmatfn',
          },
          inputOverrides: [{ pinId: 'cell-size', value: 1 }],
        },
      ],
    });

    expect(deserializeMaterialInstanceAsset(serialized)).toEqual({
      version: MATERIAL_INSTANCE_ASSET_VERSION,
      name: 'Floor',
      parent,
      parameterOverrides: [
        { parameterId: '101', value: 0.2 },
        { parameterId: '202', value: [1, 0.5, 0.25, 1] },
      ],
      functionOverrides: [
        {
          slotId: 'base-color-source',
          function: {
            guid: '22222222-2222-2222-2222-222222222222',
            pathHint: 'material_functions/checker.arcmatfn',
          },
          inputOverrides: [{ pinId: 'cell-size', value: 1 }],
        },
      ],
    });
  });

  it('does not persist inherited parent defaults', () => {
    const serialized = serializeMaterialInstanceAsset({
      version: 1,
      name: 'Clean',
      parent,
      parameterOverrides: [],
      functionOverrides: [],
    });

    expect(JSON.parse(serialized)).toEqual({
      version: 1,
      name: 'Clean',
      parent,
      parameterOverrides: [],
      functionOverrides: [],
    });
  });

  it('rejects malformed, duplicate, and unsupported persisted state', () => {
    expect(deserializeMaterialInstanceAsset('not json')).toBeNull();
    expect(
      deserializeMaterialInstanceAsset(
        JSON.stringify({
          version: 2,
          name: 'Bad',
          parent,
          parameterOverrides: [],
          functionOverrides: [],
        }),
      ),
    ).toBeNull();
    expect(
      deserializeMaterialInstanceAsset(
        JSON.stringify({
          version: 1,
          name: 'Bad',
          parent,
          parameterOverrides: [
            { parameterId: '1', value: 1 },
            { parameterId: '1', value: 2 },
          ],
          functionOverrides: [],
        }),
      ),
    ).toBeNull();
  });

  it('uses the same stable FNV parameter identity as native material reflection', () => {
    expect(materialParameterId('roughness')).toBe('14293098357166276437');
    expect(materialFunctionSlotParameterId('base-color', 'function-guid', 'scale')).toBe(
      materialParameterId('slot::base-color::function-guid::scale'),
    );
  });
});

import { describe, expect, it } from 'vitest';

import {
  MATERIAL_PARAMETER_METADATA_VERSION,
  parseMaterialParameterMetadata,
  serializeMaterialParameterMetadata,
} from './materialParameterMetadataPersistence';

describe('material parameter metadata persistence', () => {
  it('serializes deterministic normalized metadata by stable id', () => {
    expect(
      serializeMaterialParameterMetadata([
        { id: 'roughness', name: 'Roughness', group: ' Surface ', order: 8, description: '  Help text  ' },
        { id: 'tint', name: 'Tint', group: 'surface', order: 2 },
      ]),
    ).toEqual({
      version: MATERIAL_PARAMETER_METADATA_VERSION,
      parameters: [
        { id: 'roughness', group: 'Surface', order: 1, description: 'Help text' },
        { id: 'tint', group: 'Surface', order: 0, description: undefined },
      ],
    });
  });

  it('rejects duplicate stable ids before persistence', () => {
    expect(() =>
      serializeMaterialParameterMetadata([
        { id: 'same', name: 'A' },
        { id: 'same', name: 'B' },
      ]),
    ).toThrow(/unique non-empty ids/);
  });

  it('parses a valid payload without rebinding metadata by name', () => {
    expect(
      parseMaterialParameterMetadata({
        version: 1,
        parameters: [
          { id: 'stale-but-stable', group: ' Surface ', order: 4, description: '  Legacy  ' },
          { id: 'active', order: 0 },
        ],
      }),
    ).toEqual({
      version: 1,
      parameters: [
        { id: 'active', group: undefined, order: 0, description: undefined },
        { id: 'stale-but-stable', group: 'Surface', order: 4, description: 'Legacy' },
      ],
    });
  });

  it.each([
    null,
    {},
    { version: 2, parameters: [] },
    { version: 1, parameters: {} },
    { version: 1, parameters: [{ id: '', order: 0 }] },
    { version: 1, parameters: [{ id: 'a', order: -1 }] },
    {
      version: 1,
      parameters: [
        { id: 'a', order: 0 },
        { id: 'a', order: 1 },
      ],
    },
    { version: 1, parameters: [{ id: 'a', order: 0, group: 4 }] },
  ])('rejects malformed payload %#', (payload) => {
    expect(parseMaterialParameterMetadata(payload)).toBeNull();
  });
});

import { describe, expect, it } from 'vitest';

import type { MaterialGraphNode } from './materialGraphTypes';
import {
  defaultMaterialParameterGroup,
  groupMaterialParameters,
  materialParameterPresentation,
} from './materialParameterPresentation';

const node = (
  id: string,
  name: string,
  options: { exposed?: boolean; group?: string; sortOrder?: number } = {},
): MaterialGraphNode => ({
  id,
  type: 'constant',
  position: [0, 0],
  values: { value: 0 },
  parameter: {
    exposed: options.exposed ?? true,
    name,
    group: options.group,
    sortOrder: options.sortOrder,
  },
});

describe('material parameter presentation', () => {
  it('omits hidden and unnamed parameters', () => {
    expect(materialParameterPresentation(node('hidden', 'Hidden', { exposed: false }))).toBeNull();
    expect(materialParameterPresentation(node('unnamed', '   '))).toBeNull();
  });

  it('normalizes optional authoring metadata', () => {
    expect(materialParameterPresentation(node('roughness', ' Roughness ', { group: ' Surface ', sortOrder: 2 }))).toEqual({
      nodeId: 'roughness',
      name: 'Roughness',
      group: 'Surface',
      sortOrder: 2,
    });
    expect(materialParameterPresentation(node('metallic', 'Metallic'))?.group).toBe(defaultMaterialParameterGroup);
  });

  it('groups exposed parameters deterministically', () => {
    expect(
      groupMaterialParameters([
        node('roughness', 'Roughness', { group: 'Surface', sortOrder: 20 }),
        node('tint', 'Tint', { group: 'Appearance', sortOrder: 10 }),
        node('metallic', 'Metallic', { group: 'Surface', sortOrder: 10 }),
        node('alpha', 'Alpha'),
        node('hidden', 'Hidden', { exposed: false }),
      ]),
    ).toEqual([
      { name: 'Appearance', parameters: [{ nodeId: 'tint', name: 'Tint', group: 'Appearance', sortOrder: 10 }] },
      { name: 'Parameters', parameters: [{ nodeId: 'alpha', name: 'Alpha', group: 'Parameters', sortOrder: 0 }] },
      {
        name: 'Surface',
        parameters: [
          { nodeId: 'metallic', name: 'Metallic', group: 'Surface', sortOrder: 10 },
          { nodeId: 'roughness', name: 'Roughness', group: 'Surface', sortOrder: 20 },
        ],
      },
    ]);
  });
});

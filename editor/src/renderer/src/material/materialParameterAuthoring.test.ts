import { describe, expect, it } from 'vitest';

import { createDefaultMaterialGraph, createMaterialNode } from './materialGraphTypes';
import { materialAuthoringParameters, materialParameterGroups } from './materialParameterAuthoring';

describe('material parameter authoring metadata', () => {
  it('provides deterministic defaults for existing exposed parameters', () => {
    const graph = createDefaultMaterialGraph();

    expect(
      materialAuthoringParameters(graph).map(({ name, group, sortOrder }) => ({ name, group, sortOrder })),
    ).toEqual([
      { name: 'Base Color', group: 'Parameters', sortOrder: 0 },
      { name: 'Metallic', group: 'Parameters', sortOrder: 0 },
      { name: 'Roughness', group: 'Parameters', sortOrder: 0 },
    ]);
  });

  it('normalizes presentation metadata and orders parameters by group, order, name, then stable node id', () => {
    const graph = createDefaultMaterialGraph();
    const roughness = graph.nodes.find((node) => node.parameter?.name === 'Roughness');
    const metallic = graph.nodes.find((node) => node.parameter?.name === 'Metallic');
    expect(roughness?.parameter).toBeDefined();
    expect(metallic?.parameter).toBeDefined();

    Object.assign(roughness!.parameter!, {
      group: ' Surface ',
      description: '  Microsurface response  ',
      sortOrder: 20,
    });
    Object.assign(metallic!.parameter!, { group: 'Surface', sortOrder: 10 });

    const detail = createMaterialNode('constant', [200, 200], { value: 1 });
    detail.parameter = { exposed: true, name: 'Detail' };
    Object.assign(detail.parameter, { group: 'Detail', sortOrder: Number.NaN });
    graph.nodes.push(detail);

    expect(
      materialAuthoringParameters(graph).map(({ name, group, description, sortOrder }) => ({
        name,
        group,
        description,
        sortOrder,
      })),
    ).toEqual([
      { name: 'Detail', group: 'Detail', description: undefined, sortOrder: 0 },
      { name: 'Base Color', group: 'Parameters', description: undefined, sortOrder: 0 },
      { name: 'Metallic', group: 'Surface', description: undefined, sortOrder: 10 },
      { name: 'Roughness', group: 'Surface', description: 'Microsurface response', sortOrder: 20 },
    ]);
  });

  it('groups parameters without changing compiler-owned type metadata', () => {
    const graph = createDefaultMaterialGraph();
    for (const node of graph.nodes) {
      if (node.parameter)
        Object.assign(node.parameter, { group: node.parameter.name === 'Base Color' ? 'Color' : 'Surface' });
    }

    expect(
      materialParameterGroups(graph).map((group) => ({
        name: group.name,
        parameters: group.parameters.map((parameter) => ({ name: parameter.name, type: parameter.type })),
      })),
    ).toEqual([
      { name: 'Color', parameters: [{ name: 'Base Color', type: 'vec3' }] },
      {
        name: 'Surface',
        parameters: [
          { name: 'Metallic', type: 'float' },
          { name: 'Roughness', type: 'float' },
        ],
      },
    ]);
  });
});

import { describe, expect, it } from 'vitest';

import { buildAssetCreation } from './assetCreation';

const project = {
  root: 'D:/Project',
  assetRoot: 'D:/Project/Content',
};

describe('Flow asset creation', () => {
  it('creates an .arcflow asset with the Flow v1 authoring schema', () => {
    const definition = buildAssetCreation(project, {
      kind: 'flow',
      name: 'PlayerController',
      folder: 'Content/Logic',
    });

    expect(definition.asset).toMatchObject({
      name: 'PlayerController.arcflow',
      path: 'Content/Logic/PlayerController.arcflow',
      kind: 'flow',
      scope: 'project',
      status: 'ready',
    });

    const asset = JSON.parse(definition.contents) as {
      version: number;
      assetType: string;
      name: string;
      graph: { version: number; nodes: Array<{ type: string }>; connections: unknown[]; variables: unknown[] };
    };
    expect(asset).toMatchObject({ version: 1, assetType: 'flow', name: 'PlayerController' });
    expect(asset.graph.version).toBe(1);
    expect(asset.graph.nodes.map((node) => node.type)).toEqual(['beginPlay']);
    expect(asset.graph.connections).toEqual([]);
    expect(asset.graph.variables).toEqual([]);
  });

  it('strips an explicit .arcflow extension from the entered asset name', () => {
    const definition = buildAssetCreation(project, {
      kind: 'flow',
      name: 'Enemy.arcflow',
      folder: 'Content/Logic',
    });

    expect(definition.asset.path).toBe('Content/Logic/Enemy.arcflow');
  });
});

import { describe, expect, it } from 'vitest';

import { materialGraphDomain } from './materialGraphDomain';
import type { MaterialGraphNode } from './materialGraphTypes';

const textureNode = (dimension: '2d' | 'cube' | '3d'): MaterialGraphNode => ({
  id: `texture-${dimension}`,
  type: 'textureSample',
  position: [0, 0],
  values: { texture: '', dimension },
});

describe('material texture sample dimensions', () => {
  it('uses vec2 UVs for Texture2D', () => {
    const definition = materialGraphDomain.getNodeDefinition(textureNode('2d'));
    expect(definition.inputs[0]).toMatchObject({ id: 'uv', label: 'UV', type: 'vec2' });
  });

  it('uses vec3 directions for TextureCube', () => {
    const definition = materialGraphDomain.getNodeDefinition(textureNode('cube'));
    expect(definition.inputs[0]).toMatchObject({ id: 'uv', label: 'Direction', type: 'vec3' });
  });

  it('uses vec3 UVW coordinates for Texture3D', () => {
    const definition = materialGraphDomain.getNodeDefinition(textureNode('3d'));
    expect(definition.inputs[0]).toMatchObject({ id: 'uv', label: 'UVW', type: 'vec3' });
  });
});

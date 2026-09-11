import { describe, expect, it } from 'vitest';

import { materialGraphDomain } from './materialGraphDomain';
import { createMaterialNode } from './materialGraphTypes';

describe('materialGraphDomain', () => {
  it('adapts material definitions to the shared graph domain contract', () => {
    const node = createMaterialNode('multiply', [10, 20]);
    expect(materialGraphDomain.getNodeDefinition(node).title).toBe('Multiply');
    expect(materialGraphDomain.getNodeDefinitions().some((definition) => definition.type === 'textureSample')).toBe(
      true,
    );
  });

  it('keeps material-specific protection and connection rules behind the domain boundary', () => {
    const source = createMaterialNode('constant', [0, 0]);
    const target = createMaterialNode('multiply', [100, 0]);
    const sourcePin = materialGraphDomain.getNodeDefinition(source).outputs[0]!;
    const targetPin = materialGraphDomain.getNodeDefinition(target).inputs[0]!;

    expect(
      materialGraphDomain.canConnect(
        { node: source, pin: sourcePin, direction: 'output' },
        { node: target, pin: targetPin, direction: 'input' },
      ),
    ).toEqual({ allowed: true });
    expect(
      materialGraphDomain.canConnect(
        { node: source, pin: sourcePin, direction: 'input' },
        { node: target, pin: targetPin, direction: 'input' },
      ).allowed,
    ).toBe(false);
    expect(materialGraphDomain.canDeleteNode(createMaterialNode('output', [0, 0]))).toBe(false);
  });
});

import { describe, expect, it } from 'vitest';

import { flowGraphDomain, flowPinTypesCompatible } from './flowGraphDomain';
import { createFlowNode, flowNodeDefinitions } from './flowGraphTypes';

describe('Flow graph domain', () => {
  it('keeps execution and value connections separate', () => {
    expect(flowPinTypesCompatible({ kind: 'execution' }, { kind: 'execution' })).toBe(true);
    expect(flowPinTypesCompatible({ kind: 'execution' }, { kind: 'value', valueType: 'bool' })).toBe(false);
    expect(flowPinTypesCompatible({ kind: 'value', valueType: 'bool' }, { kind: 'value', valueType: 'bool' })).toBe(
      true,
    );
    expect(flowPinTypesCompatible({ kind: 'value', valueType: 'bool' }, { kind: 'value', valueType: 'float' })).toBe(
      false,
    );
  });

  it('allows any value pins to bridge typed values', () => {
    expect(flowPinTypesCompatible({ kind: 'value', valueType: 'any' }, { kind: 'value', valueType: 'entity' })).toBe(
      true,
    );
  });

  it('rejects self connections and reversed connection direction', () => {
    const branch = createFlowNode('branch', [0, 0]);
    const definition = flowNodeDefinitions.branch;
    const input = definition.inputs.find((pin) => pin.id === 'exec')!;
    const output = definition.outputs.find((pin) => pin.id === 'true')!;

    expect(
      flowGraphDomain.canConnect(
        { node: branch, pin: output, direction: 'output' },
        { node: branch, pin: input, direction: 'input' },
      ).allowed,
    ).toBe(false);
    expect(
      flowGraphDomain.canConnect(
        { node: branch, pin: input, direction: 'input' },
        { node: createFlowNode('branch', [300, 0]), pin: output, direction: 'output' },
      ).allowed,
    ).toBe(false);
  });

  it('connects deferred entity outputs to structural gameplay writes', () => {
    const create = createFlowNode('createEntity', [0, 0]);
    const destroy = createFlowNode('destroyEntity', [300, 0]);
    const entity = flowNodeDefinitions.createEntity.outputs.find((pin) => pin.id === 'entity')!;
    const target = flowNodeDefinitions.destroyEntity.inputs.find((pin) => pin.id === 'entity')!;

    expect(
      flowGraphDomain.canConnect(
        { node: create, pin: entity, direction: 'output' },
        { node: destroy, pin: target, direction: 'input' },
      ).allowed,
    ).toBe(true);
  });

  it('allows execution between different nodes', () => {
    const begin = createFlowNode('beginPlay', [0, 0]);
    const branch = createFlowNode('branch', [300, 0]);
    const from = flowNodeDefinitions.beginPlay.outputs[0];
    const to = flowNodeDefinitions.branch.inputs[0];

    expect(
      flowGraphDomain.canConnect(
        { node: begin, pin: from, direction: 'output' },
        { node: branch, pin: to, direction: 'input' },
      ).allowed,
    ).toBe(true);
  });

  it('connects Self Entity and typed literals to gameplay nodes', () => {
    const self = createFlowNode('selfEntity', [0, 0]);
    const name = createFlowNode('stringLiteral', [0, 120], { value: 'Player' });
    const setName = createFlowNode('setName', [320, 0]);

    const selfPin = flowNodeDefinitions.selfEntity.outputs.find((pin) => pin.id === 'entity')!;
    const entityInput = flowNodeDefinitions.setName.inputs.find((pin) => pin.id === 'entity')!;
    const namePin = flowNodeDefinitions.stringLiteral.outputs.find((pin) => pin.id === 'value')!;
    const nameInput = flowNodeDefinitions.setName.inputs.find((pin) => pin.id === 'name')!;

    expect(
      flowGraphDomain.canConnect(
        { node: self, pin: selfPin, direction: 'output' },
        { node: setName, pin: entityInput, direction: 'input' },
      ).allowed,
    ).toBe(true);
    expect(
      flowGraphDomain.canConnect(
        { node: name, pin: namePin, direction: 'output' },
        { node: setName, pin: nameInput, direction: 'input' },
      ).allowed,
    ).toBe(true);
  });

  it('keeps transform inputs strongly typed', () => {
    const vector3 = createFlowNode('vector3Literal', [0, 0]);
    const vector4 = createFlowNode('vector4Literal', [0, 120]);
    const setTransform = createFlowNode('setTransform', [320, 0]);
    const position = flowNodeDefinitions.setTransform.inputs.find((pin) => pin.id === 'position')!;
    const rotation = flowNodeDefinitions.setTransform.inputs.find((pin) => pin.id === 'rotation')!;

    expect(
      flowGraphDomain.canConnect(
        { node: vector3, pin: flowNodeDefinitions.vector3Literal.outputs[0], direction: 'output' },
        { node: setTransform, pin: position, direction: 'input' },
      ).allowed,
    ).toBe(true);
    expect(
      flowGraphDomain.canConnect(
        { node: vector3, pin: flowNodeDefinitions.vector3Literal.outputs[0], direction: 'output' },
        { node: setTransform, pin: rotation, direction: 'input' },
      ).allowed,
    ).toBe(false);
    expect(
      flowGraphDomain.canConnect(
        { node: vector4, pin: flowNodeDefinitions.vector4Literal.outputs[0], direction: 'output' },
        { node: setTransform, pin: rotation, direction: 'input' },
      ).allowed,
    ).toBe(true);
  });
});

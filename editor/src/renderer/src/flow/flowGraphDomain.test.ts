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
});

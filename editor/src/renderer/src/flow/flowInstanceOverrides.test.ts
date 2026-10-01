import { describe, expect, it } from 'vitest';

import type { FlowGraph } from './flowGraphTypes';
import {
  getExposedFlowVariables,
  isFlowOverrideValueCompatible,
  reconcileFlowInstanceOverrides,
} from './flowInstanceOverrides';

const graph = (variables: FlowGraph['variables']): FlowGraph => ({
  version: 1,
  variables,
  nodes: [],
  connections: [],
  viewport: { x: 0, y: 0, zoom: 1 },
});

describe('flow instance overrides', () => {
  it('exposes only variables explicitly authored for instances', () => {
    const source = graph([
      { id: 'speed', name: 'Speed', type: 'float', defaultValue: 1, exposed: true },
      { id: 'scratch', name: 'Scratch', type: 'int', defaultValue: 0, exposed: false },
    ]);

    expect(getExposedFlowVariables(source).map((variable) => variable.id)).toEqual(['speed']);
  });

  it('preserves compatible overrides by stable variable id across recompilation', () => {
    const recompiled = graph([
      { id: 'speed', name: 'Movement Speed', type: 'float', defaultValue: 2, exposed: true },
    ]);

    expect(reconcileFlowInstanceOverrides(recompiled, [{ variableId: 'speed', value: 4.5 }])).toEqual({
      overrides: [{ variableId: 'speed', value: 4.5 }],
      diagnostics: [],
    });
  });

  it('reports removed, hidden, and type-changed variables instead of silently applying stale values', () => {
    const recompiled = graph([
      { id: 'hidden', name: 'Hidden', type: 'bool', defaultValue: false, exposed: false },
      { id: 'count', name: 'Count', type: 'int', defaultValue: 0, exposed: true },
    ]);

    const result = reconcileFlowInstanceOverrides(recompiled, [
      { variableId: 'removed', value: 1 },
      { variableId: 'hidden', value: true },
      { variableId: 'count', value: 1.5 },
    ]);

    expect(result.overrides).toEqual([]);
    expect(result.diagnostics.map((diagnostic) => diagnostic.reason)).toEqual([
      'missing-variable',
      'not-exposed',
      'type-mismatch',
    ]);
  });

  it('validates scalar and vector values deterministically', () => {
    expect(isFlowOverrideValueCompatible('int', 2)).toBe(true);
    expect(isFlowOverrideValueCompatible('int', 2.5)).toBe(false);
    expect(isFlowOverrideValueCompatible('vec3', [1, 2, 3])).toBe(true);
    expect(isFlowOverrideValueCompatible('vec3', [1, 2])).toBe(false);
    expect(isFlowOverrideValueCompatible('float', Number.NaN)).toBe(false);
  });
});

import { describe, expect, it } from 'vitest';

import type { FlowGraph } from './flowGraphTypes';
import { reconcilePersistedFlowInstanceOverrides } from './flowInstanceOverrideState';

const graph = (variables: FlowGraph['variables']): FlowGraph => ({
  version: 1,
  variables,
  nodes: [],
  connections: [],
  viewport: { x: 0, y: 0, zoom: 1 },
});

describe('persisted Flow instance override state', () => {
  it('preserves compatible overrides by stable variable id across rename', () => {
    const result = reconcilePersistedFlowInstanceOverrides(
      graph([{ id: 'speed', name: 'Movement Speed', type: 'float', defaultValue: 1, exposed: true }]),
      { version: 1, overrides: [{ variableId: 'speed', value: 4.5 }] },
    );

    expect(result).toEqual({
      ok: true,
      persisted: { version: 1, overrides: [{ variableId: 'speed', value: 4.5 }] },
      diagnostics: [],
    });
  });

  it('drops stale and incompatible overrides with actionable diagnostics', () => {
    const result = reconcilePersistedFlowInstanceOverrides(
      graph([
        { id: 'speed', name: 'Speed', type: 'float', defaultValue: 1, exposed: true },
        { id: 'hidden', name: 'Internal', type: 'bool', defaultValue: false, exposed: false },
      ]),
      {
        version: 1,
        overrides: [
          { variableId: 'speed', value: 'fast' },
          { variableId: 'hidden', value: true },
          { variableId: 'removed', value: 2 },
        ],
      },
    );

    expect(result.ok).toBe(true);
    if (!result.ok) return;
    expect(result.persisted.overrides).toEqual([]);
    expect(result.diagnostics.map((diagnostic) => diagnostic.reason)).toEqual([
      'type-mismatch',
      'not-exposed',
      'missing-variable',
    ]);
  });

  it('rejects malformed persisted state before graph reconciliation', () => {
    expect(reconcilePersistedFlowInstanceOverrides(graph([]), { version: 2, overrides: [] })).toEqual({
      ok: false,
      error: 'Unsupported Flow instance override version: 2',
    });
  });
});

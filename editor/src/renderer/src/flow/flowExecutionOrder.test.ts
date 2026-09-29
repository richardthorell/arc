import { describe, expect, it } from 'vitest';
import { getFlowExecutionOutputOrder, resolveFlowExecutionOrder } from './flowExecutionOrder';
import type { FlowGraphConnection } from './flowGraphTypes';

const execution = (
  id: string,
  fromPin: string,
  toNode: string,
  kind: FlowGraphConnection['kind'] = 'execution',
): FlowGraphConnection => ({
  id,
  kind,
  from: { nodeId: 'sequence', pin: fromPin },
  to: { nodeId: toNode, pin: 'exec' },
});

describe('Flow execution order', () => {
  it('uses language declaration order for multi-output control nodes', () => {
    expect(getFlowExecutionOutputOrder('sequence')).toEqual(['then0', 'then1', 'then2', 'then3']);
    expect(getFlowExecutionOutputOrder('branch')).toEqual(['true', 'false']);
    expect(getFlowExecutionOutputOrder('add')).toEqual([]);
  });

  it('is deterministic regardless of connection insertion or canvas order', () => {
    const connections = [
      execution('c3', 'then2', 'third'),
      execution('c1', 'then0', 'first'),
      execution('c2', 'then1', 'second'),
    ];

    expect(resolveFlowExecutionOrder({ id: 'sequence', type: 'sequence' }, connections)).toEqual([
      { connectionId: 'c1', outputPinId: 'then0', targetNodeId: 'first', targetPinId: 'exec', ordinal: 0 },
      { connectionId: 'c2', outputPinId: 'then1', targetNodeId: 'second', targetPinId: 'exec', ordinal: 1 },
      { connectionId: 'c3', outputPinId: 'then2', targetNodeId: 'third', targetPinId: 'exec', ordinal: 2 },
    ]);
  });

  it('ignores value edges and stale output pins', () => {
    const connections = [
      execution('exec', 'then0', 'valid'),
      execution('value', 'then1', 'ignored-value', 'value'),
      execution('stale', 'removed-pin', 'ignored-stale'),
    ];

    expect(resolveFlowExecutionOrder({ id: 'sequence', type: 'sequence' }, connections)).toHaveLength(1);
  });

  it('uses stable connection ids as the deterministic same-pin fallback', () => {
    const connections = [execution('z', 'then0', 'later'), execution('a', 'then0', 'earlier')];
    expect(resolveFlowExecutionOrder({ id: 'sequence', type: 'sequence' }, connections).map((step) => step.connectionId))
      .toEqual(['a', 'z']);
  });
});

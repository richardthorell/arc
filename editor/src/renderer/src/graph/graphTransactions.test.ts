import { describe, expect, it } from 'vitest';

import { applyGraphTransaction, type GraphTransaction } from './graphTransactions';

type GraphState = {
  nodes: string[];
  connections: string[];
};

const initialState: GraphState = { nodes: ['a'], connections: [] };

describe('graphTransactions', () => {
  it('commits all operations as one logical edit', () => {
    const transaction: GraphTransaction<GraphState> = {
      kind: 'paste',
      label: 'Paste nodes',
      operations: [
        { id: 'create-b', apply: (state) => ({ ...state, nodes: [...state.nodes, 'b'] }) },
        { id: 'connect-a-b', apply: (state) => ({ ...state, connections: [...state.connections, 'a:b'] }) },
      ],
    };

    expect(applyGraphTransaction(initialState, transaction)).toEqual({
      committed: true,
      state: { nodes: ['a', 'b'], connections: ['a:b'] },
    });
    expect(initialState).toEqual({ nodes: ['a'], connections: [] });
  });

  it('returns the original state when an operation fails so partial edits cannot publish', () => {
    const transaction: GraphTransaction<GraphState> = {
      kind: 'connect',
      label: 'Connect nodes',
      operations: [
        { id: 'prepare', apply: (state) => ({ ...state, nodes: [...state.nodes, 'b'] }) },
        {
          id: 'invalid-connection',
          apply: () => {
            throw new Error('incompatible pins');
          },
        },
      ],
    };

    const result = applyGraphTransaction(initialState, transaction);
    expect(result.committed).toBe(false);
    expect(result.state).toBe(initialState);
    if (!result.committed) expect(result.failedOperationId).toBe('invalid-connection');
  });

  it('supports an empty transaction without changing state identity', () => {
    const result = applyGraphTransaction(initialState, {
      kind: 'move',
      label: 'Move nodes',
      operations: [],
    });

    expect(result).toEqual({ committed: true, state: initialState });
    expect(result.state).toBe(initialState);
  });
});

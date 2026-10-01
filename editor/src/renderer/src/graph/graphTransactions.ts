export type GraphTransactionKind = 'move' | 'connect' | 'create' | 'delete' | 'paste';

export type GraphTransactionOperation<T> = {
  id: string;
  apply: (state: T) => T;
};

export type GraphTransaction<T> = {
  kind: GraphTransactionKind;
  label: string;
  operations: readonly GraphTransactionOperation<T>[];
};

export type GraphTransactionResult<T> =
  | { committed: true; state: T }
  | { committed: false; state: T; failedOperationId: string; error: unknown };

/**
 * Applies a logical graph edit as one atomic transaction.
 *
 * Operations must be immutable: each operation receives the state produced by
 * the previous operation and returns a new state. If any operation fails, the
 * original state is returned so callers never publish a partially-applied edit
 * to their undo stack or document store.
 */
export const applyGraphTransaction = <T>(state: T, transaction: GraphTransaction<T>): GraphTransactionResult<T> => {
  let nextState = state;

  for (const operation of transaction.operations) {
    try {
      nextState = operation.apply(nextState);
    } catch (error) {
      return {
        committed: false,
        state,
        failedOperationId: operation.id,
        error,
      };
    }
  }

  return { committed: true, state: nextState };
};

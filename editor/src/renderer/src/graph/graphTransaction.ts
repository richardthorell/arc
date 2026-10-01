export interface GraphTransactionHost<TToken> {
  /** Opens one domain-owned undo/redo transaction for a user-visible graph edit. */
  begin: (label: string) => TToken;
  /** Publishes all edits recorded under the token as one undoable operation. */
  commit: (token: TToken) => void;
  /** Reverts every edit recorded under the token. Must not publish partial history. */
  rollback: (token: TToken) => void;
}

/**
 * Runs a graph edit inside one explicit transaction boundary.
 *
 * Graph domains keep ownership of their mutation and history implementation;
 * this helper only standardizes the begin/commit/rollback lifecycle shared by
 * Material, Flow, and future graph editors. A failed operation is rolled back
 * before the original error is rethrown, preventing callers from accidentally
 * committing a partial move/connect/create/delete/paste edit.
 */
export function runGraphTransaction<TToken, TResult>(
  host: GraphTransactionHost<TToken>,
  label: string,
  operation: (token: TToken) => TResult,
): TResult {
  const token = host.begin(label);

  try {
    const result = operation(token);
    host.commit(token);
    return result;
  } catch (error) {
    host.rollback(token);
    throw error;
  }
}

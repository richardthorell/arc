export type RemoteImportTransactionState =
  | 'pending'
  | 'downloading'
  | 'importing'
  | 'committed'
  | 'cancelled'
  | 'failed';

export type RemoteImportTransaction = Readonly<{
  state: RemoteImportTransactionState;
  stagedPaths: readonly string[];
  publishedAssetIds: readonly string[];
}>;

export const createRemoteImportTransaction = (): RemoteImportTransaction => ({
  state: 'pending',
  stagedPaths: [],
  publishedAssetIds: [],
});

const terminalStates: ReadonlySet<RemoteImportTransactionState> = new Set([
  'committed',
  'cancelled',
  'failed',
]);

const assertMutable = (transaction: RemoteImportTransaction): void => {
  if (terminalStates.has(transaction.state)) {
    throw new Error(`Remote import transaction is already ${transaction.state}`);
  }
};

export const beginRemoteImportDownload = (
  transaction: RemoteImportTransaction,
): RemoteImportTransaction => {
  assertMutable(transaction);
  if (transaction.state !== 'pending') throw new Error('Remote import download can only begin once');
  return { ...transaction, state: 'downloading' };
};

export const stageRemoteImportPath = (
  transaction: RemoteImportTransaction,
  path: string,
): RemoteImportTransaction => {
  assertMutable(transaction);
  if (transaction.state !== 'downloading') throw new Error('Remote import files can only be staged while downloading');
  if (!path || transaction.stagedPaths.includes(path)) return transaction;
  return { ...transaction, stagedPaths: [...transaction.stagedPaths, path] };
};

export const beginRemoteAssetImport = (
  transaction: RemoteImportTransaction,
): RemoteImportTransaction => {
  assertMutable(transaction);
  if (transaction.state !== 'downloading') throw new Error('Remote asset import requires a completed download stage');
  return { ...transaction, state: 'importing' };
};

export const publishRemoteImportedAsset = (
  transaction: RemoteImportTransaction,
  assetId: string,
): RemoteImportTransaction => {
  assertMutable(transaction);
  if (transaction.state !== 'importing') throw new Error('Remote assets can only be published while importing');
  if (!assetId || transaction.publishedAssetIds.includes(assetId)) return transaction;
  return { ...transaction, publishedAssetIds: [...transaction.publishedAssetIds, assetId] };
};

/**
 * Cancellation deliberately returns cleanup ownership to the caller. Staged
 * files and partially published assets stay enumerated until cleanup succeeds,
 * preventing a cancelled job from being mistaken for an empty/safe import.
 */
export const cancelRemoteImport = (
  transaction: RemoteImportTransaction,
): RemoteImportTransaction => {
  assertMutable(transaction);
  return { ...transaction, state: 'cancelled' };
};

export const failRemoteImport = (
  transaction: RemoteImportTransaction,
): RemoteImportTransaction => {
  assertMutable(transaction);
  return { ...transaction, state: 'failed' };
};

export const remoteImportCleanup = (
  transaction: RemoteImportTransaction,
): Readonly<{ stagedPaths: readonly string[]; publishedAssetIds: readonly string[] }> => ({
  stagedPaths: transaction.stagedPaths,
  publishedAssetIds: transaction.publishedAssetIds,
});

/**
 * A remote import is publishable only after callers have completed the import
 * stage. Commit does not erase cleanup records; it only seals the transaction.
 */
export const commitRemoteImport = (
  transaction: RemoteImportTransaction,
): RemoteImportTransaction => {
  assertMutable(transaction);
  if (transaction.state !== 'importing') throw new Error('Remote import can only commit from the importing state');
  return { ...transaction, state: 'committed' };
};

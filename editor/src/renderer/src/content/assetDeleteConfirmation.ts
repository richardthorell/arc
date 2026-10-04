import {
  type AssetDependencyIndex,
  type AssetReference,
  planAssetDeleteTransaction,
} from './assetDependencyOperations';

export type AssetDeleteConfirmation = {
  assetIds: readonly string[];
  internalReferenceCount: number;
  blockingReferences: readonly AssetReference[];
  blockingAssetIds: readonly string[];
  requiresConfirmation: boolean;
  blocked: boolean;
};

/**
 * Derives presentation state for delete confirmation from the authoritative
 * dependency-aware transaction plan. UI surfaces should consume this model
 * instead of reimplementing dependency safety rules.
 */
export const describeAssetDeleteConfirmation = (
  index: AssetDependencyIndex,
  assetIds: readonly string[],
): AssetDeleteConfirmation => {
  const transaction = planAssetDeleteTransaction(index, assetIds);
  const blockingAssetIds = [...new Set(transaction.blockingReferences.map((reference) => reference.sourceAssetId))].sort();

  return {
    assetIds: transaction.assetIds,
    internalReferenceCount: transaction.internalReferences.length,
    blockingReferences: transaction.blockingReferences,
    blockingAssetIds,
    requiresConfirmation: transaction.executable,
    blocked: transaction.assetIds.length > 0 && !transaction.executable,
  };
};

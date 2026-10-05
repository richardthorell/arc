import {
  type AssetDependencyIndex,
  type AssetReference,
  findAssetUsages,
  findTransitiveDependentAssetIds,
} from './assetDependencyOperations';

export type AssetUsagePresentation = {
  assetId: string;
  directReferences: readonly AssetReference[];
  directAssetIds: readonly string[];
  transitiveAssetIds: readonly string[];
  hasUsages: boolean;
};

/**
 * Builds deterministic presentation data for Find Usages surfaces from the
 * authoritative dependency index. UI components should consume this model
 * rather than independently traversing or de-duplicating asset references.
 */
export const describeAssetUsages = (index: AssetDependencyIndex, assetId: string): AssetUsagePresentation => {
  const directReferences = findAssetUsages(index, assetId);
  const directAssetIds = [...new Set(directReferences.map((reference) => reference.sourceAssetId))].sort();
  const directAssetIdSet = new Set(directAssetIds);
  const transitiveAssetIds = findTransitiveDependentAssetIds(index, assetId).filter(
    (dependentAssetId) => !directAssetIdSet.has(dependentAssetId),
  );

  return {
    assetId,
    directReferences,
    directAssetIds,
    transitiveAssetIds,
    hasUsages: directReferences.length > 0,
  };
};

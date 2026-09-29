export type AssetReference = {
  sourceAssetId: string;
  targetAssetId: string;
  kind?: string;
};

export type AssetDependencyIndex = ReadonlyMap<string, readonly AssetReference[]>;

export type AssetDependencyImpact = {
  assetId: string;
  directDependents: readonly AssetReference[];
  transitiveDependentAssetIds: readonly string[];
};

export type AssetDeletePlan = {
  assetId: string;
  dependents: readonly AssetReference[];
  safe: boolean;
};

export type AssetRelocationPlan = {
  assetId: string;
  fromPath: string;
  toPath: string;
  preserveIdentity: true;
};

const byReferenceIdentity = (left: AssetReference, right: AssetReference) => {
  const source = left.sourceAssetId.localeCompare(right.sourceAssetId);
  if (source !== 0) return source;
  const target = left.targetAssetId.localeCompare(right.targetAssetId);
  if (target !== 0) return target;
  return (left.kind ?? '').localeCompare(right.kind ?? '');
};

export const buildAssetDependencyIndex = (references: readonly AssetReference[]): AssetDependencyIndex => {
  const index = new Map<string, AssetReference[]>();

  for (const reference of references) {
    const dependents = index.get(reference.targetAssetId) ?? [];
    dependents.push(reference);
    index.set(reference.targetAssetId, dependents);
  }

  for (const dependents of index.values()) dependents.sort(byReferenceIdentity);
  return index;
};

export const findAssetUsages = (index: AssetDependencyIndex, assetId: string): readonly AssetReference[] =>
  index.get(assetId) ?? [];

export const findTransitiveDependentAssetIds = (index: AssetDependencyIndex, assetId: string): readonly string[] => {
  const visited = new Set<string>([assetId]);
  const pending = [...findAssetUsages(index, assetId).map((reference) => reference.sourceAssetId)].sort();
  const dependents: string[] = [];

  while (pending.length > 0) {
    const current = pending.shift()!;
    if (visited.has(current)) continue;

    visited.add(current);
    dependents.push(current);

    for (const reference of findAssetUsages(index, current)) {
      if (!visited.has(reference.sourceAssetId)) pending.push(reference.sourceAssetId);
    }
    pending.sort();
  }

  return dependents;
};

export const describeAssetDependencyImpact = (index: AssetDependencyIndex, assetId: string): AssetDependencyImpact => ({
  assetId,
  directDependents: findAssetUsages(index, assetId),
  transitiveDependentAssetIds: findTransitiveDependentAssetIds(index, assetId),
});

export const planAssetDelete = (index: AssetDependencyIndex, assetId: string): AssetDeletePlan => {
  const dependents = findAssetUsages(index, assetId);
  return { assetId, dependents, safe: dependents.length === 0 };
};

export const planAssetRelocation = (assetId: string, fromPath: string, toPath: string): AssetRelocationPlan => {
  if (fromPath === toPath) throw new Error('Asset relocation requires a different destination path');
  return { assetId, fromPath, toPath, preserveIdentity: true };
};

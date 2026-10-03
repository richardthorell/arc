import {
  describeAssetDependencyImpact,
  type AssetDependencyIndex,
  type AssetReference,
} from './assetDependencyOperations';

export type AssetUsageDescriptor = {
  assetId: string;
  label: string;
  logicalPath?: string;
};

export type AssetUsageRow = {
  assetId: string;
  label: string;
  logicalPath?: string;
  kind?: string;
  depth: number;
  direct: boolean;
};

export type AssetFindUsagesResult = {
  assetId: string;
  directUsageCount: number;
  rows: readonly AssetUsageRow[];
};

const byUsageRow = (left: AssetUsageRow, right: AssetUsageRow) => {
  if (left.depth !== right.depth) return left.depth - right.depth;
  const label = left.label.localeCompare(right.label);
  if (label !== 0) return label;
  return left.assetId.localeCompare(right.assetId);
};

const describeUsage = (
  reference: AssetReference,
  descriptors: ReadonlyMap<string, AssetUsageDescriptor>,
  depth: number,
  direct: boolean,
): AssetUsageRow => {
  const descriptor = descriptors.get(reference.sourceAssetId);
  return {
    assetId: reference.sourceAssetId,
    label: descriptor?.label ?? reference.sourceAssetId,
    logicalPath: descriptor?.logicalPath,
    kind: reference.kind,
    depth,
    direct,
  };
};

export const buildAssetFindUsagesResult = (
  index: AssetDependencyIndex,
  assetId: string,
  descriptors: ReadonlyMap<string, AssetUsageDescriptor>,
): AssetFindUsagesResult => {
  const impact = describeAssetDependencyImpact(index, assetId);
  const directByAssetId = new Map(impact.directDependents.map((reference) => [reference.sourceAssetId, reference]));
  const rowsByAssetId = new Map<string, AssetUsageRow>();

  for (const reference of impact.directDependents) {
    rowsByAssetId.set(reference.sourceAssetId, describeUsage(reference, descriptors, 1, true));
  }

  for (const dependentAssetId of impact.transitiveDependentAssetIds) {
    if (rowsByAssetId.has(dependentAssetId)) continue;

    // The dependency index is target -> references. Find the nearest already-known
    // ancestor deterministically so the UI can indent transitive usages without
    // making presentation state part of asset identity.
    let depth = 2;
    let kind: string | undefined;
    for (const [targetAssetId, references] of index) {
      const reference = references.find((candidate) => candidate.sourceAssetId === dependentAssetId);
      if (!reference) continue;
      kind = reference.kind;
      const parent = rowsByAssetId.get(targetAssetId);
      if (parent) depth = parent.depth + 1;
      break;
    }

    const descriptor = descriptors.get(dependentAssetId);
    rowsByAssetId.set(dependentAssetId, {
      assetId: dependentAssetId,
      label: descriptor?.label ?? dependentAssetId,
      logicalPath: descriptor?.logicalPath,
      kind,
      depth,
      direct: directByAssetId.has(dependentAssetId),
    });
  }

  return {
    assetId,
    directUsageCount: impact.directDependents.length,
    rows: [...rowsByAssetId.values()].sort(byUsageRow),
  };
};

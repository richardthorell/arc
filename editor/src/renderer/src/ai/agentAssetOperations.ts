export type AgentAssetMoveOperation = {
  kind: 'asset.move';
  assetIds: readonly string[];
  destinationFolder: string;
  expectedProjectRevision: number;
};

export type AgentAssetDeleteOperation = {
  kind: 'asset.delete';
  assetIds: readonly string[];
  expectedProjectRevision: number;
};

export type AgentAssetOperation = AgentAssetMoveOperation | AgentAssetDeleteOperation;

const requireStableAssetIds = (values: readonly string[]): string[] => {
  const assetIds = [...new Set(values.map((value) => value.trim()).filter(Boolean))];
  if (assetIds.length === 0) throw new Error('Asset mutation requires at least one stable asset ID');
  return assetIds;
};

const requireRevision = (value: number): number => {
  if (!Number.isSafeInteger(value) || value < 0) {
    throw new Error('Project revision must be a non-negative integer');
  }
  return value;
};

const requireProjectFolder = (value: string): string => {
  const normalized = value.trim().replace(/\\/g, '/').replace(/^\.\//, '');
  if (!normalized || normalized.startsWith('/') || normalized.split('/').includes('..')) {
    throw new Error('Asset destination must stay inside the project asset workspace');
  }
  return normalized.replace(/\/+$/, '');
};

/**
 * Normalizes asset mutations before they reach the authoritative asset service.
 *
 * Agent-authored asset work is expressed through stable asset identities and an
 * expected project revision rather than filesystem paths or raw file deletion.
 * Execution remains responsible for authorization, dependency/reference checks,
 * transaction/undo integration, and returning the resulting stable identities.
 */
export const normalizeAgentAssetOperation = (operation: AgentAssetOperation): AgentAssetOperation => {
  const assetIds = requireStableAssetIds(operation.assetIds);
  const expectedProjectRevision = requireRevision(operation.expectedProjectRevision);

  if (operation.kind === 'asset.move') {
    return {
      kind: 'asset.move',
      assetIds,
      destinationFolder: requireProjectFolder(operation.destinationFolder),
      expectedProjectRevision,
    };
  }

  return { kind: 'asset.delete', assetIds, expectedProjectRevision };
};

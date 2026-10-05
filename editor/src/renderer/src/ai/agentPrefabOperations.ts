export type AgentPrefabCreateOperation = {
  kind: 'prefab.create';
  entityIds: readonly string[];
  destinationPath: string;
  expectedSceneRevision: number;
};

export type AgentPrefabInstantiateOperation = {
  kind: 'prefab.instantiate';
  assetId: string;
  expectedAssetRevision: number;
  expectedSceneRevision: number;
  parentEntityId?: string;
};

export type AgentPrefabOperation = AgentPrefabCreateOperation | AgentPrefabInstantiateOperation;

const requireStableId = (value: string, label: string): string => {
  const normalized = value.trim();
  if (!normalized) throw new Error(`${label} requires a stable ID`);
  return normalized;
};

const requireRevision = (value: number, label: string): number => {
  if (!Number.isSafeInteger(value) || value < 0) throw new Error(`${label} must be a non-negative integer`);
  return value;
};

/**
 * Normalizes prefab mutations before they reach the editor/native authorization boundary.
 *
 * The contract deliberately carries stable entity/asset identities and expected revisions;
 * callers cannot express prefab work as raw document replacement. Native/editor execution
 * remains responsible for project-boundary authorization, validation, transactions and undo.
 */
export const normalizeAgentPrefabOperation = (operation: AgentPrefabOperation): AgentPrefabOperation => {
  const expectedSceneRevision = requireRevision(operation.expectedSceneRevision, 'Scene revision');

  if (operation.kind === 'prefab.create') {
    const entityIds = [...new Set(operation.entityIds.map((id) => requireStableId(id, 'Prefab source entity')))];
    if (entityIds.length === 0) throw new Error('Prefab creation requires at least one source entity');

    const destinationPath = operation.destinationPath.trim();
    if (!destinationPath) throw new Error('Prefab creation requires a destination path');
    if (destinationPath.startsWith('/') || destinationPath.includes('..')) {
      throw new Error('Prefab destination must stay inside the project asset workspace');
    }

    return { kind: 'prefab.create', entityIds, destinationPath, expectedSceneRevision };
  }

  return {
    kind: 'prefab.instantiate',
    assetId: requireStableId(operation.assetId, 'Prefab asset'),
    expectedAssetRevision: requireRevision(operation.expectedAssetRevision, 'Prefab asset revision'),
    expectedSceneRevision,
    ...(operation.parentEntityId
      ? { parentEntityId: requireStableId(operation.parentEntityId, 'Prefab parent entity') }
      : {}),
  };
};

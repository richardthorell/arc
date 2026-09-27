export type AgentAssetRollbackRecord = {
  operationId: string;
  path: string;
  operation: 'create' | 'replace' | 'remove';
  beforeRevision?: string;
  beforeContent?: string;
  committedRevision?: string;
};

export type AgentAssetRollbackRequest = {
  record: AgentAssetRollbackRecord;
  currentRevision?: string;
};

export type AgentAssetRollbackPlan =
  | { ok: true; action: 'remove'; path: string; expectedRevision: string }
  | { ok: true; action: 'restore'; path: string; expectedRevision?: string; content: string }
  | { ok: false; reason: string };

const isNonEmpty = (value: string | undefined): value is string =>
  typeof value === 'string' && value.trim().length > 0;

export const validateAgentAssetRollbackRecord = (record: AgentAssetRollbackRecord): string | undefined => {
  if (!isNonEmpty(record.operationId)) return 'Rollback record requires a stable operation ID.';
  if (!isNonEmpty(record.path)) return 'Rollback record requires an asset path.';

  if (record.operation === 'create') {
    if (!isNonEmpty(record.committedRevision)) return 'Created assets require the committed revision.';
    if (record.beforeRevision !== undefined || record.beforeContent !== undefined) {
      return 'Created assets cannot contain a prior asset revision.';
    }
    return undefined;
  }

  if (!isNonEmpty(record.beforeRevision) || record.beforeContent === undefined) {
    return 'Replaced or removed assets require the prior revision and content.';
  }
  if (record.operation === 'replace' && !isNonEmpty(record.committedRevision)) {
    return 'Replaced assets require the committed revision.';
  }
  if (record.operation === 'remove' && record.committedRevision !== undefined) {
    return 'Removed assets cannot have a committed revision.';
  }
  return undefined;
};

export const planAgentAssetRollback = ({
  record,
  currentRevision,
}: AgentAssetRollbackRequest): AgentAssetRollbackPlan => {
  const error = validateAgentAssetRollbackRecord(record);
  if (error) return { ok: false, reason: error };

  if (record.operation === 'create') {
    if (currentRevision !== record.committedRevision) {
      return { ok: false, reason: 'Asset revision changed after the agent operation; refusing rollback.' };
    }
    return { ok: true, action: 'remove', path: record.path, expectedRevision: record.committedRevision! };
  }

  if (record.operation === 'replace') {
    if (currentRevision !== record.committedRevision) {
      return { ok: false, reason: 'Asset revision changed after the agent operation; refusing rollback.' };
    }
    return {
      ok: true,
      action: 'restore',
      path: record.path,
      expectedRevision: record.committedRevision,
      content: record.beforeContent!,
    };
  }

  if (currentRevision !== undefined) {
    return { ok: false, reason: 'Removed asset path is occupied; refusing rollback.' };
  }
  return { ok: true, action: 'restore', path: record.path, content: record.beforeContent! };
};

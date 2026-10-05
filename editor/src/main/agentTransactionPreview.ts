export type AgentChangeKind = 'create' | 'update' | 'delete';
export type AgentChangeDomain = 'scene' | 'component' | 'asset' | 'graph';
export type AgentTransactionRisk = 'low' | 'medium' | 'high';

export type AgentTransactionChange = Readonly<{
  kind: AgentChangeKind;
  domain: AgentChangeDomain;
  targetId: string;
  targetLabel?: string;
  before?: Readonly<Record<string, unknown>>;
  after?: Readonly<Record<string, unknown>>;
}>;

export type AgentTransactionPreview = Readonly<{
  transactionId: string;
  expectedRevision: string;
  changes: readonly AgentTransactionChange[];
  affectedTargets: number;
  risk: AgentTransactionRisk;
  requiresApproval: boolean;
  summary: string;
}>;

export type AgentTransactionPreviewPolicy = Readonly<{
  mediumChangeCount: number;
  highChangeCount: number;
}>;

function requireNonEmpty(value: string, name: string): string {
  const normalized = value.trim();
  if (!normalized) throw new Error(`${name} must not be empty`);
  return normalized;
}

function validatePolicy(policy: AgentTransactionPreviewPolicy): void {
  if (!Number.isInteger(policy.mediumChangeCount) || policy.mediumChangeCount <= 0) {
    throw new Error('mediumChangeCount must be a positive integer');
  }
  if (!Number.isInteger(policy.highChangeCount) || policy.highChangeCount < policy.mediumChangeCount) {
    throw new Error('highChangeCount must be an integer greater than or equal to mediumChangeCount');
  }
}

function cloneRecord(value: Readonly<Record<string, unknown>> | undefined): Readonly<Record<string, unknown>> | undefined {
  return value ? structuredClone(value) : undefined;
}

function validateChange(change: AgentTransactionChange): AgentTransactionChange {
  const targetId = requireNonEmpty(change.targetId, 'targetId');
  if (change.kind === 'create' && change.before !== undefined) throw new Error('create changes cannot contain before state');
  if (change.kind === 'delete' && change.after !== undefined) throw new Error('delete changes cannot contain after state');
  if (change.kind === 'update' && (change.before === undefined || change.after === undefined)) {
    throw new Error('update changes require before and after state');
  }
  if (change.kind === 'create' && change.after === undefined) throw new Error('create changes require after state');
  if (change.kind === 'delete' && change.before === undefined) throw new Error('delete changes require before state');

  return {
    ...change,
    targetId,
    ...(change.targetLabel?.trim() ? { targetLabel: change.targetLabel.trim() } : {}),
    ...(change.before ? { before: cloneRecord(change.before) } : {}),
    ...(change.after ? { after: cloneRecord(change.after) } : {}),
  };
}

/**
 * Builds an approval-friendly snapshot of the exact revision-owned changes a transaction intends to commit.
 * Inputs are cloned and never mutated so preview generation cannot alter authoring state.
 */
export function buildAgentTransactionPreview(
  transactionId: string,
  expectedRevision: string,
  changes: readonly AgentTransactionChange[],
  policy: AgentTransactionPreviewPolicy,
): AgentTransactionPreview {
  validatePolicy(policy);
  const normalizedTransactionId = requireNonEmpty(transactionId, 'transactionId');
  const normalizedRevision = requireNonEmpty(expectedRevision, 'expectedRevision');
  if (changes.length === 0) throw new Error('transaction preview requires at least one change');

  const normalizedChanges = changes.map(validateChange);
  const identities = new Set<string>();
  for (const change of normalizedChanges) {
    const identity = `${change.domain}:${change.targetId}`;
    if (identities.has(identity)) throw new Error(`duplicate transaction target: ${identity}`);
    identities.add(identity);
  }

  const affectedTargets = normalizedChanges.length;
  const destructive = normalizedChanges.some((change) => change.kind === 'delete');
  const risk: AgentTransactionRisk =
    destructive || affectedTargets >= policy.highChangeCount
      ? 'high'
      : affectedTargets >= policy.mediumChangeCount
        ? 'medium'
        : 'low';

  const counts = normalizedChanges.reduce<Record<AgentChangeKind, number>>(
    (result, change) => ({ ...result, [change.kind]: result[change.kind] + 1 }),
    { create: 0, update: 0, delete: 0 },
  );
  const parts = (Object.keys(counts) as AgentChangeKind[])
    .filter((kind) => counts[kind] > 0)
    .map((kind) => `${counts[kind]} ${kind}`);

  return {
    transactionId: normalizedTransactionId,
    expectedRevision: normalizedRevision,
    changes: normalizedChanges,
    affectedTargets,
    risk,
    requiresApproval: risk !== 'low',
    summary: `${affectedTargets} target${affectedTargets === 1 ? '' : 's'}: ${parts.join(', ')}`,
  };
}

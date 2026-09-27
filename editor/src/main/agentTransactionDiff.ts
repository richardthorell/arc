export type AgentTransactionChange = {
  kind: 'scene' | 'asset';
  operation: string;
  target: { guid?: string; path?: string; component?: string };
  before?: unknown;
  after?: unknown;
};

export type AgentTransactionDiffPreview = {
  changeCount: number;
  sceneChangeCount: number;
  assetChangeCount: number;
  destructiveChangeCount: number;
  risk: 'low' | 'medium' | 'high';
  summary: string;
  changes: AgentTransactionChange[];
};

const destructiveOperations = new Set(['delete', 'removeAsset', 'replaceAsset']);

const clone = (value: unknown): unknown => {
  if (value === undefined) return undefined;
  return JSON.parse(JSON.stringify(value));
};

export const createAgentTransactionDiffPreview = (
  changes: readonly AgentTransactionChange[],
): AgentTransactionDiffPreview => {
  const normalized = changes.map((change) => ({
    kind: change.kind,
    operation: change.operation,
    target: { ...change.target },
    ...(change.before !== undefined ? { before: clone(change.before) } : {}),
    ...(change.after !== undefined ? { after: clone(change.after) } : {}),
  }));
  const sceneChangeCount = normalized.filter((change) => change.kind === 'scene').length;
  const assetChangeCount = normalized.length - sceneChangeCount;
  const destructiveChangeCount = normalized.filter((change) => destructiveOperations.has(change.operation)).length;
  const risk =
    destructiveChangeCount > 0 || normalized.length >= 20
      ? 'high'
      : assetChangeCount > 0 || normalized.length >= 5
        ? 'medium'
        : 'low';
  const parts = [
    sceneChangeCount > 0 ? `${sceneChangeCount} scene change${sceneChangeCount === 1 ? '' : 's'}` : '',
    assetChangeCount > 0 ? `${assetChangeCount} asset change${assetChangeCount === 1 ? '' : 's'}` : '',
  ].filter(Boolean);
  return {
    changeCount: normalized.length,
    sceneChangeCount,
    assetChangeCount,
    destructiveChangeCount,
    risk,
    summary: parts.length > 0 ? parts.join(', ') : 'No changes',
    changes: normalized,
  };
};

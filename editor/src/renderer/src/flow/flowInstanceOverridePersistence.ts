import type { FlowInstanceOverride } from './flowInstanceOverrides';

export const FLOW_INSTANCE_OVERRIDE_SCHEMA_VERSION = 1 as const;

export type PersistedFlowInstanceOverrides = {
  version: typeof FLOW_INSTANCE_OVERRIDE_SCHEMA_VERSION;
  overrides: FlowInstanceOverride[];
};

export type FlowInstanceOverrideParseResult =
  { ok: true; overrides: FlowInstanceOverride[] } | { ok: false; error: string };

const isOverride = (value: unknown): value is FlowInstanceOverride => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) return false;
  const candidate = value as Record<string, unknown>;
  return typeof candidate.variableId === 'string' && candidate.variableId.trim().length > 0 && 'value' in candidate;
};

export const serializeFlowInstanceOverrides = (
  overrides: readonly FlowInstanceOverride[],
): PersistedFlowInstanceOverrides => ({
  version: FLOW_INSTANCE_OVERRIDE_SCHEMA_VERSION,
  overrides: overrides.map((override) => ({ variableId: override.variableId, value: override.value })),
});

export const parseFlowInstanceOverrides = (value: unknown): FlowInstanceOverrideParseResult => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) {
    return { ok: false, error: 'Flow instance overrides must be an object' };
  }

  const candidate = value as Record<string, unknown>;
  if (candidate.version !== FLOW_INSTANCE_OVERRIDE_SCHEMA_VERSION) {
    return { ok: false, error: `Unsupported Flow instance override version: ${String(candidate.version)}` };
  }
  if (!Array.isArray(candidate.overrides)) {
    return { ok: false, error: 'Flow instance overrides must contain an overrides array' };
  }

  const overrides: FlowInstanceOverride[] = [];
  const seen = new Set<string>();
  for (const entry of candidate.overrides) {
    if (!isOverride(entry)) {
      return { ok: false, error: 'Flow instance override contains an invalid variable identity' };
    }
    if (seen.has(entry.variableId)) {
      return { ok: false, error: `Duplicate Flow instance override: ${entry.variableId}` };
    }
    seen.add(entry.variableId);
    overrides.push({ variableId: entry.variableId, value: entry.value });
  }

  return { ok: true, overrides };
};

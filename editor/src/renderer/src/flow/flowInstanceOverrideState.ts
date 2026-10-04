import type { FlowGraph } from './flowGraphTypes';
import {
  parseFlowInstanceOverrides,
  serializeFlowInstanceOverrides,
  type PersistedFlowInstanceOverrides,
} from './flowInstanceOverridePersistence';
import { reconcileFlowInstanceOverrides, type FlowOverrideDiagnostic } from './flowInstanceOverrides';

export type FlowInstanceOverrideStateResult =
  | {
      ok: true;
      persisted: PersistedFlowInstanceOverrides;
      diagnostics: FlowOverrideDiagnostic[];
    }
  | { ok: false; error: string };

/**
 * Loads persisted per-entity overrides through the versioned persistence boundary,
 * then reconciles them against the current graph schema before they are exposed to
 * Inspector/runtime consumers. Stale or incompatible entries are omitted from the
 * returned persisted state while diagnostics explain why they were dropped.
 */
export const reconcilePersistedFlowInstanceOverrides = (
  graph: FlowGraph,
  persisted: unknown,
): FlowInstanceOverrideStateResult => {
  const parsed = parseFlowInstanceOverrides(persisted);
  if (!parsed.ok) return parsed;

  const reconciled = reconcileFlowInstanceOverrides(graph, parsed.overrides);
  return {
    ok: true,
    persisted: serializeFlowInstanceOverrides(reconciled.overrides),
    diagnostics: reconciled.diagnostics,
  };
};

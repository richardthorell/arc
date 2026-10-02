export type SimulationChangeKind = 'component' | 'entity-created' | 'entity-deleted';

export type SimulationChangeSupport = 'supported' | 'runtime-only' | 'unsupported';

export type SimulationChange = {
  id: string;
  kind: SimulationChangeKind;
  entityId: string;
  componentType?: string;
  support: SimulationChangeSupport;
  summary: string;
};

export type SimulationChangeSelection = {
  applicable: readonly SimulationChange[];
  excluded: readonly SimulationChange[];
};

/**
 * Filters a runtime-world diff before it can become an authoring transaction.
 *
 * Runtime-only and unsupported changes remain visible to the Keep Simulation
 * Changes UI for diagnostics, but callers cannot accidentally include them in
 * the authoring mutation set. The returned order follows the authoritative
 * diff order so a later transaction builder can preserve deterministic intent.
 */
export function selectApplicableSimulationChanges(
  changes: readonly SimulationChange[],
  selectedIds: ReadonlySet<string>,
): SimulationChangeSelection {
  const applicable: SimulationChange[] = [];
  const excluded: SimulationChange[] = [];

  for (const change of changes) {
    if (!selectedIds.has(change.id)) continue;

    if (change.support === 'supported') applicable.push(change);
    else excluded.push(change);
  }

  return { applicable, excluded };
}

/**
 * Returns whether a change may be selected for authoring. Keep this decision
 * separate from presentation so checkboxes, bulk-selection, and transaction
 * construction all share the same safety rule.
 */
export function isSimulationChangeApplicable(change: SimulationChange): boolean {
  return change.support === 'supported';
}

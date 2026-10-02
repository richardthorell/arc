import { describe, expect, it } from 'vitest';

import {
  isSimulationChangeApplicable,
  selectApplicableSimulationChanges,
  type SimulationChange,
} from './simulationChangePolicy';

const changes: readonly SimulationChange[] = [
  {
    id: 'transform',
    kind: 'component',
    entityId: 'player',
    componentType: 'Transform',
    support: 'supported',
    summary: 'Player transform changed',
  },
  {
    id: 'runtime-cache',
    kind: 'component',
    entityId: 'player',
    componentType: 'RuntimeCache',
    support: 'runtime-only',
    summary: 'Runtime cache changed',
  },
  {
    id: 'unsupported-delete',
    kind: 'entity-deleted',
    entityId: 'spawned-at-runtime',
    support: 'unsupported',
    summary: 'Unsupported entity deletion',
  },
  {
    id: 'created',
    kind: 'entity-created',
    entityId: 'new-light',
    support: 'supported',
    summary: 'New light created',
  },
];

describe('simulation change policy', () => {
  it('keeps selected supported changes in authoritative diff order', () => {
    const result = selectApplicableSimulationChanges(changes, new Set(['created', 'transform']));

    expect(result.applicable.map((change) => change.id)).toEqual(['transform', 'created']);
    expect(result.excluded).toEqual([]);
  });

  it('never promotes runtime-only or unsupported state into authoring changes', () => {
    const result = selectApplicableSimulationChanges(
      changes,
      new Set(['transform', 'runtime-cache', 'unsupported-delete']),
    );

    expect(result.applicable.map((change) => change.id)).toEqual(['transform']);
    expect(result.excluded.map((change) => change.id)).toEqual(['runtime-cache', 'unsupported-delete']);
  });

  it('ignores unselected changes', () => {
    const result = selectApplicableSimulationChanges(changes, new Set(['created']));

    expect(result.applicable.map((change) => change.id)).toEqual(['created']);
    expect(result.excluded).toEqual([]);
  });

  it('shares one applicability rule with selection UI callers', () => {
    expect(changes.map(isSimulationChangeApplicable)).toEqual([true, false, false, true]);
  });
});

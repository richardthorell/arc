import { describe, expect, it } from 'vitest';
import type { LivePlayEntity } from '../../../common/livePlayProtocol';
import {
  chooseRuntimeInspection,
  matchesRuntimeEntity,
  runtimeEntityKey,
  runtimeHierarchyRows,
  type RuntimeHierarchyItem,
  type RuntimeInspectionResult,
} from './runtimeInspectionModel';

const authored: LivePlayEntity = { kind: 'authored', entityGuid: '33333333-3333-3333-3333-333333333333' };
const spawned: LivePlayEntity = { kind: 'runtime', runtimeEntityId: 'w1-x1-e9-g2' };

describe('external Live Play inspection identity and presentation', () => {
  it('isolates authored GUIDs from runtime-spawned IDs', () => {
    expect(runtimeEntityKey(authored)).not.toBe(runtimeEntityKey(spawned));
    expect(matchesRuntimeEntity(authored, authored)).toBe(true);
    expect(matchesRuntimeEntity(authored, spawned)).toBe(false);
    expect(matchesRuntimeEntity(null, spawned)).toBe(false);
  });

  it('renders hierarchy depth without trusting circular or missing parents', () => {
    const rows: RuntimeHierarchyItem[] = [
      { entity: authored, parent: null, name: 'Authoring Cube' },
      { entity: spawned, parent: authored, name: 'Runtime Spawn' },
      { entity: { kind: 'runtime', runtimeEntityId: 'child' }, parent: spawned, name: 'Child' },
      {
        entity: { kind: 'runtime', runtimeEntityId: 'loop' },
        parent: { kind: 'runtime', runtimeEntityId: 'loop' },
        name: 'Cycle',
      },
      {
        entity: { kind: 'runtime', runtimeEntityId: 'orphan' },
        parent: { kind: 'runtime', runtimeEntityId: 'missing' },
        name: 'Orphan',
      },
    ];
    expect(runtimeHierarchyRows(rows).map((row) => row.depth)).toEqual([0, 1, 2, 0, 0]);
  });

  it('does not replace the newest inspected revision with a stale result', () => {
    const old: RuntimeInspectionResult = {
      entity: authored,
      component: { typeId: 'arc.transform', schemaVersion: 1 },
      runtimeRevision: 3,
      properties: [{ path: 'position', value: [0, 0, 0] }],
    };
    const latest: RuntimeInspectionResult = { ...old, runtimeRevision: 4 };
    expect(chooseRuntimeInspection(latest, old)).toBe(latest);
    expect(chooseRuntimeInspection(old, latest)).toBe(latest);
  });
});

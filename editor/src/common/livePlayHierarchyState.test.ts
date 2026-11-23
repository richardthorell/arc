import { describe, expect, it } from 'vitest';
import { LIVE_PLAY_PROTOCOL, LIVE_PLAY_VERSION } from './livePlayProtocol';
import {
  applyLivePlayHierarchy,
  emptyLivePlayHierarchy,
  type HierarchyChangeMessage,
  type HierarchySnapshotMessage,
} from './livePlayHierarchyState';

const authored = { kind: 'authored' as const, entityGuid: '11111111-1111-1111-1111-111111111111' };
const spawned = { kind: 'runtime' as const, runtimeEntityId: 'w1-x1-e8-g1' };
const identity = {
  protocol: LIVE_PLAY_PROTOCOL,
  protocolVersion: LIVE_PLAY_VERSION,
  sender: 'runtime' as const,
  projectGuid: '22222222-2222-2222-2222-222222222222',
  buildId: 'dev',
  targetProfile: 'windows',
  sessionId: 'session-1',
  instanceId: 'instance-1',
  messageId: 'hierarchy-1',
  sequence: 1,
} as const;
const item = (n: number) => ({
  entity: { kind: 'runtime' as const, runtimeEntityId: `w1-x1-e${n}-g1` },
  parent: null,
  name: `Entity ${n}`,
});
function snapshot(
  revision: number,
  entities: HierarchySnapshotMessage['payload']['entities'],
  options: Partial<HierarchySnapshotMessage['payload']> = {},
): HierarchySnapshotMessage {
  return {
    ...identity,
    type: 'hierarchy.snapshot',
    payload: { worldRevision: revision, entities, complete: true, ...options },
  };
}
function changed(
  baseRevision: number,
  worldRevision: number,
  changes: HierarchyChangeMessage['payload']['changes'],
): HierarchyChangeMessage {
  return {
    ...identity,
    type: 'hierarchy.changed',
    payload: { baseRevision, worldRevision, changes },
  };
}

describe('L1.5d revisioned external runtime hierarchy', () => {
  it('assembles 260 entities from independent, out-of-order pages without showing a partial tree', () => {
    const scene = Array.from({ length: 260 }, (_, n) => item(n));
    const pages = Array.from({ length: 5 }, (_, n) =>
      snapshot(3, scene.slice(n * 64, n * 64 + 64), { pageIndex: n, pageCount: 5 }),
    );
    let state = applyLivePlayHierarchy(emptyLivePlayHierarchy(), pages[3]);
    expect(state.hierarchy).toBeNull();
    expect(state.pending?.pages.filter(Boolean)).toHaveLength(1);
    for (const i of [1, 0, 4]) state = applyLivePlayHierarchy(state, pages[i]);
    expect(state.hierarchy).toBeNull();
    state = applyLivePlayHierarchy(state, pages[2]);
    expect(state.pending).toBeNull();
    expect(state.hierarchy?.payload.entities).toHaveLength(260);
    expect(state.hierarchy?.payload.worldRevision).toBe(3);
    expect(state.requiresSnapshot).toBe(false);
    expect(state.hierarchy?.payload.entities[0].name).toBe('Entity 0');
    expect(state.hierarchy?.payload.entities[259].name).toBe('Entity 259');
  });

  it('applies spawn, rename, reparent and removal against a verified base', () => {
    const before = snapshot(3, [
      { entity: authored, name: 'Cube', parent: null },
      { entity: spawned, name: 'Old', parent: authored },
    ]);
    let state = applyLivePlayHierarchy(emptyLivePlayHierarchy(), before);
    state = applyLivePlayHierarchy(
      state,
      changed(3, 4, [
        { kind: 'upsert', value: { entity: spawned, name: 'Renamed', parent: null } },
        { kind: 'upsert', value: item(3) },
        { kind: 'removed', entity: authored },
      ]),
    );
    expect(state.hierarchy?.payload.entities).toHaveLength(2);
    expect(state.hierarchy?.payload.entities[0]).toEqual({
      entity: spawned,
      name: 'Renamed',
      parent: null,
    });
    expect(state.hierarchy?.payload.entities[1].name).toBe('Entity 3');
    expect(state.hierarchy?.payload.worldRevision).toBe(4);
  });

  it('rejects a wrong base and retains last verified tree rather than silently applying it', () => {
    const first = applyLivePlayHierarchy(emptyLivePlayHierarchy(), snapshot(3, [item(0)]));
    const invalid = applyLivePlayHierarchy(first, changed(2, 4, [{ kind: 'removed', entity: item(0).entity }]));
    expect(invalid.requiresSnapshot).toBe(true);
    expect(invalid.hierarchy?.payload.entities).toHaveLength(1);
    expect(invalid.hierarchy?.payload.worldRevision).toBe(3);
    const newBase = applyLivePlayHierarchy(invalid, snapshot(5, [item(2)]));
    expect(newBase.requiresSnapshot).toBe(false);
    expect(newBase.hierarchy?.payload.entities[0].name).toBe('Entity 2');
  });

  it('refuses stale, duplicate, mixed-session and malformed incomplete batches', () => {
    let state = applyLivePlayHierarchy(emptyLivePlayHierarchy(), snapshot(3, [item(0)]));
    expect(applyLivePlayHierarchy(state, snapshot(2, [item(1)]))).toBe(state);
    expect(applyLivePlayHierarchy(state, changed(3, 3, []))).toBe(state);

    state = applyLivePlayHierarchy(state, snapshot(4, [item(1)], { pageIndex: 0, pageCount: 2 }));
    expect(state.hierarchy?.payload.worldRevision).toBe(3);
    state = applyLivePlayHierarchy(state, changed(3, 5, [{ kind: 'upsert', value: item(1) }]));
    expect(state.requiresSnapshot).toBe(true);
    expect(state.pending).toBeNull();

    const malicious = snapshot(6, [item(2), item(2)]);
    state = applyLivePlayHierarchy(state, malicious);
    expect(state.requiresSnapshot).toBe(true);

    const wrongInstance = { ...snapshot(7, [item(3)]), instanceId: 'other-instance' };
    state = applyLivePlayHierarchy(state, wrongInstance);
    expect(state.hierarchy).toBeNull();
    expect(state.requiresSnapshot).toBe(true);
  });

  it('tracks oversized but explicitly incomplete hierarchies and prevents applying deltas', () => {
    const initial = snapshot(1, [item(0)], { complete: false });
    const state = applyLivePlayHierarchy(emptyLivePlayHierarchy(), initial);
    expect(state.hierarchy?.payload.complete).toBe(false);
    const delta = applyLivePlayHierarchy(state, changed(1, 2, [{ kind: 'upsert', value: item(2) }]));
    expect(delta.requiresSnapshot).toBe(true);
    expect(delta.hierarchy?.payload.entities).toHaveLength(1);
  });

  it('honors wire schema bounds when parsing paged and delta envelopes', async () => {
    const { livePlayMessageSchema } = await import('./livePlayProtocol');
    expect(livePlayMessageSchema.safeParse(snapshot(1, [item(1)], { pageIndex: 0, pageCount: 2 })).success).toBe(true);
    expect(livePlayMessageSchema.safeParse(changed(1, 2, [{ kind: 'removed', entity: spawned }])).success).toBe(true);
    expect(livePlayMessageSchema.safeParse(snapshot(1, [item(1)], { pageIndex: 70, pageCount: 80 })).success).toBe(
      false,
    );
    const validDelta = changed(1, 2, [{ kind: 'removed', entity: spawned }]);
    const malformed = {
      ...validDelta,
      payload: { ...validDelta.payload, changes: [{ kind: 'removed', entity: spawned, unsafe: true }] },
    };
    expect(livePlayMessageSchema.safeParse(malformed).success).toBe(false);
  });
});

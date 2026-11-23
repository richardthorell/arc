import { describe, expect, it, vi } from 'vitest';
import { LIVE_PLAY_PROTOCOL, LIVE_PLAY_VERSION } from '../common/livePlayProtocol';
import type { HierarchyChangeMessage, HierarchySnapshotMessage } from '../common/livePlayHierarchyState';
import { LivePlayHierarchyRecovery } from './livePlayHierarchyRecovery';

const entity = (id: string) => ({
  entity: { kind: 'runtime' as const, runtimeEntityId: id },
  parent: null,
  name: id,
});
const authority = {
  protocol: LIVE_PLAY_PROTOCOL,
  protocolVersion: LIVE_PLAY_VERSION,
  sender: 'runtime' as const,
  projectGuid: '11111111-1111-1111-1111-111111111111',
  buildId: 'dev',
  targetProfile: 'windows',
  sessionId: 'session-one',
  instanceId: 'runtime-one',
  messageId: 'hierarchy',
  sequence: 2,
} as const;

function page(revision: number, items: ReturnType<typeof entity>[], index = 0, count = 1): HierarchySnapshotMessage {
  return {
    ...authority,
    type: 'hierarchy.snapshot',
    payload: { worldRevision: revision, entities: items, complete: true, pageIndex: index, pageCount: count },
  };
}
function delta(baseRevision: number, worldRevision: number): HierarchyChangeMessage {
  return {
    ...authority,
    type: 'hierarchy.changed',
    payload: { baseRevision, worldRevision, changes: [{ kind: 'upsert', value: entity('new') }] },
  };
}

describe('L1.5e hierarchy recovery coordination', () => {
  it('requests a full snapshot after stalled partial pages, then recovers atomically', () => {
    let clock = 1000;
    const request = vi.fn(() => true);
    const recovery = new LivePlayHierarchyRecovery(request, () => clock);
    recovery.connectionState('starting');
    recovery.connectionState('connected');
    recovery.observe(page(2, [entity('first')], 0, 2));
    expect(recovery.snapshot().hierarchy).toBeNull();
    clock += 1599;
    recovery.tick();
    expect(request).not.toHaveBeenCalled();
    clock++;
    recovery.tick();
    expect(request).toHaveBeenCalledWith('overflow');
    expect(recovery.snapshot()).toMatchObject({ pending: null, requiresSnapshot: true });

    recovery.observe(page(3, [entity('A')], 0, 2));
    expect(recovery.snapshot().hierarchy).toBeNull();
    recovery.observe(page(3, [entity('B')], 1, 2));
    expect(recovery.snapshot().hierarchy?.payload.entities.map((item) => item.name)).toEqual(['A', 'B']);
    expect(recovery.status()).toEqual({ stale: false, attempts: 0, reason: null });
  });

  it('requests bounded retries on an incorrect delta base and does not apply it', () => {
    let clock = 1000;
    const request = vi.fn(() => true);
    const recovery = new LivePlayHierarchyRecovery(request, () => clock);
    recovery.connectionState('connected');
    recovery.observe(page(4, [entity('first')]));
    recovery.observe(delta(3, 5));
    expect(recovery.snapshot().hierarchy?.payload.worldRevision).toBe(4);
    expect(recovery.snapshot().requiresSnapshot).toBe(true);
    expect(request).toHaveBeenCalledWith('gap');
    for (let i = 0; i < 8; ++i) {
      clock += 2401;
      recovery.tick();
    }
    expect(request).toHaveBeenCalledTimes(4);
    expect(recovery.status()).toEqual({ stale: true, reason: 'gap', attempts: 4 });
    recovery.observe(page(6, [entity('fixed')]));
    expect(recovery.snapshot().requiresSnapshot).toBe(false);
    expect(recovery.status().attempts).toBe(0);
  });

  it('invalidates old bases and selects a fresh full snapshot after reconnect', () => {
    const request = vi.fn(() => true);
    const recovery = new LivePlayHierarchyRecovery(request);
    recovery.connectionState('connected');
    recovery.observe(page(5, [entity('before')]));
    recovery.connectionState('disconnected');
    expect(recovery.snapshot().requiresSnapshot).toBe(true);
    expect(recovery.snapshot().hierarchy?.payload.entities[0].name).toBe('before');

    recovery.connectionState('connected');
    expect(request).toHaveBeenCalledWith('reconnected');
    recovery.observe(delta(5, 6));
    expect(recovery.snapshot().requiresSnapshot).toBe(true);
    recovery.observe(page(7, [entity('after')]));
    expect(recovery.snapshot().hierarchy?.payload.entities[0].name).toBe('after');
    expect(recovery.status().stale).toBe(false);
    recovery.connectionState('stopped');
    expect(recovery.snapshot().hierarchy).toBeNull();
  });

  it('does not send requests while disconnected, failed or stopped', () => {
    const request = vi.fn(() => true);
    const recovery = new LivePlayHierarchyRecovery(request);
    recovery.connectionState('connected');
    recovery.observe(page(5, [entity('initial')]));
    recovery.connectionState('disconnected');
    recovery.tick();
    recovery.connectionState('failed');
    recovery.tick();
    recovery.connectionState('stopped');
    recovery.tick();
    expect(request).not.toHaveBeenCalled();
  });
});

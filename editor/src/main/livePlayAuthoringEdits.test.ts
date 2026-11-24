import { randomUUID } from 'node:crypto';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import type { LivePlayMessage } from '../common/livePlayProtocol';

type EditPayload = Extract<LivePlayMessage, { type: 'entity.property.set' }>['payload'];
import { LivePlayAuthoringEdits } from './livePlayAuthoringEdits';
import type { LivePlaySceneBaseline } from './livePlaySceneBaseline';

const sceneGuid = '22222222-2222-2222-2222-222222222222';
const authoredGuid = '33333333-3333-3333-3333-333333333333';
const projectGuid = '11111111-1111-1111-1111-111111111111';
const roots: string[] = [];

type NativeTransform = { position: number[]; rotation: number[]; scale: number[] };

function nativeTransform(position = [0, 0, 0]): NativeTransform {
  return { position, rotation: [0, 0, 0, 1], scale: [1, 1, 1] };
}

function snapshot(position = [0, 0, 0], id = sceneGuid, guid = authoredGuid) {
  return {
    sceneGuid: id,
    entities: [{ guid, transform: nativeTransform(position) }],
  };
}

function fixture() {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'arc-l14a-edits-'));
  roots.push(root);
  const scenePath = path.join(root, 'initial-scene.arcscene');
  fs.writeFileSync(
    scenePath,
    JSON.stringify({
      format: 'arc.scene',
      scene: { id: sceneGuid },
      entities: [{ id: authoredGuid, components: { Transform: nativeTransform() } }],
    }),
  );
  const baseline: LivePlaySceneBaseline = {
    projectGuid,
    buildId: 'dev-build-42',
    targetProfile: 'windows-x64-vulkan',
    sessionId: randomUUID(),
    instanceId: randomUUID(),
    sceneGuid,
    sceneName: 'Level',
    sourceScenePath: '',
    sceneRevision: 11,
    historyRevision: 8,
    entityCount: 1,
    sha256: 'a'.repeat(64),
    byteLength: fs.statSync(scenePath).size,
    directory: root,
    scenePath,
  };
  let count = 0;
  const send = vi.fn((_payload: EditPayload): string | null => `edit-${++count}`);
  const host = { query: vi.fn(async () => ({ succeeded: true, payload: snapshot() })) };
  const tracker = new LivePlayAuthoringEdits(host, send);
  tracker.begin(baseline);
  return { tracker, send, host, baseline, scenePath };
}

function runtimeMessage(baseline: LivePlaySceneBaseline, type: string, payload: unknown): LivePlayMessage {
  return {
    protocol: 'arc.live-play',
    protocolVersion: 1,
    sender: 'runtime',
    projectGuid,
    buildId: baseline.buildId,
    targetProfile: baseline.targetProfile,
    sessionId: baseline.sessionId,
    instanceId: baseline.instanceId,
    sequence: 1,
    messageId: 'msg-1',
    type,
    payload,
  } as LivePlayMessage;
}

beforeEach(() => {
  vi.useFakeTimers();
  vi.setSystemTime(new Date('2026-11-21T20:00:00Z'));
});

afterEach(() => {
  vi.useRealTimers();
  for (const root of roots.splice(0)) fs.rmSync(root, { recursive: true, force: true });
});

describe('Live Play L1.4a authoring transform producer', () => {
  it('holds unsaved edits until the correct native baseline is active then sends committed values by authored GUID', () => {
    const t = fixture();
    t.tracker.observeSnapshot(snapshot([4, 5, 6]));
    expect(t.tracker.snapshot()).toMatchObject({ active: false, pending: 1, revision: 1 });
    expect(t.send).not.toHaveBeenCalled();

    t.tracker.activate('wrong-session');
    expect(t.send).not.toHaveBeenCalled();
    t.tracker.onRuntimeMessage(
      runtimeMessage(t.baseline, 'scene.baseline.status', {
        requestId: 'offer',
        sceneGuid,
        cookedGeneration: 1,
        cookedReleaseSha256: 'f'.repeat(64),
        status: 'active',
        reason: 'none',
      }),
    );
    expect(t.send).toHaveBeenCalledOnce();
    expect(t.send.mock.calls[0][0]).toEqual({
      entity: { kind: 'authored', entityGuid: authoredGuid },
      component: { typeId: 'arc.transform', schemaVersion: 1 },
      property: 'position',
      value: [4, 5, 6],
      authoringRevision: 1,
      expectedRuntimeRevision: 0,
      phase: 'commit',
    });
    t.tracker.onRuntimeMessage(
      runtimeMessage(t.baseline, 'command.ack', {
        requestId: 'edit-1',
        status: 'applied',
        appliedRevision: 9,
      }),
    );
    expect(t.tracker.snapshot().pending).toBe(0);
    t.tracker.observeSnapshot(snapshot([7, 8, 9]));
    expect(t.send.mock.calls.at(-1)?.[0]).toMatchObject({
      value: [7, 8, 9],
      authoringRevision: 2,
      expectedRuntimeRevision: 9,
      phase: 'update',
    });
  });

  it('throttles intermediate viewport drags but reliably commits the final value after quiet time', async () => {
    const t = fixture();
    t.tracker.activate(t.baseline.sessionId);
    t.tracker.observeSnapshot(snapshot([1, 0, 0]));
    expect(t.send.mock.calls.map((call) => call[0].phase)).toEqual(['update']);
    await vi.advanceTimersByTimeAsync(10);
    t.tracker.observeSnapshot(snapshot([2, 0, 0]));
    expect(t.send).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(10);
    t.tracker.observeSnapshot(snapshot([3, 0, 0]));
    expect(t.send).toHaveBeenCalledTimes(1);
    await vi.advanceTimersByTimeAsync(120);
    expect(t.send.mock.calls.at(-1)?.[0]).toMatchObject({
      phase: 'commit',
      value: [3, 0, 0],
      authoringRevision: 3,
    });
    expect(t.send).toHaveBeenCalledTimes(2);
  });

  it('captures native component mutations and Undo/Redo without trusting renderer values', async () => {
    const t = fixture();
    t.host.query.mockResolvedValue({ succeeded: true, payload: snapshot([2, 0, 0]) });
    t.tracker.onHostEvent({ type: 'component.changed', message: 'Entity transform changed' });
    await vi.advanceTimersByTimeAsync(16);
    expect(t.host.query).toHaveBeenCalledWith('scene.hierarchy');
    expect(t.tracker.snapshot().pending).toBe(1);
    t.tracker.activate(t.baseline.sessionId);
    expect(t.send.mock.calls.at(-1)?.[0]).toMatchObject({ phase: 'commit', value: [2, 0, 0] });
    t.host.query.mockResolvedValue({ succeeded: true, payload: snapshot([0, 0, 0]) });
    t.tracker.onHostEvent({ type: 'scene.changed', message: 'Undo completed' });
    await vi.advanceTimersByTimeAsync(16);
    expect(t.send.mock.calls.at(-1)?.[0]).toMatchObject({ phase: 'update', value: [0, 0, 0] });
    t.tracker.onHostEvent({ type: 'entity.selected', message: 'Selection only' });
    await vi.advanceTimersByTimeAsync(16);
    expect(t.host.query).toHaveBeenCalledTimes(2);
  });

  it('keeps final values queued when delivery fails and accepts a later transport retry', () => {
    const t = fixture();
    t.send.mockReturnValueOnce(null);
    t.tracker.observeSnapshot(snapshot([5, 0, 0]));
    t.tracker.activate(t.baseline.sessionId);
    expect(t.tracker.snapshot().pending).toBe(1);
    t.tracker.flushCommits(true);
    expect(t.send).toHaveBeenCalledTimes(2);
    expect(t.send.mock.calls.at(-1)?.[0]).toMatchObject({ phase: 'commit', value: [5, 0, 0] });
  });

  it('retries the final drag commit automatically after a transient transport refusal', async () => {
    const t = fixture();
    t.send.mockReturnValueOnce('preview-update');
    t.send.mockReturnValueOnce(null);
    t.tracker.activate(t.baseline.sessionId);
    t.tracker.observeSnapshot(snapshot([11, 0, 0]));
    await vi.advanceTimersByTimeAsync(120);
    expect(t.send.mock.calls.map((call) => call[0].phase)).toEqual(['update', 'commit']);
    expect(t.tracker.snapshot()).toMatchObject({ pending: 1, active: true });
    await vi.advanceTimersByTimeAsync(240);
    expect(t.send.mock.calls.at(-1)?.[0]).toMatchObject({
      phase: 'commit',
      value: [11, 0, 0],
      authoringRevision: 1,
    });
    expect(t.tracker.snapshot().pending).toBe(1);
  });

  it('retries a lost final ACK, ignores wrong-instance ACKs and reconciles only the latest revision', async () => {
    const t = fixture();
    t.tracker.activate(t.baseline.sessionId);
    t.tracker.observeSnapshot(snapshot([2, 0, 0]));
    await vi.advanceTimersByTimeAsync(120);
    const firstCommitId = t.send.mock.results.at(-1)?.value as string;
    expect(t.send.mock.calls.at(-1)?.[0].phase).toBe('commit');
    t.tracker.onRuntimeMessage({
      ...runtimeMessage(t.baseline, 'command.ack', {
        status: 'applied',
        requestId: firstCommitId,
        appliedRevision: 1,
      }),
      instanceId: 'wrong-runtime',
    });
    expect(t.tracker.snapshot().pending).toBe(1);
    await vi.advanceTimersByTimeAsync(720);
    const retryId = t.send.mock.results.at(-1)?.value as string;
    expect(retryId).not.toBe(firstCommitId);
    expect(t.send.mock.calls.at(-1)?.[0]).toMatchObject({
      phase: 'commit',
      value: [2, 0, 0],
      authoringRevision: 1,
    });

    // The original ACK may arrive after retry has been sent. Both IDs refer
    // to the same immutable revision; either valid ACK settles the final value.
    t.tracker.onRuntimeMessage(
      runtimeMessage(t.baseline, 'command.ack', {
        status: 'applied',
        requestId: firstCommitId,
        appliedRevision: 1,
      }),
    );
    expect(t.tracker.snapshot()).toMatchObject({
      pending: 0,
      lastAppliedRevision: 1,
      applied: 1,
    });
    await vi.advanceTimersByTimeAsync(1000);
    expect(t.tracker.snapshot().pending).toBe(0);
    t.tracker.onRuntimeMessage(
      runtimeMessage(t.baseline, 'command.ack', {
        status: 'applied',
        requestId: retryId,
        appliedRevision: 1,
      }),
    );
    expect(t.tracker.snapshot().applied).toBe(1);
  });

  it('fails closed after bounded unacknowledged commits instead of retrying indefinitely', async () => {
    const t = fixture();
    t.tracker.activate(t.baseline.sessionId);
    t.tracker.observeSnapshot(snapshot([14, 0, 0]));
    await vi.advanceTimersByTimeAsync(4800);
    expect(t.tracker.snapshot()).toMatchObject({
      active: false,
      pending: 0,
      error: expect.stringContaining('not acknowledged'),
    });
    const sent = t.send.mock.calls.length;
    await vi.advanceTimersByTimeAsync(3000);
    expect(t.send.mock.calls).toHaveLength(sent);
  });

  it('does not let stale ACKs clear newer drag changes, and clears all authority on disconnect', async () => {
    const t = fixture();
    t.tracker.activate(t.baseline.sessionId);
    t.tracker.observeSnapshot(snapshot([1, 0, 0]));
    await vi.advanceTimersByTimeAsync(120);
    const oldCommit = t.send.mock.results.at(-1)?.value as string;
    t.tracker.observeSnapshot(snapshot([8, 0, 0]));
    t.tracker.onRuntimeMessage(
      runtimeMessage(t.baseline, 'command.ack', {
        status: 'applied',
        requestId: oldCommit,
        appliedRevision: 2,
      }),
    );
    expect(t.tracker.snapshot().pending).toBe(1);
    t.tracker.disconnect();
    expect(t.tracker.snapshot()).toMatchObject({ active: false, pending: 0 });
    t.tracker.activate(t.baseline.sessionId);
    await vi.advanceTimersByTimeAsync(1000);
    expect(t.tracker.snapshot().pending).toBe(0);
  });

  it('fails closed if an authored entity is structurally removed or created', () => {
    const removed = fixture();
    removed.tracker.activate(removed.baseline.sessionId);
    removed.tracker.observeSnapshot({ sceneGuid, entities: [] });
    expect(removed.tracker.snapshot()).toMatchObject({
      active: false,
      error: expect.stringContaining('removed'),
    });
    const added = fixture();
    added.tracker.activate(added.baseline.sessionId);
    added.tracker.observeSnapshot({
      ...snapshot(),
      entities: [
        ...snapshot().entities,
        { guid: '44444444-4444-4444-4444-444444444444', transform: nativeTransform([2, 0, 0]) },
      ],
    });
    expect(added.tracker.snapshot()).toMatchObject({
      active: false,
      error: expect.stringContaining('requires a new Live Play baseline'),
    });
    expect(removed.send).not.toHaveBeenCalled();
    expect(added.send).not.toHaveBeenCalled();
  });

  it('rejects another scene, non-finite values and unknown runtime-only entities', () => {
    const t = fixture();
    t.tracker.activate(t.baseline.sessionId);
    t.tracker.observeSnapshot({
      sceneGuid,
      entities: [{ guid: authoredGuid, transform: nativeTransform([Number.POSITIVE_INFINITY, 0, 0]) }],
    });
    expect(t.send).not.toHaveBeenCalled();
    t.tracker.observeSnapshot(snapshot([2, 2, 2], sceneGuid, '44444444-4444-4444-4444-444444444444'));
    expect(t.send).not.toHaveBeenCalled();
    t.tracker.observeSnapshot(snapshot([5, 5, 5], '55555555-5555-5555-5555-555555555555'));
    expect(t.tracker.snapshot()).toMatchObject({ active: false, error: expect.stringContaining('scene changed') });
    expect(t.send).not.toHaveBeenCalled();
  });

  it('drops a pending native query on Stop and rejects stale session ACKs', async () => {
    const t = fixture();
    let complete!: (data: { succeeded: boolean; payload: ReturnType<typeof snapshot> }) => void;
    t.host.query.mockImplementation(
      () =>
        new Promise((resolve) => {
          complete = resolve;
        }),
    );
    t.tracker.onHostEvent({ type: 'component.changed' });
    await vi.advanceTimersByTimeAsync(16);
    t.tracker.end();
    complete({ succeeded: true, payload: snapshot([9, 9, 9]) });
    await Promise.resolve();
    expect(t.tracker.snapshot()).toMatchObject({ active: false, sessionId: null, pending: 0 });
    expect(t.send).not.toHaveBeenCalled();
    t.tracker.onRuntimeMessage(
      runtimeMessage(t.baseline, 'command.ack', {
        status: 'applied',
        requestId: 'edit-9',
        appliedRevision: 400,
      }),
    );
    expect(t.tracker.snapshot().revision).toBe(0);
  });
});

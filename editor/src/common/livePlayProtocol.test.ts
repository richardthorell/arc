import { describe, expect, it } from 'vitest';
import {
  LIVE_PLAY_MAX_FRAME_BYTES,
  LIVE_PLAY_PROTOCOL,
  LIVE_PLAY_VERSION,
  livePlayMessageSchema,
  receiveLivePlayFrame,
  type LivePlayReceiveContext,
} from './livePlayProtocol';

const projectGuid = '11111111-1111-1111-1111-111111111111';
const entityGuid = '22222222-2222-2222-2222-222222222222';
const hash = 'a'.repeat(64);

const authority = {
  projectGuid,
  buildId: 'dev-build-42',
  targetProfile: 'windows-x64-vulkan',
  sessionId: 'session-1',
  instanceId: 'runtime-1',
};
const editorContext: LivePlayReceiveContext = {
  authority,
  authenticated: true,
  expectedSchemaHash: hash,
  peerRole: 'editor',
  lastSequence: 0,
  grantedCapabilities: ['session.observe', 'session.control', 'entity.inspect', 'entity.edit'],
};
const runtimeContext: LivePlayReceiveContext = {
  ...editorContext,
  peerRole: 'runtime',
  grantedCapabilities: ['session.observe', 'entity.inspect'],
};

function frame(type: string, payload: unknown, sender: 'editor' | 'runtime', sequence = 1): string {
  return JSON.stringify({
    protocol: LIVE_PLAY_PROTOCOL,
    protocolVersion: LIVE_PLAY_VERSION,
    sender,
    ...authority,
    messageId: `message-${sequence}`,
    sequence,
    type,
    payload,
  });
}

const hello = {
  schemaHash: hash,
  lastReceivedSequence: 0,
  capabilities: ['session.observe', 'entity.inspect', 'entity.edit'],
};
const edit = {
  entity: { kind: 'authored', entityGuid },
  component: { typeId: 'arc.transform', schemaVersion: 1 },
  property: 'position',
  value: [2, 1.5, 0],
  authoringRevision: 12,
  expectedRuntimeRevision: 8,
  phase: 'commit',
};
const inspection = {
  entity: { kind: 'runtime', runtimeEntityId: 'spawned-9' },
  component: { typeId: 'arc.transform', schemaVersion: 1 },
};

describe('Live Play L1 wire contract', () => {
  it('can handshake mock editor and native peers over transport-neutral JSON frames', () => {
    const editorHello = receiveLivePlayFrame(frame('hello', hello, 'editor'), editorContext);
    expect(editorHello).toMatchObject({ accepted: true, nextSequence: 1 });

    const nativeHello = receiveLivePlayFrame(frame('hello', { ...hello, capabilities: ['entity.inspect'] }, 'runtime'), runtimeContext);
    expect(nativeHello).toMatchObject({ accepted: true, nextSequence: 1 });

    const nativeAccepted = receiveLivePlayFrame(
      frame('hello.accepted', { capabilities: ['entity.inspect'], lastReceivedSequence: 1 }, 'runtime', 2),
      { ...runtimeContext, lastSequence: 1 },
    );
    expect(nativeAccepted).toMatchObject({ accepted: true, nextSequence: 2 });
  });

  it('accepts revisioned editor edits, runtime ACKs, and runtime-only entity inspection', () => {
    const command = receiveLivePlayFrame(frame('entity.property.set', edit, 'editor', 2), {
      ...editorContext,
      lastSequence: 1,
    });
    expect(command).toMatchObject({
      accepted: true,
      nextSequence: 2,
      message: { payload: { authoringRevision: 12, expectedRuntimeRevision: 8 } },
    });

    const acknowledged = receiveLivePlayFrame(
      frame('command.ack', { status: 'applied', requestId: 'message-2', appliedRevision: 9 }, 'runtime', 2),
      { ...runtimeContext, lastSequence: 1 },
    );
    expect(acknowledged).toMatchObject({ accepted: true, message: { payload: { appliedRevision: 9 } } });

    const inspect = receiveLivePlayFrame(frame('component.inspect.request', inspection, 'editor', 3), {
      ...editorContext,
      lastSequence: 2,
    });
    expect(inspect.accepted).toBe(true);
    expect(
      receiveLivePlayFrame(
        frame('component.inspect.result', { ...inspection, runtimeRevision: 11, properties: [{ path: 'position', value: [3, 4, 5] }] }, 'runtime', 3),
        { ...runtimeContext, lastSequence: 2 },
      ),
    ).toMatchObject({ accepted: true, nextSequence: 3 });
  });

  it('keeps runtime-only entity IDs scoped to their session/instance', () => {
    const snapshot = {
      worldRevision: 3,
      entities: [{ entity: { kind: 'runtime', runtimeEntityId: 'spawned-1' }, parent: null, name: 'Enemy' }],
      complete: true,
    };
    expect(receiveLivePlayFrame(frame('hierarchy.snapshot', snapshot, 'runtime'), runtimeContext).accepted).toBe(true);
    const otherSession = frame('hierarchy.snapshot', snapshot, 'runtime').replace('"session-1"', '"session-2"');
    expect(receiveLivePlayFrame(otherSession, runtimeContext)).toEqual({
      accepted: false,
      code: 'identity_mismatch',
    });
  });

  it('rejects wrong project, build, profile and instance even with matching session IDs', () => {
    const valid = frame('hello', hello, 'runtime');
    for (const [key, changed] of [
      ['projectGuid', '33333333-3333-3333-3333-333333333333'],
      ['buildId', 'other-build'],
      ['targetProfile', 'android-arm64-vulkan'],
      ['instanceId', 'runtime-2'],
    ]) {
      const mutated = JSON.parse(valid) as Record<string, unknown>;
      mutated[key] = changed;
      expect(receiveLivePlayFrame(JSON.stringify(mutated), runtimeContext)).toEqual({
        accepted: false,
        code: 'identity_mismatch',
      });
    }
  });

  it('rejects unauthenticated, malformed, unknown and oversized messages', () => {
    const valid = frame('hello', hello, 'runtime');
    expect(receiveLivePlayFrame(valid, { ...runtimeContext, authenticated: false })).toEqual({
      accepted: false,
      code: 'unauthenticated',
    });
    expect(receiveLivePlayFrame('{', runtimeContext)).toEqual({ accepted: false, code: 'invalid_frame' });
    expect(receiveLivePlayFrame(frame('process.exec', { command: 'rm -rf /' }, 'runtime'), runtimeContext)).toEqual({
      accepted: false,
      code: 'invalid_frame',
    });
    expect(receiveLivePlayFrame('x'.repeat(LIVE_PLAY_MAX_FRAME_BYTES + 1), runtimeContext)).toEqual({
      accepted: false,
      code: 'frame_too_large',
    });
    const injection = JSON.parse(valid) as Record<string, unknown>;
    injection.authToken = 'must-not-be-logged-or-parsed';
    expect(receiveLivePlayFrame(JSON.stringify(injection), runtimeContext)).toMatchObject({
      accepted: false,
      code: 'invalid_frame',
    });
  });

  it('rejects unsupported protocol/schema, negative or unsafe sequence, and oversized arrays', () => {
    const valid = JSON.parse(frame('hello', hello, 'runtime')) as Record<string, unknown>;
    expect(receiveLivePlayFrame(JSON.stringify({ ...valid, protocolVersion: 2 }), runtimeContext)).toEqual({
      accepted: false,
      code: 'unsupported_version',
    });
    expect(receiveLivePlayFrame(JSON.stringify({ ...valid, payload: { ...hello, schemaHash: 'b'.repeat(64) } }), runtimeContext)).toEqual({
      accepted: false,
      code: 'schema_mismatch',
    });
    expect(livePlayMessageSchema.safeParse({ ...valid, sequence: -1 }).success).toBe(false);
    expect(livePlayMessageSchema.safeParse({ ...valid, sequence: Number.MAX_SAFE_INTEGER + 1 }).success).toBe(false);
    expect(
      livePlayMessageSchema.safeParse({
        ...valid,
        payload: { ...hello, capabilities: Array.from({ length: 100 }, () => 'entity.inspect') },
      }).success,
    ).toBe(false);
  });

  it('rejects handshake attempts to elevate negotiated capabilities', () => {
    expect(
      receiveLivePlayFrame(
        frame('hello.accepted', { capabilities: ['entity.edit'], lastReceivedSequence: 0 }, 'runtime'),
        runtimeContext,
      ),
    ).toEqual({ accepted: false, code: 'permission_denied' });
  });

  it('rejects remote control without permission and messages sent in the wrong direction', () => {
    expect(
      receiveLivePlayFrame(frame('entity.property.set', edit, 'editor'), {
        ...editorContext,
        grantedCapabilities: ['entity.inspect'],
      }),
    ).toEqual({ accepted: false, code: 'permission_denied' });
    expect(receiveLivePlayFrame(frame('entity.property.set', edit, 'runtime'), runtimeContext)).toEqual({
      accepted: false,
      code: 'wrong_direction',
    });
    expect(receiveLivePlayFrame(frame('hierarchy.snapshot', { worldRevision: 0, entities: [], complete: true }, 'editor'), editorContext)).toEqual({
      accepted: false,
      code: 'wrong_direction',
    });
  });

  it('deduplicates replayed commands and supports reconnect resync with sender sequence continuity', () => {
    const first = receiveLivePlayFrame(frame('entity.property.set', edit, 'editor', 30), {
      ...editorContext,
      lastSequence: 29,
    });
    expect(first).toMatchObject({ accepted: true, nextSequence: 30 });
    expect(
      receiveLivePlayFrame(frame('entity.property.set', edit, 'editor', 30), {
        ...editorContext,
        lastSequence: 30,
      }),
    ).toEqual({ accepted: false, code: 'replayed_sequence' });

    const reconnect = receiveLivePlayFrame(
      frame('session.resync.request', { lastReceivedSequence: 24, reason: 'reconnected' }, 'editor', 31),
      { ...editorContext, lastSequence: 30 },
    );
    expect(reconnect).toMatchObject({ accepted: true, nextSequence: 31 });
    expect(
      receiveLivePlayFrame(frame('hierarchy.snapshot', { worldRevision: 40, entities: [], complete: true }, 'runtime', 35), {
        ...runtimeContext,
        lastSequence: 24,
      }).accepted,
    ).toBe(true);
  });

  it('exchanges content generation status without embedding any cooked blob bytes', () => {
    expect(
      receiveLivePlayFrame(
        frame('content.generation', { generation: 4, status: 'active' }, 'runtime'),
        runtimeContext,
      ).accepted,
    ).toBe(true);
    expect(
      livePlayMessageSchema.safeParse({
        ...JSON.parse(frame('content.generation', { generation: 4, status: 'active' }, 'runtime')),
        payload: { generation: 4, status: 'active', blobs: ['arbitrary bytes'] },
      }).success,
    ).toBe(false);
  });
});

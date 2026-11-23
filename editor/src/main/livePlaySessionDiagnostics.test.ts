import { describe, expect, it, vi } from 'vitest';
import { LIVE_PLAY_PROTOCOL, LIVE_PLAY_VERSION, type LivePlayMessage } from '../common/livePlayProtocol';
import type { WindowsLivePlaySnapshot } from '../common/livePlaySessionTypes';
import type { LivePlayControlStatus } from './livePlayControlServer';
import { LivePlaySessionDiagnostics, safeText } from './livePlaySessionDiagnostics';

const authority = {
  projectGuid: '11111111-1111-1111-1111-111111111111',
  buildId: 'fixture-build',
  targetProfile: 'windows-default',
  sessionId: 'live-session-1',
  instanceId: 'game-instance-1',
};

const connected: WindowsLivePlaySnapshot = {
  state: 'connected',
  sessionId: authority.sessionId,
  instanceId: authority.instanceId,
  pid: 1234,
  lastReceivedSequence: 1,
  error: null,
};

const transport: LivePlayControlStatus = {
  state: 'connected',
  port: 9876,
  sessionId: authority.sessionId,
  instanceId: authority.instanceId,
  lastReceivedSequence: 1,
  lastError: null,
  capabilities: ['session.observe', 'entity.inspect', 'entity.edit'],
  lastHeartbeatAtMs: null,
};

function message(type: LivePlayMessage['type'], payload: unknown, sequence = 2): LivePlayMessage {
  return {
    protocol: LIVE_PLAY_PROTOCOL,
    protocolVersion: LIVE_PLAY_VERSION,
    sender: 'runtime',
    ...authority,
    messageId: `message-${sequence}`,
    sequence,
    type,
    payload,
  } as LivePlayMessage;
}

describe('L1.5c authenticated session diagnostics', () => {
  it('observes capabilities, runtime state, heartbeat and content generation from a single game instance', () => {
    let now = 1000;
    const publish = vi.fn();
    const diagnostics = new LivePlaySessionDiagnostics(publish, () => now);
    diagnostics.begin(authority.sessionId, authority.instanceId);
    diagnostics.observeStatus(connected, transport);
    expect(diagnostics.snapshot()).toMatchObject({
      connection: 'connected',
      negotiatedCapabilities: transport.capabilities,
      lastHeartbeatAtMs: null,
    });

    now = 1100;
    diagnostics.observeMessage(message('heartbeat', { lastReceivedSequence: 1 }, 2));
    diagnostics.observeMessage(message('session.state', { state: 'running', tick: 42, revision: 9 }, 3));
    diagnostics.observeMessage(message('content.generation', { status: 'active', generation: 7 }, 4));
    expect(diagnostics.snapshot()).toMatchObject({
      lastHeartbeatAtMs: 1100,
      runtimeState: 'running',
      runtimeTick: 42,
      runtimeRevision: 9,
      contentGeneration: 7,
      contentStatus: 'active',
      lastReceivedSequence: 4,
    });
    expect(publish).toHaveBeenCalled();
    const snapshot = diagnostics.snapshot();
    snapshot.events.length = 0;
    expect(diagnostics.snapshot().events.length).toBeGreaterThan(0);
  });

  it('measures ACK and inspection response round trips, never claiming visible-frame latency', () => {
    let now = 100;
    const diagnostics = new LivePlaySessionDiagnostics(
      () => {},
      () => now,
    );
    diagnostics.begin(authority.sessionId, authority.instanceId);
    diagnostics.observeStatus(connected, transport);
    diagnostics.requestSent('edit-one', 'edit');
    now = 137;
    diagnostics.observeMessage(
      message('command.ack', { status: 'applied', requestId: 'edit-one', appliedRevision: 9 }),
    );
    diagnostics.requestSent('inspect-one', 'inspect');
    now = 171;
    diagnostics.observeMessage(
      message('component.inspect.result', {
        entity: { kind: 'authored', entityGuid: authority.projectGuid },
        component: { typeId: 'arc.transform', schemaVersion: 1 },
        runtimeRevision: 3,
        properties: [],
        requestId: 'inspect-one',
      }),
    );
    diagnostics.requestSent('edit-two', 'edit');
    now = 193;
    diagnostics.observeMessage(
      message('command.ack', {
        status: 'rejected',
        requestId: 'edit-two',
        error: { code: 'stale_revision', message: 'stale edit' },
      }),
    );
    expect(diagnostics.snapshot()).toMatchObject({
      appliedCommands: 1,
      rejectedCommands: 1,
      lastAckLatencyMs: 22,
      lastInspectionLatencyMs: 34,
      lastAckAtMs: 193,
    });
    expect(diagnostics.snapshot().events.map((event) => event.message)).toContain('Command rejected: stale_revision');
  });

  it('bounds session logs, drops stdout floods and redacts recognizable credentials', () => {
    let now = 1000;
    const diagnostics = new LivePlaySessionDiagnostics(() => {}, () => now);
    diagnostics.begin(authority.sessionId, authority.instanceId);
    diagnostics.observeStatus(connected, transport);
    diagnostics.observeMessage(
      message('log.record', {
        level: 'warning',
        text: 'Auth Bearer secret-jwt token=abc123 https://host.test/assets?secret=leak',
      }),
    );
    for (let i = 0; i < 200; i++)
      diagnostics.observeMessage(message('log.record', { level: 'debug', text: 'x'.repeat(500) }));
    for (let i = 0; i < 32; i++) diagnostics.observeProcessOutput('stdout', Buffer.from('line\n'));
    const snapshot = diagnostics.snapshot();
    expect(snapshot.events).toHaveLength(160);
    expect(snapshot.droppedLogLines).toBe(12);
    expect(snapshot.events.every((entry) => entry.message.length <= 320)).toBe(true);
    expect(snapshot.events.some((event) => event.message.includes('abc123'))).toBe(false);
    expect(safeText('Bearer secret-jwt token=abc123 https://host.test/assets?secret=leak')).not.toMatch(
      /secret-jwt|abc123|secret=leak/u,
    );
    now += 1001;
    diagnostics.observeProcessOutput('stderr', Buffer.from('another line\n'));
    expect(diagnostics.snapshot().events.at(-1)?.level).toBe('warning');
  });

  it('ignores late messages from older instances and resets counters for the next launch', () => {
    const diagnostics = new LivePlaySessionDiagnostics();
    diagnostics.begin(authority.sessionId, authority.instanceId);
    diagnostics.observeStatus(connected, transport);
    diagnostics.requestSent('edit-one', 'edit');
    diagnostics.begin('next-session', 'next-instance');
    diagnostics.observeMessage(
      message('command.ack', { status: 'applied', requestId: 'edit-one', appliedRevision: 4 }),
    );
    diagnostics.observeStatus({ ...connected, state: 'disconnected', error: 'old connection' }, transport);
    expect(diagnostics.snapshot()).toMatchObject({
      sessionId: 'next-session',
      instanceId: 'next-instance',
      connection: 'starting',
      appliedCommands: 0,
      lastAckLatencyMs: null,
      negotiatedCapabilities: [],
    });
  });

  it('does not let a dead renderer callback interrupt observation', () => {
    const diagnostics = new LivePlaySessionDiagnostics(() => {
      throw new Error('renderer destroyed');
    });
    expect(() => diagnostics.begin(authority.sessionId, authority.instanceId)).not.toThrow();
    expect(() => diagnostics.observeMessage(message('log.record', { level: 'error', text: 'failed' }))).not.toThrow();
  });
});

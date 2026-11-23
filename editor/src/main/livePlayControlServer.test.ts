import { createConnection, type Socket } from 'node:net';
import { afterEach, describe, expect, it } from 'vitest';
import {
  LIVE_PLAY_PROTOCOL,
  LIVE_PLAY_VERSION,
  type LivePlayCapability,
  type LivePlayMessage,
} from '../common/livePlayProtocol';
import {
  LivePlayControlServer,
  type LivePlayControlOptions,
  type LivePlayControlStatus,
} from './livePlayControlServer';

const authority = {
  projectGuid: '11111111-1111-1111-1111-111111111111',
  buildId: 'dev-build-42',
  targetProfile: 'windows-x64-vulkan',
  sessionId: 'session-1',
  instanceId: 'runtime-1',
};
const schemaHash = 'a'.repeat(64);
const cap: LivePlayCapability[] = ['session.observe', 'entity.inspect', 'content.observe'];

function frame(type: string, payload: unknown, sequence = 1): Buffer {
  return Buffer.from(
    JSON.stringify({
      protocol: LIVE_PLAY_PROTOCOL,
      protocolVersion: LIVE_PLAY_VERSION,
      sender: 'runtime',
      ...authority,
      messageId: `native-message-${sequence}`,
      sequence,
      type,
      payload,
    }),
    'utf8',
  );
}

function packet(data: Buffer): Buffer {
  const head = Buffer.alloc(4);
  head.writeUInt32BE(data.length);
  return Buffer.concat([head, data]);
}

type Pending = { resolve: (frame: Buffer) => void; reject: (error: Error) => void; timer: NodeJS.Timeout };
class TestPeer {
  private buffer = Buffer.alloc(0);
  private readonly frames: Buffer[] = [];
  private readonly pending: Pending[] = [];
  readonly closed: Promise<void>;
  constructor(readonly socket: Socket) {
    this.closed = new Promise((resolve) => {
      socket.once('close', () => {
        for (const wait of this.pending.splice(0)) {
          clearTimeout(wait.timer);
          wait.reject(new Error('Socket closed'));
        }
        resolve();
      });
    });
    socket.on('error', () => {
      // Rejections and socket shutdown are asserted using the close promise.
    });
    socket.on('data', (data: Buffer) => {
      this.buffer = Buffer.concat([this.buffer, data]);
      while (this.buffer.length >= 4) {
        const size = this.buffer.readUInt32BE(0);
        if (this.buffer.length < size + 4) break;
        const body = this.buffer.subarray(4, 4 + size);
        this.buffer = this.buffer.subarray(4 + size);
        const waiter = this.pending.shift();
        if (!waiter) this.frames.push(body);
        else {
          clearTimeout(waiter.timer);
          waiter.resolve(body);
        }
      }
    });
  }
  write(data: Buffer): void {
    this.socket.write(packet(data));
  }
  next(): Promise<Buffer> {
    const item = this.frames.shift();
    if (item) return Promise.resolve(item);
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        const i = this.pending.findIndex((entry) => entry.timer === timer);
        if (i >= 0) this.pending.splice(i, 1);
        reject(new Error('Timed out waiting for Live Play frame'));
      }, 2000);
      this.pending.push({ resolve, reject, timer });
    });
  }
}

const servers: LivePlayControlServer[] = [];
const sockets: Socket[] = [];
afterEach(async () => {
  for (const socket of sockets.splice(0)) socket.destroy();
  for (const server of servers.splice(0)) await server.stop();
});

async function start(overrides: Partial<LivePlayControlOptions> = {}): Promise<LivePlayControlServer> {
  const server = new LivePlayControlServer({
    authority,
    schemaHash,
    grantedCapabilities: cap,
    ...overrides,
  });
  servers.push(server);
  await server.start();
  return server;
}

async function connect(port: number): Promise<TestPeer> {
  const socket = createConnection({ host: '127.0.0.1', port });
  sockets.push(socket);
  const peer = new TestPeer(socket);
  await new Promise<void>((resolve, reject) => {
    socket.once('connect', resolve);
    socket.once('error', reject);
  });
  return peer;
}

async function handshake(server: LivePlayControlServer, sequence = 1): Promise<TestPeer> {
  const handoff = server.launchHandoff();
  const peer = await connect(handoff.port);
  peer.write(handoff.secret);
  peer.write(
    frame(
      'hello',
      { schemaHash, lastReceivedSequence: 0, capabilities: ['session.observe', 'entity.inspect', 'content.observe'] },
      sequence,
    ),
  );
  const welcome = JSON.parse((await peer.next()).toString('utf8')) as LivePlayMessage;
  expect(welcome).toMatchObject({
    type: 'hello.accepted',
    sender: 'editor',
    sessionId: authority.sessionId,
    instanceId: authority.instanceId,
    payload: { capabilities: ['session.observe', 'entity.inspect', 'content.observe'] },
  });
  return peer;
}

describe('Live Play L1.2a editor control endpoint', () => {
  it('reports only negotiated capabilities and authenticated runtime heartbeats', async () => {
    const server = await start();
    const peer = await handshake(server);
    expect(server.status().capabilities).toEqual(['session.observe', 'entity.inspect', 'content.observe']);
    expect(server.status().lastHeartbeatAtMs).toBeNull();

    peer.write(frame('heartbeat', { lastReceivedSequence: 1 }, 2));
    await expect.poll(() => server.status().lastHeartbeatAtMs).not.toBeNull();
    expect(server.status().lastHeartbeatAtMs).toEqual(expect.any(Number));

    peer.socket.destroy();
    await peer.closed;
    expect(server.status().capabilities).toEqual([]);
  });

  it('binds loopback, performs token authentication and L1.1 handshake, and relays only validated runtime state', async () => {
    const received: LivePlayMessage[] = [];
    const statuses: LivePlayControlStatus[] = [];
    const server = await start({
      onMessage: (message) => received.push(message),
      onStatus: (status) => statuses.push(status),
    });
    expect(server.status()).toMatchObject({ state: 'waiting', port: expect.any(Number) });
    const peer = await handshake(server);

    expect(server.status().state).toBe('connected');
    peer.write(frame('session.state', { state: 'running', tick: 7, revision: 3 }, 2));
    peer.write(frame('content.generation', { generation: 1, status: 'active' }, 3));
    await expect.poll(() => received.map((entry) => entry.type)).toEqual(['session.state', 'content.generation']);
    expect(server.status().lastReceivedSequence).toBe(3);
    expect(statuses.some((entry) => entry.state === 'connected')).toBe(true);
    expect(server.sendHeartbeat()).toBe(true);
    const heartbeat = JSON.parse((await peer.next()).toString('utf8')) as LivePlayMessage;
    expect(heartbeat).toMatchObject({ type: 'heartbeat', payload: { lastReceivedSequence: 3 } });
    expect(JSON.stringify(server.status())).not.toContain(server.launchHandoff().secret.toString('hex'));
  });

  it('drops unauthenticated peers before processing a forged native message', async () => {
    const received: LivePlayMessage[] = [];
    const server = await start({ onMessage: (message) => received.push(message) });
    const handoff = server.launchHandoff();

    const peer = await connect(handoff.port);
    peer.write(Buffer.alloc(32, 0));
    peer.write(frame('session.state', { state: 'running', tick: 1, revision: 1 }));
    await peer.closed;
    expect(received).toEqual([]);
    expect(server.status().lastReceivedSequence).toBe(0);
    expect(server.status().lastError).toBe('authentication_failed');

    const valid = await handshake(server);
    expect(valid.socket.destroyed).toBe(false);
    expect(server.status().state).toBe('connected');
  });

  it('rejects wrong identity, unknown commands and replayed sequences', async () => {
    const server = await start();
    const first = await handshake(server);
    first.write(
      Buffer.from(
        frame('session.state', { state: 'running', tick: 1, revision: 1 }, 2)
          .toString('utf8')
          .replace('"runtime-1"', '"runtime-evil"'),
        'utf8',
      ),
    );
    await first.closed;
    expect(server.status().lastError).toBe('identity_mismatch');

    const second = await handshake(server, 2);
    second.write(frame('session.state', { state: 'running', tick: 1, revision: 1 }, 2));
    await second.closed;
    expect(server.status().lastError).toBe('replayed_sequence');

    const third = await handshake(server, 3);
    third.write(frame('process.exec', { command: 'not-allowed' }, 4));
    await third.closed;
    expect(server.status().lastError).toBe('invalid_frame');
  });

  it('rejects mismatched schema, capability escalation and oversized frames', async () => {
    const server = await start({ grantedCapabilities: ['session.observe'] });
    const handoff = server.launchHandoff();
    const badSchema = await connect(handoff.port);
    badSchema.write(handoff.secret);
    badSchema.write(frame('hello', { schemaHash: 'f'.repeat(64), lastReceivedSequence: 0, capabilities: [] }));
    await badSchema.closed;
    expect(server.status().lastError).toBe('schema_mismatch');

    const peer = await connect(handoff.port);
    peer.write(handoff.secret);
    peer.write(
      frame('hello', { schemaHash, lastReceivedSequence: 0, capabilities: ['session.observe', 'entity.edit'] }, 2),
    );
    const welcome = JSON.parse((await peer.next()).toString('utf8')) as LivePlayMessage;
    expect(welcome).toMatchObject({ type: 'hello.accepted', payload: { capabilities: ['session.observe'] } });
    peer.write(frame('hierarchy.snapshot', { worldRevision: 1, entities: [], complete: true }, 3));
    await peer.closed;
    expect(server.status().lastError).toBe('permission_denied');

    const oversized = await connect(handoff.port);
    oversized.write(handoff.secret);
    const header = Buffer.alloc(4);
    header.writeUInt32BE(65537);
    oversized.socket.write(header);
    await oversized.closed;
    expect(server.status().lastError).toBe('invalid_frame');
  });

  it('continues sequence numbers across reconnect but rejects old sessions', async () => {
    const received: LivePlayMessage[] = [];
    const server = await start({ onMessage: (message) => received.push(message) });
    const peer = await handshake(server);
    peer.write(frame('session.state', { state: 'running', tick: 3, revision: 2 }, 2));
    await expect.poll(() => server.status().lastReceivedSequence).toBe(2);
    peer.socket.end();
    await peer.closed;
    expect(server.status().state).toBe('disconnected');

    const resumed = await handshake(server, 3);
    resumed.write(frame('session.state', { state: 'paused', tick: 4, revision: 3 }, 4));
    await expect.poll(() => server.status().lastReceivedSequence).toBe(4);
    expect(received).toHaveLength(2);
    resumed.socket.destroy();
    await resumed.closed;
    expect(server.status().state).toBe('disconnected');

    const newRun = await start({ authority: { ...authority, sessionId: 'other-session' } });
    const wrong = await connect(newRun.launchHandoff().port);
    wrong.write(newRun.launchHandoff().secret);
    wrong.write(frame('hello', { schemaHash, lastReceivedSequence: 0, capabilities: ['session.observe'] }));
    await wrong.closed;
    expect(newRun.status().lastError).toBe('identity_mismatch');
  });

  it('does not let unauthenticated or competing peers invalidate an already-connected session', async () => {
    const server = await start();
    const active = await handshake(server);
    const handoff = server.launchHandoff();

    const rejected = await connect(handoff.port);
    rejected.write(Buffer.alloc(32, 0));
    await rejected.closed;
    expect(server.status().state).toBe('connected');
    expect(server.status().lastError).toBeNull();

    const competing = await connect(handoff.port);
    competing.write(handoff.secret);
    competing.write(frame('hello', { schemaHash, lastReceivedSequence: 0, capabilities: ['session.observe'] }, 2));
    await competing.closed;
    expect(server.status().state).toBe('connected');
    active.write(frame('session.state', { state: 'running', tick: 1, revision: 1 }, 2));
    await expect.poll(() => server.status().lastReceivedSequence).toBe(2);
    expect(server.sendHeartbeat()).toBe(true);
  });

  it('enforces the capabilities actually negotiated with the native client', async () => {
    const server = await start();
    const handoff = server.launchHandoff();
    const peer = await connect(handoff.port);
    peer.write(handoff.secret);
    peer.write(frame('hello', { schemaHash, lastReceivedSequence: 0, capabilities: ['session.observe'] }));
    const welcome = JSON.parse((await peer.next()).toString('utf8')) as LivePlayMessage;
    expect(welcome).toMatchObject({ payload: { capabilities: ['session.observe'] } });
    peer.write(frame('hierarchy.snapshot', { worldRevision: 1, entities: [], complete: true }, 2));
    await peer.closed;
    expect(server.status().lastError).toBe('permission_denied');
  });

  it('offers a cooked baseline only over the authenticated negotiated channel and checks its ACK identity', async () => {
    const received: LivePlayMessage[] = [];
    const server = await start({ onMessage: (message) => received.push(message) });
    const offer = {
      sceneGuid: '22222222-2222-2222-2222-222222222222',
      authoringRevision: 9,
      authoringSha256: 'b'.repeat(64),
      cookedGeneration: 3,
      cookedReleaseSha256: 'c'.repeat(64),
    };
    expect(server.sendBaselineOffer(offer)).toBe(false);
    const peer = await handshake(server);
    expect(server.sendBaselineOffer(offer)).toBe(true);
    expect(server.sendBaselineOffer(offer)).toBe(false);
    const offered = JSON.parse((await peer.next()).toString('utf8')) as LivePlayMessage;
    expect(offered).toMatchObject({ type: 'scene.baseline.offer', payload: offer });
    const rejected = {
      requestId: offered.messageId,
      sceneGuid: offer.sceneGuid,
      cookedGeneration: offer.cookedGeneration,
      cookedReleaseSha256: offer.cookedReleaseSha256,
      status: 'rejected',
      reason: 'unavailable',
    };
    peer.write(frame('scene.baseline.status', rejected, 2));
    await expect.poll(() => received.map((m) => m.type)).toEqual(['scene.baseline.status']);
    expect(server.sendBaselineOffer(offer)).toBe(true);
    const second = JSON.parse((await peer.next()).toString('utf8')) as LivePlayMessage;
    const wrongRelease = {
      requestId: second.messageId,
      sceneGuid: offer.sceneGuid,
      cookedGeneration: offer.cookedGeneration,
      cookedReleaseSha256: 'd'.repeat(64),
      status: 'active',
      reason: 'none',
    };
    peer.write(frame('scene.baseline.status', wrongRelease, 3));
    await peer.closed;
    expect(server.status().lastError).toBe('baseline_identity_mismatch');
    expect(received).toHaveLength(1);
  });

  it('does not offer a baseline to a native session without content.observe', async () => {
    const server = await start();
    const handoff = server.launchHandoff();
    const peer = await connect(handoff.port);
    peer.write(handoff.secret);
    peer.write(frame('hello', { schemaHash, lastReceivedSequence: 0, capabilities: ['session.observe'] }));
    await peer.next();
    const offer = {
      sceneGuid: '22222222-2222-2222-2222-222222222222',
      authoringRevision: 1,
      authoringSha256: schemaHash,
      cookedGeneration: 1,
      cookedReleaseSha256: schemaHash,
    };
    expect(server.sendBaselineOffer(offer)).toBe(false);
  });

  it('tears down connections and invalidates the launch secret on Stop', async () => {
    const server = await start();
    const handoff = server.launchHandoff();
    const peer = await handshake(server);
    await server.stop();
    await peer.closed;
    expect(server.status()).toMatchObject({ state: 'stopped', port: null });
    expect(() => server.launchHandoff()).toThrow(/not listening/);
    expect(server.sendHeartbeat()).toBe(false);
    await server.start();
    expect(server.launchHandoff().secret.equals(handoff.secret)).toBe(false);
    expect(server.status().state).toBe('waiting');
  });
});

import { randomBytes, randomUUID, timingSafeEqual } from 'node:crypto';
import { createServer, type Server, type Socket } from 'node:net';
import {
  LIVE_PLAY_MAX_FRAME_BYTES,
  LIVE_PLAY_PROTOCOL,
  LIVE_PLAY_VERSION,
  livePlayMessageSchema,
  receiveLivePlayFrame,
  type LivePlayAuthority,
  type LivePlayCapability,
  type LivePlayMessage,
} from '../common/livePlayProtocol';

/**
 * L1.2a: editor-main-process control endpoint. Native Development clients are
 * added by L1.2b; #1140 owns launching the process and secure secret handoff.
 * Never start this service for in-editor Play or from renderer IPC.
 */
export type LivePlayControlOptions = {
  authority: LivePlayAuthority;
  schemaHash: string;
  grantedCapabilities: readonly LivePlayCapability[];
  onMessage?: (message: LivePlayMessage) => void;
  onStatus?: (status: LivePlayControlStatus) => void;
  /** Test/diagnostic override; production always gets an ephemeral port. */
  port?: number;
  handshakeTimeoutMs?: number;
  idleTimeoutMs?: number;
};

export type LivePlayControlStatus = {
  state: 'stopped' | 'waiting' | 'connected' | 'disconnected' | 'error';
  port: number | null;
  sessionId: string;
  instanceId: string;
  lastReceivedSequence: number;
  lastError: string | null;
};

type Peer = {
  socket: Socket;
  buffer: Buffer;
  authenticated: boolean;
  ready: boolean;
  capabilities: readonly LivePlayCapability[];
};

const maxUnauthenticatedPeers = 4;
const maxBufferedBytes = 4 * (LIVE_PLAY_MAX_FRAME_BYTES + 4);
const authSecretBytes = 32;

function framePayload(payload: Buffer): Buffer {
  if (payload.length > LIVE_PLAY_MAX_FRAME_BYTES) throw new Error('Live Play frame exceeds protocol limit');
  const header = Buffer.allocUnsafe(4);
  header.writeUInt32BE(payload.length, 0);
  return Buffer.concat([header, payload]);
}

/**
 * The launcher transports the returned secret through trusted OS IPC or
 * inherited pipes/handles. It must never pass it via argv, URL, disk or logs.
 */
export class LivePlayControlServer {
  private server: Server | null = null;
  private readonly peers = new Set<Peer>();
  private activePeer: Peer | null = null;
  private secret: Buffer = randomBytes(authSecretBytes);
  private state: LivePlayControlStatus['state'] = 'stopped';
  private port: number | null = null;
  private lastError: string | null = null;
  private lastReceivedSequence = 0;
  private outgoingSequence = 0;

  constructor(private readonly options: LivePlayControlOptions) {}

  status(): LivePlayControlStatus {
    return {
      state: this.state,
      port: this.port,
      sessionId: this.options.authority.sessionId,
      instanceId: this.options.authority.instanceId,
      lastReceivedSequence: this.lastReceivedSequence,
      lastError: this.lastError,
    };
  }

  private notify(): void {
    try {
      this.options.onStatus?.(this.status());
    } catch {
      // Renderer/event observer errors must not affect transport authority.
    }
  }

  /** Available exclusively to trusted main-process launch orchestration. */
  launchHandoff(): { port: number; secret: Buffer } {
    if (!this.server || this.port === null || this.state === 'stopped')
      throw new Error('Live Play session is not listening');
    return { port: this.port, secret: Buffer.from(this.secret) };
  }

  async start(): Promise<void> {
    if (this.server) throw new Error('Live Play control server is already started');
    // Validate trusted launch authority before opening even a loopback port.
    const sample = this.message('heartbeat', { lastReceivedSequence: 0 });
    if (!livePlayMessageSchema.safeParse(sample).success) throw new Error('Invalid Live Play launch authority');
    if (
      !/^[a-f\d]{64}$/iu.test(this.options.schemaHash) ||
      new Set(this.options.grantedCapabilities).size !== this.options.grantedCapabilities.length
    )
      throw new Error('Invalid Live Play schema or capability grant');

    this.secret.fill(0);
    this.secret = randomBytes(authSecretBytes);
    this.lastReceivedSequence = 0;
    this.outgoingSequence = 0;
    this.lastError = null;
    const server = createServer((socket) => this.accept(socket));
    this.server = server;
    try {
      await new Promise<void>((resolve, reject) => {
        server.once('error', reject);
        server.listen(this.options.port ?? 0, '127.0.0.1', () => {
          server.removeListener('error', reject);
          resolve();
        });
      });
      const address = server.address();
      if (!address || typeof address === 'string') throw new Error('No Live Play TCP listener port');
      server.on('error', () => {
        // An established listener can fail independently of any connected peer.
        this.lastError = 'listener_failed';
        this.state = 'error';
        this.notify();
        for (const peer of this.peers) peer.socket.destroy();
      });
      this.port = address.port;
      this.state = 'waiting';
      this.notify();
    } catch (error) {
      this.server = null;
      try {
        server.close();
      } catch {
        // An unsuccessful listen may never have opened a server handle.
      }
      this.secret.fill(0);
      this.port = null;
      this.state = 'error';
      this.lastError = error instanceof Error ? error.message : 'Listener failed';
      this.notify();
      throw error;
    }
  }

  async stop(): Promise<void> {
    const server = this.server;
    this.server = null;
    this.port = null;
    this.state = 'stopped';
    this.lastError = null;
    for (const peer of this.peers) peer.socket.destroy();
    this.peers.clear();
    this.activePeer = null;
    this.lastReceivedSequence = 0;
    this.outgoingSequence = 0;
    this.secret.fill(0);
    this.notify();
    if (server) await new Promise<void>((resolve) => server.close(() => resolve()));
  }

  /** Bounded outbound status heartbeat; gameplay/ECS commands are not wired by L1.2a. */
  sendHeartbeat(): boolean {
    if (!this.activePeer?.ready) return false;
    const payload = this.message('heartbeat', { lastReceivedSequence: this.lastReceivedSequence });
    return this.send(this.activePeer, payload);
  }

  private message(type: 'heartbeat' | 'hello.accepted', payload: unknown): LivePlayMessage {
    const item = {
      protocol: LIVE_PLAY_PROTOCOL,
      protocolVersion: LIVE_PLAY_VERSION,
      sender: 'editor' as const,
      ...this.options.authority,
      messageId: randomUUID(),
      sequence: this.outgoingSequence + 1,
      type,
      payload,
    };
    return livePlayMessageSchema.parse(item);
  }

  private send(peer: Peer, item: LivePlayMessage): boolean {
    const data = framePayload(Buffer.from(JSON.stringify(item), 'utf8'));
    if (peer.socket.destroyed || peer.socket.writableLength + data.length > maxBufferedBytes) {
      this.reject(peer, 'output_backpressure');
      return false;
    }
    if (!peer.socket.write(data)) {
      this.reject(peer, 'output_backpressure');
      return false;
    }
    this.outgoingSequence = item.sequence;
    return true;
  }

  private accept(socket: Socket): void {
    if (this.peers.size >= maxUnauthenticatedPeers || socket.remoteAddress !== '127.0.0.1') {
      socket.destroy();
      return;
    }
    socket.setNoDelay(true);
    socket.setTimeout(this.options.handshakeTimeoutMs ?? 5000);
    const peer: Peer = { socket, buffer: Buffer.alloc(0), authenticated: false, ready: false, capabilities: [] };
    this.peers.add(peer);
    socket.on('data', (chunk: Buffer) => this.read(peer, chunk));
    socket.on('timeout', () => this.reject(peer, 'timeout'));
    socket.on('error', () => this.reject(peer, 'connection_failed'));
    socket.on('close', () => {
      this.peers.delete(peer);
      if (this.activePeer === peer) {
        this.activePeer = null;
        if (this.server) {
          this.state = 'disconnected';
          this.notify();
        }
      }
    });
  }

  private read(peer: Peer, chunk: Buffer): void {
    if (peer.socket.destroyed) return;
    if (peer.buffer.length + chunk.length > maxBufferedBytes) {
      this.reject(peer, 'input_backpressure');
      return;
    }
    peer.buffer = Buffer.concat([peer.buffer, chunk]);
    while (peer.buffer.length >= 4 && !peer.socket.destroyed) {
      const size = peer.buffer.readUInt32BE(0);
      if (size === 0 || size > LIVE_PLAY_MAX_FRAME_BYTES) {
        this.reject(peer, 'invalid_frame');
        return;
      }
      if (peer.buffer.length < size + 4) break;
      const payload = peer.buffer.subarray(4, size + 4);
      peer.buffer = peer.buffer.subarray(size + 4);
      if (!peer.authenticated) {
        if (size !== authSecretBytes || !timingSafeEqual(payload, this.secret)) {
          this.reject(peer, 'authentication_failed');
          return;
        }
        peer.authenticated = true;
      } else {
        this.receiveMessage(peer, payload.toString('utf8'));
      }
    }
  }

  private receiveMessage(peer: Peer, frame: string): void {
    const result = receiveLivePlayFrame(frame, {
      authority: this.options.authority,
      expectedSchemaHash: this.options.schemaHash,
      authenticated: peer.authenticated,
      peerRole: 'runtime',
      lastSequence: this.lastReceivedSequence,
      grantedCapabilities: this.options.grantedCapabilities,
    });
    if (!result.accepted) {
      this.reject(peer, result.code);
      return;
    }
    const item = result.message;
    if (!peer.ready) {
      if (item.type !== 'hello' || this.activePeer) {
        this.reject(peer, 'unexpected_handshake');
        return;
      }
      if (item.payload.lastReceivedSequence > this.outgoingSequence) {
        this.reject(peer, 'invalid_resume_sequence');
        return;
      }
      const capabilities = item.payload.capabilities.filter((cap) => this.options.grantedCapabilities.includes(cap));
      const accepted = this.message('hello.accepted', {
        capabilities,
        lastReceivedSequence: result.nextSequence,
      });
      if (!this.send(peer, accepted)) return;
      this.lastReceivedSequence = result.nextSequence;
      peer.capabilities = capabilities;
      peer.ready = true;
      this.activePeer = peer;
      peer.socket.setTimeout(this.options.idleTimeoutMs ?? 30000);
      this.state = 'connected';
      this.lastError = null;
      this.notify();
      return;
    }
    if (item.type === 'hello' || item.type === 'hello.accepted' || item.type === 'hello.rejected') {
      this.reject(peer, 'unexpected_handshake');
      return;
    }
    const required =
      item.type === 'hierarchy.snapshot' ||
      item.type === 'hierarchy.changed' ||
      item.type === 'component.inspect.result'
        ? 'entity.inspect'
        : item.type === 'content.generation'
          ? 'content.observe'
          : 'session.observe';
    if (!peer.capabilities.includes(required)) {
      this.reject(peer, 'permission_denied');
      return;
    }
    this.lastReceivedSequence = result.nextSequence;
    this.notify();
    try {
      this.options.onMessage?.(item);
    } catch {
      // Never dispatch untrusted frames a second time after an observer failure.
    }
  }

  private reject(peer: Peer, reason: string): void {
    if (peer.socket.destroyed) return;
    // Only safe, enumerated reasons. Secrets and JSON payloads never enter status/logs.
    // A rejected competing or unauthenticated peer must not disrupt an
    // already connected, authoritative Development runtime session.
    if (!this.activePeer || this.activePeer === peer) {
      this.lastError = reason;
      this.state = 'error';
      this.notify();
    }
    peer.socket.destroy();
  }
}

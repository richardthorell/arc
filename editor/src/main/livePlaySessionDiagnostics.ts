import { StringDecoder } from 'node:string_decoder';
import type { LivePlayMessage } from '../common/livePlayProtocol';
import type { LivePlayDiagnosticsSnapshot, LivePlayDiagnosticEntry } from '../common/livePlayDiagnosticsTypes';
import type { WindowsLivePlaySnapshot } from '../common/livePlaySessionTypes';
import type { LivePlayControlStatus } from './livePlayControlServer';

const maxEvents = 160;
const maxPending = 128;
const maxTextLength = 320;
const maxOutputBuffer = 4096;
const maxOutputLinesPerSecond = 20;

type Stream = 'stdout' | 'stderr';
type PendingRequest = { kind: 'edit' | 'inspect'; sentAtMs: number };
type OutputState = { decoder: StringDecoder; tail: string };
const blank = (): LivePlayDiagnosticsSnapshot => ({
  sessionId: null,
  instanceId: null,
  connection: 'stopped',
  negotiatedCapabilities: [],
  lastReceivedSequence: 0,
  lastHeartbeatAtMs: null,
  lastRuntimeMessageAtMs: null,
  runtimeState: null,
  runtimeTick: null,
  runtimeRevision: null,
  contentGeneration: null,
  contentStatus: null,
  lastAckAtMs: null,
  lastAckLatencyMs: null,
  lastInspectionLatencyMs: null,
  appliedCommands: 0,
  rejectedCommands: 0,
  droppedLogLines: 0,
  events: [],
});

/**
 * Main-process-only telemetry. It never exposes raw frames, secrets, process
 * environment, URLs, launch arguments or arbitrary ECS data to the renderer.
 */
export class LivePlaySessionDiagnostics {
  private data = blank();
  private nextId = 0;
  private readonly pending = new Map<string, PendingRequest>();
  private readonly output: Record<Stream, OutputState> = {
    stdout: { decoder: new StringDecoder('utf8'), tail: '' },
    stderr: { decoder: new StringDecoder('utf8'), tail: '' },
  };
  private outputWindowMs = 0;
  private outputLines = 0;

  constructor(
    private readonly publish: (value: LivePlayDiagnosticsSnapshot) => void = () => {},
    private readonly now: () => number = Date.now,
  ) {}

  snapshot(): LivePlayDiagnosticsSnapshot {
    return {
      ...this.data,
      negotiatedCapabilities: [...this.data.negotiatedCapabilities],
      events: this.data.events.map((event) => ({ ...event })),
    };
  }

  begin(sessionId: string, instanceId: string): void {
    this.data = { ...blank(), sessionId, instanceId, connection: 'starting' };
    this.pending.clear();
    for (const output of Object.values(this.output)) {
      output.decoder = new StringDecoder('utf8');
      output.tail = '';
    }
    this.outputLines = 0;
    this.outputWindowMs = 0;
    this.add('connection', 'info', 'Development game launch starting');
  }

  observeStatus(snapshot: WindowsLivePlaySnapshot, transport?: LivePlayControlStatus): void {
    if (snapshot.sessionId && this.data.sessionId !== snapshot.sessionId) return;
    if (snapshot.instanceId && this.data.instanceId !== snapshot.instanceId) return;
    const connectionChanged = this.data.connection !== snapshot.state;
    const capabilities = transport?.capabilities ?? this.data.negotiatedCapabilities;
    const capabilitiesChanged = capabilities.join(',') !== this.data.negotiatedCapabilities.join(',');
    this.data.connection = snapshot.state;
    this.data.lastReceivedSequence = snapshot.lastReceivedSequence;
    this.data.lastHeartbeatAtMs = transport?.lastHeartbeatAtMs ?? this.data.lastHeartbeatAtMs;
    this.data.negotiatedCapabilities = [...capabilities];
    if (snapshot.state !== 'connected') this.pending.clear();
    if (connectionChanged || capabilitiesChanged) {
      if (connectionChanged)
        this.add(
          'connection',
          snapshot.state === 'failed' || snapshot.state === 'disconnected' ? 'warning' : 'info',
          `Connection: ${snapshot.state}${snapshot.error ? ` (${safeText(snapshot.error)})` : ''}`,
        );
      else this.emit();
    }
  }

  requestSent(requestId: string | null, kind: PendingRequest['kind']): void {
    if (!requestId || this.data.connection !== 'connected') return;
    if (this.pending.size >= maxPending) {
      const first = this.pending.keys().next().value;
      if (first) this.pending.delete(first);
    }
    this.pending.set(requestId, { kind, sentAtMs: this.now() });
  }

  observeMessage(message: LivePlayMessage): void {
    if (message.sessionId !== this.data.sessionId || message.instanceId !== this.data.instanceId) return;
    const now = this.now();
    this.data.lastRuntimeMessageAtMs = now;
    this.data.lastReceivedSequence = message.sequence;

    switch (message.type) {
      case 'heartbeat':
        this.data.lastHeartbeatAtMs = now;
        this.emit();
        return;
      case 'session.state':
        if (message.payload.state !== this.data.runtimeState)
          this.add('runtime', 'info', `Game state: ${message.payload.state}`);
        this.data.runtimeState = message.payload.state;
        this.data.runtimeTick = message.payload.tick;
        this.data.runtimeRevision = message.payload.revision;
        this.emit();
        return;
      case 'log.record':
        this.add('log', message.payload.level, message.payload.text);
        return;
      case 'command.ack': {
        const pending = this.pending.get(message.payload.requestId);
        if (pending) this.pending.delete(message.payload.requestId);
        const elapsed = pending ? Math.max(0, now - pending.sentAtMs) : null;
        this.data.lastAckAtMs = now;
        if (pending?.kind === 'edit') this.data.lastAckLatencyMs = elapsed;
        if (pending?.kind === 'inspect') this.data.lastInspectionLatencyMs = elapsed;
        if (message.payload.status === 'applied') {
          this.data.appliedCommands++;
          if (pending?.kind !== 'inspect')
            this.add('command', 'info', `Command applied at revision ${message.payload.appliedRevision}`);
          else this.emit();
        } else {
          this.data.rejectedCommands++;
          this.add('command', 'warning', `Command rejected: ${message.payload.error.code}`);
        }
        return;
      }
      case 'component.inspect.result': {
        if (message.payload.requestId) {
          const pending = this.pending.get(message.payload.requestId);
          if (pending?.kind === 'inspect') {
            this.data.lastInspectionLatencyMs = Math.max(0, now - pending.sentAtMs);
            this.pending.delete(message.payload.requestId);
            this.emit();
          }
        }
        return;
      }
      case 'content.generation':
        this.data.contentGeneration = message.payload.generation;
        this.data.contentStatus = message.payload.status;
        this.add(
          'content',
          'info',
          `Cooked content ${message.payload.status} (generation ${message.payload.generation})`,
        );
        return;
      case 'scene.baseline.status':
        this.add(
          'content',
          message.payload.status === 'rejected' ? 'warning' : 'info',
          `Scene baseline ${message.payload.status}: ${message.payload.reason}`,
        );
        return;
      default:
        return;
    }
  }

  /** Child-process output is bounded line-by-line and never stored in raw form. */
  observeProcessOutput(channel: Stream, bytes: Buffer): void {
    if (this.data.connection === 'stopped') return;
    const state = this.output[channel];
    state.tail += state.decoder.write(bytes);
    if (state.tail.length > maxOutputBuffer) {
      this.outputLine(channel, state.tail.slice(0, maxTextLength) + ' …[truncated]');
      state.tail = '';
    }
    const lines = state.tail.split(/\r?\n/u);
    state.tail = lines.pop() ?? '';
    for (const line of lines) if (line.trim()) this.outputLine(channel, line);
  }

  private outputLine(channel: Stream, line: string): void {
    const now = this.now();
    if (now - this.outputWindowMs >= 1000) {
      this.outputWindowMs = now;
      this.outputLines = 0;
    }
    if (++this.outputLines > maxOutputLinesPerSecond) {
      this.data.droppedLogLines++;
      return;
    }
    this.add('log', channel === 'stderr' ? 'warning' : 'info', `${channel}: ${line}`);
  }

  private add(kind: LivePlayDiagnosticEntry['kind'], level: LivePlayDiagnosticEntry['level'], message: string): void {
    this.data.events.push({
      id: ++this.nextId,
      timestampMs: this.now(),
      kind,
      level,
      message: safeText(message),
    });
    if (this.data.events.length > maxEvents) this.data.events.splice(0, this.data.events.length - maxEvents);
    this.emit();
  }

  private emit(): void {
    try {
      this.publish(this.snapshot());
    } catch {
      // A renderer reload cannot disrupt the Development runtime process.
    }
  }
}

export function safeText(input: string): string {
  return input
    .replace(/\bBearer\s+[^\s]+/giu, 'Bearer [redacted]')
    .replace(/\b((?:auth[_-]?)?token|secret|password|api[_-]?key)\s*[:=]\s*[^\s&,;]+/giu, '$1=[redacted]')
    .replace(/(https?:\/\/[^\s?#]+)\?[^\s#]+/giu, '$1?[redacted]')
    .replace(/[\x00-\x1f\x7f]/gu, ' ')
    .slice(0, maxTextLength);
}

import fs from 'node:fs';
import { livePlayMessageSchema, type LivePlayMessage } from '../common/livePlayProtocol';
import type { LivePlaySceneBaseline } from './livePlaySceneBaseline';

type EditPayload = Extract<LivePlayMessage, { type: 'entity.property.set' }>['payload'];
type Property = 'position' | 'rotation' | 'scale';
type Transform = Record<Property, readonly number[]>;
type SceneEntity = { guid?: unknown; transform?: unknown };
type SceneSnapshot = { sceneGuid?: unknown; entities?: unknown };
type HostQuery = { query(type: string): Promise<{ succeeded: boolean; payload?: unknown }> };
type HostEvent = { type: string; message?: string };
type SendEdit = (payload: EditPayload) => string | null;
type Pending = { guid: string; property: Property; value: number[]; revision: number; changedAt: number; lastUpdate: number };
type Outstanding = { key: string; revision: number; phase: 'update' | 'commit' };

const guidPattern = /^(?:[a-f\d]{32}|[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12})$/iu;
const properties: readonly Property[] = ['position', 'rotation', 'scale'];
const lengths: Record<Property, number> = { position: 3, rotation: 4, scale: 3 };
const updateIntervalMs = 33;
const commitQuietMs = 120;
const maxEntities = 4096;
const maxPending = 256;
const maxOutstanding = 512;

function transform(value: unknown): Transform | null {
  if (!value || typeof value !== 'object') return null;
  const source = value as Record<string, unknown>;
  const result = {} as Record<Property, readonly number[]>;
  for (const property of properties) {
    const candidate = source[property];
    if (
      !Array.isArray(candidate) ||
      candidate.length !== lengths[property] ||
      !candidate.every((item) => typeof item === 'number' && Number.isFinite(item))
    )
      return null;
    result[property] = [...candidate];
  }
  return result;
}

function extractScene(document: unknown, nested: boolean): Map<string, Transform> {
  if (!document || typeof document !== 'object') throw new Error('Live Play scene snapshot is invalid');
  const entities = (document as SceneSnapshot).entities;
  if (!Array.isArray(entities) || entities.length > maxEntities)
    throw new Error('Live Play scene exceeds the bounded transform tracking limit');
  const found = new Map<string, Transform>();
  for (const entity of entities as SceneEntity[]) {
    if (!entity || typeof entity !== 'object') continue;
    const key = nested ? (entity as { id?: unknown }).id : entity.guid;
    if (typeof key !== 'string' || !guidPattern.test(key)) continue;
    const values = nested
      ? transform((entity as { components?: { Transform?: unknown } }).components?.Transform)
      : transform(entity.transform);
    if (values) found.set(key.toLowerCase(), values);
  }
  return found;
}

function differs(left: readonly number[], right: readonly number[]): boolean {
  return left.some((value, index) => value !== right[index]);
}

/**
 * L1.4a editor-side authoring change capture. This observes native Host
 * mutations (including Undo/Redo), not renderer-controlled commands. It
 * never writes scene files or changes Authoring/Play ECS state.
 *
 * The captured L1.3 baseline is the comparison anchor. Until a native game
 * explicitly ACKs that baseline active, pending values stay private and no
 * entity edits go over the wire. L1.4b supplies runtime tick-safe apply.
 */
export class LivePlayAuthoringEdits {
  private baseline: LivePlaySceneBaseline | null = null;
  private observed = new Map<string, Transform>();
  private readonly pending = new Map<string, Pending>();
  private readonly outstanding = new Map<string, Outstanding>();
  private revision = 0;
  private runtimeRevision = 0;
  private active = false;
  private generation = 0;
  private refreshTimer: ReturnType<typeof setTimeout> | null = null;
  private commitTimer: ReturnType<typeof setTimeout> | null = null;
  private refreshing = false;
  private again = false;
  private lastError: string | null = null;

  constructor(
    private readonly host: HostQuery,
    private readonly send: SendEdit,
  ) {}

  snapshot(): { sessionId: string | null; active: boolean; pending: number; revision: number; error: string | null } {
    return {
      sessionId: this.baseline?.sessionId ?? null,
      active: this.active,
      pending: this.pending.size,
      revision: this.revision,
      error: this.lastError,
    };
  }

  begin(baseline: LivePlaySceneBaseline): void {
    this.end();
    const document = JSON.parse(fs.readFileSync(baseline.scenePath, 'utf8')) as {
      format?: unknown;
      scene?: { id?: unknown };
    };
    if (document.format !== 'arc.scene' || document.scene?.id?.toString().toLowerCase() !== baseline.sceneGuid.toLowerCase())
      throw new Error('Live Play authoring edits need a matching captured scene');
    const entities = extractScene(document, true);
    this.baseline = baseline;
    this.observed = entities;
    this.generation++;
  }

  end(): void {
    this.generation++;
    if (this.refreshTimer) clearTimeout(this.refreshTimer);
    if (this.commitTimer) clearTimeout(this.commitTimer);
    this.refreshTimer = null;
    this.commitTimer = null;
    this.baseline = null;
    this.active = false;
    this.refreshing = false;
    this.again = false;
    this.observed.clear();
    this.pending.clear();
    this.outstanding.clear();
    this.revision = 0;
    this.runtimeRevision = 0;
    this.lastError = null;
  }

  onHostEvent(event: HostEvent): void {
    if (!this.baseline) return;
    if (event.type === 'scene.changed') {
      if (!/undo completed|redo completed|edit transaction cancelled/i.test(event.message ?? '')) return;
    } else if (event.type !== 'component.changed') {
      return;
    }
    this.scheduleRefresh();
  }

  private scheduleRefresh(): void {
    if (this.refreshing) {
      this.again = true;
      return;
    }
    if (this.refreshTimer) return;
    this.refreshTimer = setTimeout(() => {
      this.refreshTimer = null;
      void this.refresh();
    }, 16);
  }

  private async refresh(): Promise<void> {
    const epoch = this.generation;
    this.refreshing = true;
    try {
      const result = await this.host.query('scene.hierarchy');
      if (epoch !== this.generation || !this.baseline) return;
      if (!result.succeeded || !result.payload || typeof result.payload !== 'object')
        throw new Error('Native authoring scene query failed');
      const snapshot = result.payload as SceneSnapshot;
      this.observeSnapshot(snapshot);
    } catch {
      if (epoch === this.generation) this.lastError = 'Cannot observe the current native authoring scene';
    } finally {
      if (epoch === this.generation) {
        this.refreshing = false;
        if (this.again) {
          this.again = false;
          this.scheduleRefresh();
        }
      }
    }
  }

  /** Accepts only full native Authoring World snapshots for this scene. */
  observeSnapshot(snapshot: SceneSnapshot): void {
    if (!this.baseline) return;
    if (typeof snapshot.sceneGuid !== 'string' || snapshot.sceneGuid.toLowerCase() !== this.baseline.sceneGuid.toLowerCase()) {
      this.lastError = 'Authoring scene changed during the attached Play session';
      this.active = false;
      this.pending.clear();
      this.outstanding.clear();
      return;
    }
    let next: Map<string, Transform>;
    try {
      next = extractScene(snapshot, false);
    } catch {
      this.lastError = 'Authoring transform snapshot exceeds safe limits';
      this.active = false;
      this.pending.clear();
      return;
    }
    const now = Date.now();
    for (const [guid, before] of this.observed) {
      const after = next.get(guid);
      if (!after) {
        for (const property of properties) this.pending.delete(`${guid}:${property}`);
        continue; // No create/delete transport until L1.4c.
      }
      for (const property of properties) {
        if (!differs(before[property], after[property])) continue;
        if (this.revision >= Number.MAX_SAFE_INTEGER || this.pending.size >= maxPending) {
          this.lastError = 'Live Play transform queue exceeded safe limits';
          this.active = false;
          this.pending.clear();
          return;
        }
        const key = `${guid}:${property}`;
        const previous = this.pending.get(key);
        this.pending.set(key, {
          guid,
          property,
          value: [...after[property]],
          revision: ++this.revision,
          changedAt: now,
          lastUpdate: previous?.lastUpdate ?? Number.NEGATIVE_INFINITY,
        });
      }
    }
    this.observed = next;
    if (!this.active || this.pending.size === 0) return;
    this.flushUpdates(now);
    this.scheduleCommit();
  }

  /** Called only after the native runtime explicitly confirms a verified baseline. */
  activate(sessionId: string): void {
    if (!this.baseline || this.baseline.sessionId !== sessionId || this.lastError) return;
    this.active = true;
    // Changes made while cooking are sent as final values, not replayed drags.
    this.flushCommits(true);
  }

  disconnect(): void {
    this.active = false;
    if (this.commitTimer) clearTimeout(this.commitTimer);
    this.commitTimer = null;
    // Never replay a command from the old native game instance on reconnect.
    this.outstanding.clear();
  }

  onRuntimeMessage(message: LivePlayMessage): void {
    if (!this.baseline || message.sessionId !== this.baseline.sessionId) return;
    if (message.type === 'scene.baseline.status' && message.payload.status === 'active') {
      this.activate(message.sessionId);
    } else if (message.type === 'command.ack') {
      const request = this.outstanding.get(message.payload.requestId);
      if (!request) return;
      this.outstanding.delete(message.payload.requestId);
      if (message.payload.status === 'applied') {
        this.runtimeRevision = Math.max(this.runtimeRevision, message.payload.appliedRevision);
        if (request.phase === 'commit' && this.pending.get(request.key)?.revision === request.revision)
          this.pending.delete(request.key);
      } else {
        // L1.4b will implement resynchronization after explicit rejection.
        // Never silently retry stale or incompatible commands.
        this.lastError = `Runtime rejected authoring edit: ${message.payload.error.code}`;
        this.active = false;
        this.pending.clear();
      }
    }
  }

  private sendPending(item: Pending, phase: 'update' | 'commit'): boolean {
    if (!this.active || this.outstanding.size >= maxOutstanding) return false;
    const payload: EditPayload = {
      entity: { kind: 'authored', entityGuid: item.guid },
      component: { typeId: 'arc.transform', schemaVersion: 1 },
      property: item.property,
      value: item.value,
      authoringRevision: item.revision,
      expectedRuntimeRevision: this.runtimeRevision,
      phase,
    };
    if (!livePlayMessageSchema.safeParse({
      protocol: 'arc.live-play',
      protocolVersion: 1,
      sender: 'editor',
      projectGuid: this.baseline?.projectGuid,
      buildId: this.baseline?.buildId,
      targetProfile: this.baseline?.targetProfile,
      sessionId: this.baseline?.sessionId,
      instanceId: this.baseline?.instanceId,
      messageId: 'preview-validation',
      sequence: 1,
      type: 'entity.property.set',
      payload,
    }).success) {
      this.lastError = 'Invalid authoring transform command';
      this.active = false;
      return false;
    }
    const id = this.send(payload);
    if (!id) return false;
    this.outstanding.set(id, { key: `${item.guid}:${item.property}`, revision: item.revision, phase });
    return true;
  }

  private flushUpdates(now: number): void {
    for (const item of this.pending.values()) {
      if (now - item.lastUpdate < updateIntervalMs) continue;
      if (this.sendPending(item, 'update')) item.lastUpdate = now;
    }
  }

  private scheduleCommit(): void {
    if (this.commitTimer) clearTimeout(this.commitTimer);
    this.commitTimer = setTimeout(() => {
      this.commitTimer = null;
      this.flushCommits(false);
    }, commitQuietMs);
  }

  /** Flushes the latest final value even if prior drag updates were dropped. */
  flushCommits(force: boolean): void {
    if (!this.active) return;
    const now = Date.now();
    for (const item of this.pending.values()) {
      if (!force && now - item.changedAt < commitQuietMs) continue;
      if (!this.sendPending(item, 'commit')) continue; // Keep the final value.
    }
  }
}

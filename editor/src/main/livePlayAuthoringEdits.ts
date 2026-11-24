import fs from 'node:fs';
import { livePlayMessageSchema, type LivePlayMessage } from '../common/livePlayProtocol';
import type { LivePlaySceneBaseline } from './livePlaySceneBaseline';

type EditPayload = Extract<LivePlayMessage, { type: 'entity.property.set' }>['payload'];
type TransformProperty = 'position' | 'rotation' | 'scale';
type LightProperty = 'intensity' | 'color';
type Property = TransformProperty | LightProperty;
type Transform = Record<TransformProperty, readonly number[]>;
type LightComponent = 'arc.light.directional' | 'arc.light.point' | 'arc.light.spot' | 'arc.light.area';
type Light = { component: LightComponent; intensity: readonly number[]; color: readonly number[] };
type SceneEntity = {
  guid?: unknown;
  id?: unknown;
  transform?: unknown;
  light?: unknown;
  components?: Record<string, unknown>;
};
type SceneSnapshot = { sceneGuid?: unknown; entities?: unknown };
type HostQuery = { query(type: string): Promise<{ succeeded: boolean; payload?: unknown }> };
type HostEvent = { type: string; message?: string };
type SendEdit = (payload: EditPayload) => string | null;
type Pending = {
  guid: string;
  component: string;
  property: Property;
  value: number[];
  revision: number;
  changedAt: number;
  lastUpdate: number;
  commitAttempts: number;
  lastCommitAttempt: number;
  commitInFlight: boolean;
};
type Outstanding = { key: string; revision: number; phase: 'update' | 'commit'; sentAt: number };

const guidPattern = /^(?:[a-f\d]{32}|[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12})$/iu;
const properties: readonly TransformProperty[] = ['position', 'rotation', 'scale'];
const lightProperties: readonly LightProperty[] = ['intensity', 'color'];
const lengths: Record<TransformProperty, number> = { position: 3, rotation: 4, scale: 3 };
const updateIntervalMs = 33;
const commitQuietMs = 120;
const commitRetryMs = 240;
const commitAckTimeoutMs = 650;
const maxCommitAttempts = 6;
const maxEntities = 4096;
const maxPending = 256;
const maxOutstanding = 512;

function transform(value: unknown): Transform | null {
  if (!value || typeof value !== 'object') return null;
  const source = value as Record<string, unknown>;
  const result = {} as Record<TransformProperty, readonly number[]>;
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

const lightKinds: Record<string, LightComponent> = {
  DirectionalLight: 'arc.light.directional',
  PointLight: 'arc.light.point',
  SpotLight: 'arc.light.spot',
  AreaLight: 'arc.light.area',
  directional: 'arc.light.directional',
  point: 'arc.light.point',
  spot: 'arc.light.spot',
  disk: 'arc.light.area',
  rectangle: 'arc.light.area',
};

function parseLight(value: unknown, component: LightComponent): Light | null {
  if (!value || typeof value !== 'object') return null;
  const record = value as Record<string, unknown>;
  const valid = (number: unknown) =>
    typeof number === 'number' && Number.isFinite(number) && number >= 0 && number <= 1e6;
  if (
    !valid(record.intensity) ||
    !Array.isArray(record.color) ||
    record.color.length !== 3 ||
    !record.color.every(valid)
  )
    return null;
  return { component, intensity: [record.intensity as number], color: [...record.color] };
}

/** Read only the native host's allowlisted light values, keyed by stable authored GUID. */
function extractLights(document: unknown, nested: boolean): Map<string, Light> {
  if (!document || typeof document !== 'object') throw new Error('Invalid light scene snapshot');
  const entities = (document as SceneSnapshot).entities;
  if (!Array.isArray(entities) || entities.length > maxEntities) throw new Error('Unbounded light scene snapshot');
  const result = new Map<string, Light>();
  for (const entity of entities as SceneEntity[]) {
    if (!entity || typeof entity !== 'object') continue;
    const guid = nested ? entity.id : entity.guid;
    if (typeof guid !== 'string' || !guidPattern.test(guid)) continue;
    let kind: LightComponent | undefined;
    let source: unknown;
    if (nested) {
      for (const [name, component] of Object.entries(entity.components ?? {})) {
        if (!lightKinds[name] || !name.endsWith('Light')) continue;
        if (kind) throw new Error('Ambiguous authored light component');
        kind = lightKinds[name];
        source = component;
      }
    } else if (entity.light && typeof entity.light === 'object') {
      const raw = entity.light as { kind?: unknown };
      if (typeof raw.kind !== 'string' || !lightKinds[raw.kind]) throw new Error('Unknown native light component');
      kind = lightKinds[raw.kind];
      source = entity.light;
    }
    if (!kind) continue;
    const found = parseLight(source, kind);
    if (!found || result.has(guid.toLowerCase())) throw new Error('Invalid or ambiguous native light properties');
    result.set(guid.toLowerCase(), found);
  }
  return result;
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
  private observedLights = new Map<string, Light>();
  private readonly pending = new Map<string, Pending>();
  private readonly outstanding = new Map<string, Outstanding>();
  private revision = 0;
  private runtimeRevision = 0;
  private lastAckLatencyMs: number | null = null;
  private lastAppliedRevision = 0;
  private applied = 0;
  private rejected = 0;
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

  snapshot(): {
    sessionId: string | null;
    active: boolean;
    pending: number;
    revision: number;
    error: string | null;
    lastAppliedRevision: number;
    lastAckLatencyMs: number | null;
    applied: number;
    rejected: number;
  } {
    return {
      sessionId: this.baseline?.sessionId ?? null,
      active: this.active,
      pending: this.pending.size,
      revision: this.revision,
      error: this.lastError,
      lastAppliedRevision: this.lastAppliedRevision,
      lastAckLatencyMs: this.lastAckLatencyMs,
      applied: this.applied,
      rejected: this.rejected,
    };
  }

  begin(baseline: LivePlaySceneBaseline): void {
    this.end();
    const document = JSON.parse(fs.readFileSync(baseline.scenePath, 'utf8')) as {
      format?: unknown;
      scene?: { id?: unknown };
    };
    if (
      document.format !== 'arc.scene' ||
      document.scene?.id?.toString().toLowerCase() !== baseline.sceneGuid.toLowerCase()
    )
      throw new Error('Live Play authoring edits need a matching captured scene');
    const entities = extractScene(document, true);
    const lights = extractLights(document, true);
    this.baseline = baseline;
    this.observed = entities;
    this.observedLights = lights;
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
    this.observedLights.clear();
    this.pending.clear();
    this.outstanding.clear();
    this.revision = 0;
    this.runtimeRevision = 0;
    this.lastAppliedRevision = 0;
    this.lastAckLatencyMs = null;
    this.applied = 0;
    this.rejected = 0;
    this.lastError = null;
  }

  onHostEvent(event: HostEvent): void {
    if (!this.baseline) return;
    // Every native scene mutation, including open/new scene and Undo/Redo,
    // must recheck the active scene GUID before sending another edit.
    if (event.type !== 'scene.changed' && event.type !== 'component.changed') return;
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
    if (
      typeof snapshot.sceneGuid !== 'string' ||
      snapshot.sceneGuid.toLowerCase() !== this.baseline.sceneGuid.toLowerCase()
    ) {
      this.lastError = 'Authoring scene changed during the attached Play session';
      this.active = false;
      this.pending.clear();
      this.outstanding.clear();
      return;
    }
    let next: Map<string, Transform>;
    let nextLights: Map<string, Light>;
    try {
      next = extractScene(snapshot, false);
      nextLights = extractLights(snapshot, false);
    } catch {
      this.lastError = 'Authoring scene property snapshot exceeds safe limits';
      this.active = false;
      this.pending.clear();
      return;
    }
    const now = Date.now();
    for (const [guid, before] of this.observed) {
      const after = next.get(guid);
      if (!after) {
        // There is no safe native delete/replace transaction in L1.4 yet.
        // Sending further commands for an ambiguous removed GUID would risk
        // editing a stale world; require a new cooked baseline instead.
        this.lastError = 'Authored entity removed; reload the Live Play baseline to apply structural edits';
        this.active = false;
        this.pending.clear();
        this.outstanding.clear();
        return;
      }
      for (const property of properties) {
        if (!differs(before[property], after[property])) continue;
        if (!this.queueEdit(guid, 'arc.transform', property, after[property], now)) return;
      }
    }
    for (const guid of next.keys()) {
      if (!this.observed.has(guid)) {
        this.lastError = 'New authored entity requires a new Live Play baseline';
        this.active = false;
        this.pending.clear();
        this.outstanding.clear();
        return;
      }
    }
    for (const [guid, before] of this.observedLights) {
      const after = nextLights.get(guid);
      if (!after || after.component !== before.component) {
        this.lastError = 'Authored light component changed; reload the Live Play baseline';
        this.active = false;
        this.pending.clear();
        this.outstanding.clear();
        return;
      }
      for (const property of lightProperties) {
        if (
          differs(before[property], after[property]) &&
          !this.queueEdit(guid, before.component, property, after[property], now)
        )
          return;
      }
    }
    for (const guid of nextLights.keys()) {
      if (!this.observedLights.has(guid)) {
        this.lastError = 'New authored light component requires a new Live Play baseline';
        this.active = false;
        this.pending.clear();
        this.outstanding.clear();
        return;
      }
    }
    this.observed = next;
    this.observedLights = nextLights;
    if (!this.active || this.pending.size === 0) return;
    this.flushUpdates(now);
    this.scheduleCommit();
  }

  private queueEdit(
    guid: string,
    component: string,
    property: Property,
    value: readonly number[],
    now: number,
  ): boolean {
    const key = `${guid}:${component}:${property}`;
    if (this.revision >= Number.MAX_SAFE_INTEGER || (!this.pending.has(key) && this.pending.size >= maxPending)) {
      this.lastError = 'Live Play property queue exceeded safe limits';
      this.active = false;
      this.pending.clear();
      this.outstanding.clear();
      return false;
    }
    for (const [id, entry] of this.outstanding) {
      if (entry.key === key) this.outstanding.delete(id);
    }
    const previous = this.pending.get(key);
    this.pending.set(key, {
      guid,
      component,
      property,
      value: [...value],
      revision: ++this.revision,
      changedAt: now,
      lastUpdate: previous?.lastUpdate ?? Number.NEGATIVE_INFINITY,
      commitAttempts: 0,
      lastCommitAttempt: Number.NEGATIVE_INFINITY,
      commitInFlight: false,
    });
    return true;
  }

  /** Called only after the native runtime explicitly confirms a verified baseline. */
  activate(sessionId: string): void {
    if (!this.baseline || this.baseline.sessionId !== sessionId || this.lastError) return;
    this.active = true;
    // Changes made while cooking are sent as final values, not replayed drags.
    this.flushCommits(true);
    if (this.active && this.pending.size) this.scheduleCommit();
  }

  disconnect(): void {
    this.active = false;
    if (this.commitTimer) clearTimeout(this.commitTimer);
    this.commitTimer = null;
    // Do not replay old edits after reconnect or run a commit against a
    // native instance whose baseline/authority has been invalidated.
    this.outstanding.clear();
    this.pending.clear();
  }

  onRuntimeMessage(message: LivePlayMessage): void {
    if (
      !this.baseline ||
      message.projectGuid.toLowerCase() !== this.baseline.projectGuid.toLowerCase() ||
      message.buildId !== this.baseline.buildId ||
      message.targetProfile !== this.baseline.targetProfile ||
      message.sessionId !== this.baseline.sessionId ||
      message.instanceId !== this.baseline.instanceId
    )
      return;
    if (message.type === 'scene.baseline.status') {
      if (message.payload.status === 'active') this.activate(message.sessionId);
      else this.disconnect();
    } else if (message.type === 'command.ack') {
      const request = this.outstanding.get(message.payload.requestId);
      if (!request) return;
      this.outstanding.delete(message.payload.requestId);
      this.lastAckLatencyMs = Math.max(0, Date.now() - request.sentAt);
      if (message.payload.status === 'applied') {
        this.applied++;
        this.lastAppliedRevision = Math.max(this.lastAppliedRevision, message.payload.appliedRevision);
        this.runtimeRevision = Math.max(this.runtimeRevision, message.payload.appliedRevision);
        if (request.phase === 'commit' && this.pending.get(request.key)?.revision === request.revision)
          this.pending.delete(request.key);
        if (!this.pending.has(request.key)) {
          for (const [id, entry] of this.outstanding) {
            if (entry.key === request.key && entry.revision <= request.revision) this.outstanding.delete(id);
          }
        }
      } else {
        // Rejections are explicit failures. A stale revision may mean that a
        // previous ACK was lost; do not report success without confirmation.
        if (this.pending.get(request.key)?.revision !== request.revision) return;
        this.rejected++;
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
      component: { typeId: item.component, schemaVersion: 1 },
      property: item.property,
      value: item.value,
      authoringRevision: item.revision,
      expectedRuntimeRevision: this.runtimeRevision,
      phase,
    };
    if (
      !livePlayMessageSchema.safeParse({
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
      }).success
    ) {
      this.lastError = 'Invalid authoring property command';
      this.active = false;
      return false;
    }
    let id: string | null;
    try {
      id = this.send(payload);
    } catch {
      return false; // Retain the latest final value for explicit retry.
    }
    if (!id) return false;
    this.outstanding.set(id, {
      key: `${item.guid}:${item.component}:${item.property}`,
      revision: item.revision,
      phase,
      sentAt: Date.now(),
    });
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
      if (this.active && this.pending.size) this.scheduleCommit();
    }, commitQuietMs);
  }

  /** Flushes the latest final value even if prior drag updates were dropped. */
  flushCommits(force: boolean): void {
    if (!this.active) return;
    const now = Date.now();
    for (const item of this.pending.values()) {
      if (!force && now - item.changedAt < commitQuietMs) continue;
      const threshold = item.commitInFlight ? commitAckTimeoutMs : commitRetryMs;
      if (!force && now - item.lastCommitAttempt < threshold) continue;
      if (item.commitAttempts >= maxCommitAttempts) {
        this.lastError = 'Final Live Play edit was not acknowledged; reload the baseline before more edits';
        this.active = false;
        this.pending.clear();
        this.outstanding.clear();
        return;
      }
      item.commitAttempts++;
      item.lastCommitAttempt = now;
      item.commitInFlight = this.sendPending(item, 'commit');
    }
  }
}

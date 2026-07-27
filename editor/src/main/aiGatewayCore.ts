import { randomBytes, randomUUID } from 'node:crypto';

export type GatewayHostResponse = {
  kind: 'response';
  requestId: number;
  succeeded: boolean;
  error: string;
  payload: unknown;
  sceneRevision: number;
  worldEpoch: number;
  frameRevision: number;
};

export interface GatewayHostTransport {
  command(type: string, payload?: Record<string, unknown>, edit?: Record<string, unknown>,
    expectedSceneRevision?: number): Promise<GatewayHostResponse>;
  query(type: string, payload?: Record<string, unknown>): Promise<GatewayHostResponse>;
}

export type GatewayAuditEntry = {
  sequence: number;
  timestamp: string;
  clientId: string;
  category: 'connection' | 'read' | 'viewport' | 'edit' | 'security' | 'error';
  operation: string;
  succeeded: boolean;
  detail: string;
};

export type GatewayEvent = {
  sequence: number;
  timestamp: string;
  type: string;
  entity?: { index: number; generation: number };
  message: string;
  payload: unknown;
};

export type GatewayClient = {
  id: string;
  name: string;
  connectedAt: string;
  lastSeenAt: string;
};

export type GatewayEditRequest = {
  id: string;
  clientId: string;
  clientName: string;
  label: string;
  requestedAt: string;
  state: 'pending' | 'approved' | 'denied' | 'expired';
  expiresAt?: string;
};

export type GatewayEditSession = {
  id: string;
  transactionId: number;
  clientId: string;
  label: string;
  startedAt: string;
  lastActivityAt: string;
  expectedSceneRevision: number;
};

export type GatewayStatus = {
  enabled: boolean;
  endpoint: string;
  discoveryFile: string;
  protocolVersion: 1;
  sceneRevision: number;
  worldEpoch: number;
  frameRevision: number;
  eventSequence: number;
  clients: GatewayClient[];
  pendingEditRequests: GatewayEditRequest[];
  activeEditSession: GatewayEditSession | null;
  lastCommittedEdit: {
    clientId: string;
    label: string;
    sceneRevision: number;
    committedAt: string;
  } | null;
  viewportLease: { clientId: string; expiresAt: string } | null;
  audit: GatewayAuditEntry[];
};

type StatusListener = (status: GatewayStatus) => void;
type EventListener = (event: GatewayEvent) => void;

const now = (): string => new Date().toISOString();
const editIdleMilliseconds = 15 * 60 * 1000;
const maximumAuditEntries = 500;
const captureTimeoutMilliseconds = 10_000;
const viewportLeaseMilliseconds = 30_000;
const maximumGatewayEvents = 500;
const maximumRememberedCaptures = 8;
const maximumRememberedCaptureBytes = 128 * 1024 * 1024;

type ViewportRenderOptions = {
  renderMode: 'shaded' | 'wireframe';
  visualization: string;
  overlay: 'none' | 'selectedWireframe' | 'allWireframe';
  shadows: boolean;
  environment: {
    sky: boolean;
    fog: boolean;
    terrain: boolean;
    water: boolean;
    vegetation: boolean;
    decals: boolean;
  };
};

type CapturedImage = {
  channel: string;
  format: string;
  width: number;
  height: number;
  data: string;
};

type RememberedCapture = {
  captureId: number;
  frameIndex: number;
  images: CapturedImage[];
  objects: Array<{ id: number; guid: string }>;
};

const asObject = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? value as Record<string, unknown> : {};

const requireString = (value: unknown, name: string): string => {
  if (typeof value !== 'string' || value.trim() === '') {
    throw new Error(`${name} must be a non-empty string`);
  }
  return value;
};

const requireRevision = (value: unknown): number => {
  if (!Number.isSafeInteger(value) || Number(value) < 1) {
    throw new Error('expectedSceneRevision must be a positive integer');
  }
  return Number(value);
};

const requireProjectAssetPath = (value: unknown, name: string): string => {
  const assetPath = requireString(value, name);
  if (assetPath.includes('\\') || assetPath.startsWith('/') || /^[A-Za-z]:/.test(assetPath) ||
      assetPath.split('/').some((segment) => segment === '..' || segment === '.')) {
    throw new Error(`${name} must be a normalized project-relative path`);
  }
  return assetPath;
};

export class SceneGatewayCore {
  readonly token = randomBytes(32).toString('base64url');
  private endpoint = '';
  private discoveryFile = '';
  private auditSequence = 0;
  private eventSequence = 0;
  private sceneRevision = 0;
  private worldEpoch = 0;
  private frameRevision = 0;
  private readonly clients = new Map<string, GatewayClient>();
  private readonly auditEntries: GatewayAuditEntry[] = [];
  private readonly editRequests = new Map<string, GatewayEditRequest>();
  private readonly approvedClients = new Map<string, number>();
  private activeEdit: GatewayEditSession | null = null;
  private lastCommittedEdit: GatewayStatus['lastCommittedEdit'] = null;
  private viewportLease: { clientId: string; expiresAt: number } | null = null;
  private readonly listeners = new Set<StatusListener>();
  private readonly eventListeners = new Set<EventListener>();
  private readonly recentEvents: GatewayEvent[] = [];
  private readonly eventWaiters = new Set<(event: GatewayEvent) => void>();
  private nextTransactionId = Math.max(1, Math.floor(Math.random() * 0x3fffffff));
  private nextCaptureId = Math.max(1, Date.now());
  private readonly recentHostLogs: Array<{
    timestamp: string; level: string; source: string; message: string;
  }> = [];
  private readonly rememberedCaptures = new Map<number, RememberedCapture>();

  constructor(private readonly host: GatewayHostTransport) {}

  configure(endpoint: string, discoveryFile: string): void {
    this.endpoint = endpoint;
    this.discoveryFile = discoveryFile;
    this.notify();
  }

  onStatus(listener: StatusListener): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  onEvent(listener: EventListener): () => void {
    this.eventListeners.add(listener);
    return () => this.eventListeners.delete(listener);
  }

  recordHostEvent(event: {
    type: string;
    entity?: { index: number; generation: number };
    message: string;
    payload: unknown;
  }): void {
    this.appendEvent({
      type: event.type,
      entity: event.entity,
      message: event.message,
      payload: event.payload,
    });
  }

  status(): GatewayStatus {
    this.expirePermissions();
    return {
      enabled: Boolean(this.endpoint),
      endpoint: this.endpoint,
      discoveryFile: this.discoveryFile,
      protocolVersion: 1,
      sceneRevision: this.sceneRevision,
      worldEpoch: this.worldEpoch,
      frameRevision: this.frameRevision,
      eventSequence: this.eventSequence,
      clients: [...this.clients.values()],
      pendingEditRequests: [...this.editRequests.values()].filter((request) => request.state === 'pending'),
      activeEditSession: this.activeEdit,
      lastCommittedEdit: this.lastCommittedEdit,
      viewportLease: this.viewportLease ? {
        clientId: this.viewportLease.clientId,
        expiresAt: new Date(this.viewportLease.expiresAt).toISOString(),
      } : null,
      audit: this.auditEntries.slice(-100),
    };
  }

  touchClient(clientId: string, clientName = 'Local AI client'): GatewayClient {
    const timestamp = now();
    const existing = this.clients.get(clientId);
    if (existing) {
      existing.lastSeenAt = timestamp;
      if (clientName) existing.name = clientName;
      return existing;
    }
    const client = { id: clientId, name: clientName, connectedAt: timestamp, lastSeenAt: timestamp };
    this.clients.set(clientId, client);
    this.audit(clientId, 'connection', 'connect', true, clientName);
    this.notify();
    return client;
  }

  async disconnectClient(clientId: string): Promise<void> {
    if (this.activeEdit?.clientId === clientId) {
      const transactionId = this.activeEdit.transactionId;
      try {
        this.expect(await this.host.command('history.cancelTransaction', { id: transactionId }));
      } catch (error) {
        this.audit(clientId, 'error', 'edit.disconnectCancel', false,
          error instanceof Error ? error.message : String(error));
      } finally {
        this.activeEdit = null;
      }
    }
    if (this.clients.delete(clientId)) {
      if (this.viewportLease?.clientId === clientId) this.viewportLease = null;
      this.approvedClients.delete(clientId);
      this.audit(clientId, 'connection', 'disconnect', true, '');
      this.notify();
    }
  }

  requestEdit(clientId: string, label: string): GatewayEditRequest {
    const client = this.touchClient(clientId);
    const existing = [...this.editRequests.values()].find(
      (request) => request.clientId === clientId && request.state === 'pending');
    if (existing) return existing;
    const request: GatewayEditRequest = {
      id: randomUUID(),
      clientId,
      clientName: client.name,
      label: label.trim() || 'AI Scene Edit',
      requestedAt: now(),
      state: 'pending',
    };
    this.editRequests.set(request.id, request);
    this.audit(clientId, 'edit', 'request', true, request.label);
    this.notify();
    return request;
  }

  approveEdit(requestId: string): boolean {
    const request = this.editRequests.get(requestId);
    if (!request || request.state !== 'pending') return false;
    request.state = 'approved';
    const expires = Date.now() + editIdleMilliseconds;
    request.expiresAt = new Date(expires).toISOString();
    this.approvedClients.set(request.clientId, expires);
    this.audit(request.clientId, 'edit', 'approve', true, request.label);
    this.notify();
    return true;
  }

  denyEdit(requestId: string): boolean {
    const request = this.editRequests.get(requestId);
    if (!request || request.state !== 'pending') return false;
    request.state = 'denied';
    this.audit(request.clientId, 'edit', 'deny', true, request.label);
    this.notify();
    return true;
  }

  async revokeClient(clientId: string): Promise<void> {
    if (this.activeEdit?.clientId === clientId) {
      await this.cancelEdit(this.activeEdit.id, clientId);
    }
    this.approvedClients.delete(clientId);
    for (const request of this.editRequests.values()) {
      if (request.clientId === clientId && request.state === 'pending') request.state = 'denied';
    }
    if (this.viewportLease?.clientId === clientId) this.viewportLease = null;
    this.audit(clientId, 'security', 'revoke', true, '');
    this.notify();
  }

  async invalidateAuthority(reason: string): Promise<void> {
    if (this.activeEdit) {
      try {
        await this.host.command('history.cancelTransaction', { id: this.activeEdit.transactionId });
      } catch {
        // The native host may already have replaced the world; clearing the lease is still required.
      }
    }
    this.activeEdit = null;
    this.lastCommittedEdit = null;
    this.viewportLease = null;
    this.approvedClients.clear();
    for (const request of this.editRequests.values()) {
      if (request.state === 'pending') request.state = 'expired';
    }
    this.audit('editor', 'security', 'authority.invalidate', true, reason);
    this.notify();
  }

  recordHostLog(event: { level: string; source: string; message: string; timestamp?: string }): void {
    this.recentHostLogs.push({
      timestamp: event.timestamp ?? now(),
      level: event.level,
      source: event.source,
      message: event.message,
    });
    if (this.recentHostLogs.length > 200)
      this.recentHostLogs.splice(0, this.recentHostLogs.length - 200);
    this.appendEvent({
      type: 'diagnostic.log',
      message: event.message,
      payload: { level: event.level, source: event.source },
    });
  }

  async invoke(method: string, rawParams: unknown, clientId: string): Promise<unknown> {
    await this.expireInactiveAuthority();
    const params = asObject(rawParams);
    this.touchClient(clientId, typeof params.clientName === 'string' ? params.clientName : undefined);
    try {
      const result = await this.dispatch(method, params, clientId);
      const category = method.startsWith('viewport.') ? 'viewport' :
        method.startsWith('edit.') || method.startsWith('history.') ? 'edit' : 'read';
      this.audit(clientId, category, method, true, '');
      return result;
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      this.audit(clientId, 'error', method, false, message);
      throw error;
    }
  }

  private async dispatch(method: string, params: Record<string, unknown>, clientId: string): Promise<unknown> {
    switch (method) {
      case 'gateway.status':
        return this.publicStatus();
      case 'scene.overview':
        return this.expect(await this.host.query('gateway.sceneEntities', {
          search: '', offset: 0, limit: 200,
        }));
      case 'scene.findEntities':
        return this.expect(await this.host.query('gateway.sceneEntities', {
          search: typeof params.search === 'string' ? params.search : '',
          offset: Number.isSafeInteger(params.offset) ? params.offset : 0,
          limit: Number.isSafeInteger(params.limit) ? params.limit : 100,
        }));
      case 'scene.getEntity':
        return this.expect(await this.host.query('gateway.entity', {
          guid: requireString(params.guid, 'guid'),
        }));
      case 'scene.componentSchemas':
        return this.expect(await this.host.query('gateway.componentSchemas'));
      case 'scene.spatialQuery':
        return this.expect(await this.host.query('gateway.spatialQuery', {
          kind: typeof params.kind === 'string' ? params.kind : 'nearby',
          origin: params.origin,
          direction: params.direction,
          center: params.center,
          extent: params.extent,
          radius: Number(params.radius) || 10,
          limit: Number.isSafeInteger(params.limit) ? params.limit : 100,
        }));
      case 'scene.changes': {
        const since = Math.max(0, Math.floor(Number(params.sinceSceneRevision) || 0));
        const response = await this.host.query('gateway.sceneEntities', { search: '', offset: 0, limit: 200 });
        const snapshot = this.expect(response) as Record<string, unknown>;
        return since === response.sceneRevision
          ? { sceneRevision: response.sceneRevision, worldEpoch: response.worldEpoch, changes: [] }
          : { ...snapshot, changes: [], fullSnapshotRequired: true, sinceSceneRevision: since };
      }
      case 'assets.list':
        return this.expect(await this.host.query('project.assets'));
      case 'diagnostics.get':
        return this.diagnostics();
      case 'events.wait':
        return this.waitForEvent(params);
      case 'viewport.state':
        return this.expect(await this.host.query('viewport.state'));
      case 'viewport.move':
        this.claimViewport(clientId);
        return this.moveViewport(params);
      case 'viewport.setRenderOptions':
        this.claimViewport(clientId);
        return this.setViewportRenderOptions(params);
      case 'viewport.debug':
        this.claimViewport(clientId);
        return this.debugViewport(params);
      case 'viewport.inspectPixel':
        return this.inspectViewportPixel(params);
      case 'viewport.compare':
        return this.compareViewportCaptures(params);
      case 'viewport.pick':
        this.claimViewport(clientId);
        return this.pickViewport(params);
      case 'viewport.observe':
        return this.captureViewport(params);
      case 'edit.request':
        return this.requestEdit(clientId, typeof params.label === 'string' ? params.label : 'AI Scene Edit');
      case 'edit.begin':
        return this.beginEdit(clientId, params);
      case 'edit.apply':
        return this.applyEdit(clientId, params);
      case 'edit.commit':
        return this.commitEdit(requireString(params.editSessionId, 'editSessionId'), clientId,
          requireRevision(params.expectedSceneRevision));
      case 'edit.cancel':
        return this.cancelEdit(requireString(params.editSessionId, 'editSessionId'), clientId);
      case 'history.undo':
      case 'history.redo':
        this.requireApproved(clientId);
        return this.expect(await this.host.command(method, {}, undefined,
          requireRevision(params.expectedSceneRevision)));
      default:
        throw new Error(`Unsupported gateway method: ${method}`);
    }
  }

  private async diagnostics(): Promise<unknown> {
    const [scene, viewport, renderer, history, capture] = await Promise.all([
      this.host.query('gateway.sceneEntities', { search: '', offset: 0, limit: 200 }),
      this.host.query('viewport.state'),
      this.host.query('gateway.diagnostics'),
      this.host.query('history.state'),
      this.captureViewport({ color: true, depth: true, objectId: true, normals: true }),
    ]);
    return {
      scene: this.expect(scene),
      viewport: this.expect(viewport),
      renderer: this.expect(renderer),
      history: this.expect(history),
      recentHostLogs: this.recentHostLogs.slice(-100),
      capture,
      anomalies: this.collectDiagnosticAnomalies(
        this.expect(renderer) as Record<string, unknown>,
        asObject(capture)),
    };
  }

  private async waitForEvent(params: Record<string, unknown>): Promise<unknown> {
    const kind = typeof params.kind === 'string' ? params.kind : 'scene';
    if (!['scene', 'frame', 'selection', 'diagnostic'].includes(kind))
      throw new Error('events.wait kind must be scene, frame, selection, or diagnostic');
    const timeout = Math.min(30_000, Math.max(1, Math.floor(Number(params.timeoutMs) || 10_000)));
    if (kind === 'frame') {
      const afterFrame = Math.max(
        this.frameRevision,
        Math.floor(Number(params.afterFrameRevision) || 0));
      await Promise.race([
        this.waitForFrame(afterFrame + 1),
        new Promise<never>((_resolve, reject) =>
          setTimeout(() => reject(new Error('Timed out waiting for a frame event')), timeout)),
      ]);
      return this.expect(await this.host.query('viewport.state'));
    }

    const afterSequence = Math.max(0, Math.floor(Number(params.afterSequence) || 0));
    const matches = (event: GatewayEvent) => event.sequence > afterSequence && (
      kind === 'selection' ? event.type === 'entity.selected' :
      kind === 'diagnostic' ? event.type.startsWith('diagnostic.') :
      event.type !== 'entity.selected' && !event.type.startsWith('diagnostic.'));
    let event = this.recentEvents.find(matches);
    if (!event) {
      event = await new Promise<GatewayEvent>((resolve, reject) => {
        const waiter = (candidate: GatewayEvent) => {
          if (!matches(candidate)) return;
          clearTimeout(timer);
          this.eventWaiters.delete(waiter);
          resolve(candidate);
        };
        const timer = setTimeout(() => {
          this.eventWaiters.delete(waiter);
          reject(new Error(`Timed out waiting for a ${kind} event`));
        }, timeout);
        this.eventWaiters.add(waiter);
      });
    }
    const state = kind === 'diagnostic'
      ? this.expect(await this.host.query('gateway.diagnostics'))
      : this.expect(await this.host.query('gateway.sceneEntities', { search: '', offset: 0, limit: 200 }));
    return { event, state };
  }

  private async captureViewport(params: Record<string, unknown>): Promise<unknown> {
    const viewportBefore = this.expect(await this.host.query('viewport.state')) as Record<string, unknown>;
    const waitFrames = Math.min(120, Math.max(0, Math.floor(Number(params.waitFrames) || 0)));
    const targetFrame = Number(viewportBefore.frameIndex ?? 0) + waitFrames;
    if (waitFrames > 0) await this.waitForFrame(targetFrame);

    const captureId = this.nextCaptureId++;
    const captureMaxWidth = Math.min(1920, Math.max(1, Math.floor(Number(params.maxWidth) || 1280)));
    const captureMaxHeight = Math.min(1080, Math.max(1, Math.floor(Number(params.maxHeight) || 1080)));
    this.expect(await this.host.command('viewport.capture', {
      captureId,
      color: params.color !== false,
      depth: params.depth !== false,
      objectId: params.objectId !== false,
      normals: params.normals !== false,
      sceneColor: params.sceneColor === true,
      baseColor: params.baseColor === true,
      materialProperties: params.materialProperties === true,
      emissive: params.emissive === true,
    }));

    const deadline = Date.now() + captureTimeoutMilliseconds;
    while (Date.now() < deadline) {
      const response = await this.host.query('viewport.captureResult', { captureId });
      const result = this.expect(response) as Record<string, unknown>;
      if (result.pending !== true) {
        const enriched = this.enrichCapture(result);
        this.rememberCapture(enriched);
        const samples = Array.isArray(params.samplePixels)
          ? params.samplePixels.slice(0, 64).map((sample) => {
            const point = asObject(sample);
            return this.sampleCapturePixel(
              enriched,
              Math.floor(Number(point.x) || 0),
              Math.floor(Number(point.y) || 0));
          })
          : undefined;
        return {
          ...enriched,
          ...(samples ? { samples } : {}),
          captureMaxWidth,
          captureMaxHeight,
        };
      }
      await new Promise((resolve) => setTimeout(resolve, 8));
    }
    throw new Error(`Viewport capture ${captureId} did not complete within 10 seconds`);
  }

  private async waitForFrame(targetFrame: number): Promise<void> {
    const deadline = Date.now() + captureTimeoutMilliseconds;
    while (Date.now() < deadline) {
      const state = this.expect(await this.host.query('viewport.state')) as Record<string, unknown>;
      if (Number(state.frameIndex ?? 0) >= targetFrame) return;
      await new Promise((resolve) => setTimeout(resolve, 8));
    }
    throw new Error(`Viewport did not reach frame ${targetFrame} within 10 seconds`);
  }

  private async moveViewport(params: Record<string, unknown>): Promise<unknown> {
    const action = typeof params.action === 'string' ? params.action : 'orbit';
    if (action === 'frame') {
      const entity = await this.resolveEntity(requireString(params.guid, 'guid'));
      this.expect(await this.host.command('entity.select', { entity }));
      this.expect(await this.host.command('viewport.cameraInput', { focusSelected: true }));
      return this.settleViewport(params);
    }
    if (action === 'place') {
      this.expect(await this.host.command('viewport.setPose', {
        position: params.position,
        target: params.target,
      }));
      return this.settleViewport(params);
    }
    const payload: Record<string, unknown> = {};
    if (action === 'orbit') {
      payload.orbitX = Number(params.x) || 0;
      payload.orbitY = Number(params.y) || 0;
    } else if (action === 'pan') {
      payload.panX = Number(params.x) || 0;
      payload.panY = Number(params.y) || 0;
    } else if (action === 'dolly') {
      payload.zoom = Number(params.amount) || 0;
    } else {
      throw new Error('viewport.move action must be orbit, pan, dolly, frame, or place');
    }
    this.expect(await this.host.command('viewport.cameraInput', payload));
    return this.settleViewport(params);
  }

  private async setViewportRenderOptions(params: Record<string, unknown>): Promise<unknown> {
    const renderModes = new Set(['shaded', 'wireframe']);
    const visualizations = new Set([
      'standard', 'albedo', 'opacity', 'worldNormal', 'specularity', 'gloss',
      'metalness', 'ao', 'emission', 'lighting', 'uv0', 'cascadeDebug',
      'shadowMask', 'lightComplexity', 'clusterDebug',
    ]);
    const overlays = new Set(['none', 'selectedWireframe', 'allWireframe']);
    const before = this.expect(await this.host.query('viewport.state')) as Record<string, unknown>;
    const previous = this.normalizedRenderOptions(asObject(before.renderOptions));
    const renderMode = typeof params.renderMode === 'string' ? params.renderMode : previous.renderMode;
    const visualization = typeof params.visualization === 'string'
      ? params.visualization : previous.visualization;
    const overlay = typeof params.overlay === 'string' ? params.overlay : previous.overlay;
    if (!renderModes.has(renderMode)) throw new Error(`Unsupported viewport render mode: ${renderMode}`);
    if (!visualizations.has(visualization))
      throw new Error(`Unsupported viewport visualization: ${visualization}`);
    if (!overlays.has(overlay)) throw new Error(`Unsupported viewport overlay: ${overlay}`);
    const environment = params.environment && typeof params.environment === 'object'
      ? params.environment as Record<string, unknown>
      : {};
    const visibility = {
      sky: typeof environment.sky === 'boolean' ? environment.sky : previous.environment.sky,
      fog: typeof environment.fog === 'boolean' ? environment.fog : previous.environment.fog,
      terrain: typeof environment.terrain === 'boolean' ? environment.terrain : previous.environment.terrain,
      water: typeof environment.water === 'boolean' ? environment.water : previous.environment.water,
      vegetation: typeof environment.vegetation === 'boolean'
        ? environment.vegetation : previous.environment.vegetation,
      decals: typeof environment.decals === 'boolean' ? environment.decals : previous.environment.decals,
    };
    const requested: ViewportRenderOptions = {
      renderMode: renderMode as ViewportRenderOptions['renderMode'],
      visualization,
      overlay: overlay as ViewportRenderOptions['overlay'],
      shadows: typeof params.shadows === 'boolean' ? params.shadows : previous.shadows,
      environment: visibility,
    };
    const acknowledgement = this.expect(
      await this.host.command('viewport.setRenderOptions', requested)) as Record<string, unknown>;
    const settled = await this.settleViewport(params) as Record<string, unknown>;
    const effective = this.normalizedRenderOptions(asObject(settled.renderOptions));
    const mismatches = this.renderOptionMismatches(requested, effective);
    return {
      requested,
      effective,
      applied: mismatches.length === 0,
      mismatches,
      acknowledgement,
      state: settled,
      appliedFrameRevision: Number(settled.frameRevision ?? this.frameRevision),
    };
  }

  private async debugViewport(params: Record<string, unknown>): Promise<unknown> {
    const before = this.expect(await this.host.query('viewport.state')) as Record<string, unknown>;
    let renderOptions: unknown;
    const requestedOptions = asObject(params.renderOptions);
    if (Object.keys(requestedOptions).length > 0) {
      renderOptions = await this.setViewportRenderOptions({ ...requestedOptions, waitFrames: 0 });
    }
    const movement = asObject(params.camera);
    if (Object.keys(movement).length > 0) {
      await this.moveViewport({ ...movement, waitFrames: 0 });
    }
    const preSettle = this.expect(await this.host.query('viewport.state')) as Record<string, unknown>;
    const waitFrames = Math.min(120, Math.max(1, Math.floor(Number(params.waitFrames) || 2)));
    await this.waitForFrame(Number(preSettle.frameIndex ?? 0) + waitFrames);
    const capture = asObject(await this.captureViewport({
      ...asObject(params.capture),
      samplePixels: params.samplePixels,
      waitFrames: 0,
    }));
    const [afterResponse, rendererResponse] = await Promise.all([
      this.host.query('viewport.state'),
      this.host.query('gateway.diagnostics'),
    ]);
    const after = this.expect(afterResponse) as Record<string, unknown>;
    const renderer = this.expect(rendererResponse) as Record<string, unknown>;
    const baselineCaptureId = Math.floor(Number(params.baselineCaptureId) || 0);
    const comparison = baselineCaptureId > 0
      ? this.compareRememberedCaptures(
        this.requireRememberedCapture(baselineCaptureId),
        this.requireRememberedCapture(Number(capture.captureId)))
      : undefined;
    return {
      operation: 'configure-apply-settle-capture',
      requested: {
        renderOptions: Object.keys(requestedOptions).length > 0 ? requestedOptions : undefined,
        camera: Object.keys(movement).length > 0 ? movement : undefined,
        settleFrames: waitFrames,
      },
      effective: {
        renderOptions: this.normalizedRenderOptions(asObject(after.renderOptions)),
        camera: after.camera,
      },
      transition: {
        fromFrame: Number(before.frameIndex ?? 0),
        appliedFrame: Number(preSettle.frameIndex ?? 0),
        capturedFrame: Number(capture.frameIndex ?? 0),
        finalFrame: Number(after.frameIndex ?? 0),
      },
      renderOptions,
      renderer,
      capture,
      comparison,
      anomalies: [
        ...this.collectDiagnosticAnomalies(renderer, capture),
        ...this.cameraAnomalies(asObject(capture.camera)),
      ],
      sceneRevision: this.sceneRevision,
      worldEpoch: this.worldEpoch,
      frameRevision: this.frameRevision,
    };
  }

  private async inspectViewportPixel(params: Record<string, unknown>): Promise<unknown> {
    const captureId = Math.floor(Number(params.captureId) || 0);
    const capture = captureId > 0
      ? this.requireRememberedCapture(captureId)
      : this.requireRememberedCapture(Number(
        asObject(await this.captureViewport({
          color: true, depth: true, objectId: true, normals: true, waitFrames: params.waitFrames,
        })).captureId));
    return {
      ...this.sampleCapturePixel(
        capture,
        Math.floor(Number(params.x) || 0),
        Math.floor(Number(params.y) || 0)),
      sceneRevision: this.sceneRevision,
      worldEpoch: this.worldEpoch,
      frameRevision: this.frameRevision,
    };
  }

  private async compareViewportCaptures(params: Record<string, unknown>): Promise<unknown> {
    const baseline = this.requireRememberedCapture(
      Math.floor(Number(params.baselineCaptureId) || 0));
    let currentId = Math.floor(Number(params.currentCaptureId) || 0);
    if (currentId <= 0) {
      const current = asObject(await this.captureViewport({
        color: true, depth: true, objectId: true, normals: true,
        waitFrames: params.waitFrames,
      }));
      currentId = Number(current.captureId);
    }
    return {
      ...this.compareRememberedCaptures(baseline, this.requireRememberedCapture(currentId)),
      sceneRevision: this.sceneRevision,
      worldEpoch: this.worldEpoch,
      frameRevision: this.frameRevision,
    };
  }

  private async pickViewport(params: Record<string, unknown>): Promise<unknown> {
    const before = this.expect(await this.host.query('viewport.state')) as Record<string, unknown>;
    const request = this.expect(await this.host.command('viewport.pick', {
      x: Math.max(0, Math.floor(Number(params.x) || 0)),
      y: Math.max(0, Math.floor(Number(params.y) || 0)),
    }));
    await this.waitForFrame(Number(before.frameIndex ?? 0) + 2);
    const selection = this.expect(await this.host.query('entity.selected'));
    return { request, selection };
  }

  private async settleViewport(params: Record<string, unknown>): Promise<unknown> {
    const state = this.expect(await this.host.query('viewport.state')) as Record<string, unknown>;
    const waitFrames = Math.min(120, Math.max(0, Math.floor(Number(params.waitFrames) || 0)));
    if (waitFrames > 0) await this.waitForFrame(Number(state.frameIndex ?? 0) + waitFrames);
    return this.expect(await this.host.query('viewport.state'));
  }

  private normalizedRenderOptions(value: Record<string, unknown>): ViewportRenderOptions {
    const environment = asObject(value.environment);
    return {
      renderMode: value.renderMode === 'wireframe' ? 'wireframe' : 'shaded',
      visualization: typeof value.visualization === 'string' ? value.visualization : 'standard',
      overlay: value.overlay === 'selectedWireframe' || value.overlay === 'allWireframe'
        ? value.overlay : 'none',
      shadows: typeof value.shadows === 'boolean' ? value.shadows : true,
      environment: {
        sky: typeof environment.sky === 'boolean' ? environment.sky : true,
        fog: typeof environment.fog === 'boolean' ? environment.fog : true,
        terrain: typeof environment.terrain === 'boolean' ? environment.terrain : true,
        water: typeof environment.water === 'boolean' ? environment.water : true,
        vegetation: typeof environment.vegetation === 'boolean' ? environment.vegetation : true,
        decals: typeof environment.decals === 'boolean' ? environment.decals : true,
      },
    };
  }

  private renderOptionMismatches(requested: ViewportRenderOptions, effective: ViewportRenderOptions): string[] {
    const mismatches: string[] = [];
    if (requested.renderMode !== effective.renderMode) mismatches.push('renderMode');
    if (requested.visualization !== effective.visualization) mismatches.push('visualization');
    if (requested.overlay !== effective.overlay) mismatches.push('overlay');
    if (requested.shadows !== effective.shadows) mismatches.push('shadows');
    for (const key of Object.keys(requested.environment) as Array<keyof ViewportRenderOptions['environment']>) {
      if (requested.environment[key] !== effective.environment[key])
        mismatches.push(`environment.${key}`);
    }
    return mismatches;
  }

  private enrichCapture(value: Record<string, unknown>): Record<string, unknown> {
    const images = Array.isArray(value.images)
      ? value.images.map((image) => asObject(image)).filter((image) => typeof image.data === 'string')
      : [];
    const camera = asObject(value.camera);
    return {
      ...value,
      camera: {
        ...camera,
        telemetry: this.cameraTelemetry(camera),
      },
      analysis: {
        channels: Object.fromEntries(images.map((image) => [
          String(image.channel),
          this.analyzeImage(image as unknown as CapturedImage),
        ])),
      },
    };
  }

  private rememberCapture(value: Record<string, unknown>): void {
    const captureId = Math.floor(Number(value.captureId) || 0);
    if (captureId <= 0) return;
    const images = Array.isArray(value.images)
      ? value.images.map((image) => asObject(image)).filter((image) =>
        typeof image.channel === 'string' && typeof image.format === 'string' &&
        typeof image.data === 'string' && Number(image.width) > 0 && Number(image.height) > 0)
        .map((image) => ({
          channel: String(image.channel),
          format: String(image.format),
          width: Math.floor(Number(image.width)),
          height: Math.floor(Number(image.height)),
          data: String(image.data),
        }))
      : [];
    const objects = Array.isArray(value.objects)
      ? value.objects.map((object) => asObject(object)).filter((object) =>
        Number.isSafeInteger(object.id) && typeof object.guid === 'string')
        .map((object) => ({ id: Number(object.id), guid: String(object.guid) }))
      : [];
    this.rememberedCaptures.set(captureId, {
      captureId,
      frameIndex: Math.floor(Number(value.frameIndex) || 0),
      images,
      objects,
    });
    const retainedBytes = () => [...this.rememberedCaptures.values()].reduce(
      (total, capture) => total + capture.images.reduce(
        (captureTotal, image) => captureTotal + Buffer.byteLength(image.data, 'base64'),
        0),
      0);
    while (this.rememberedCaptures.size > maximumRememberedCaptures ||
           retainedBytes() > maximumRememberedCaptureBytes) {
      const oldest = this.rememberedCaptures.keys().next().value as number | undefined;
      if (oldest === undefined) break;
      this.rememberedCaptures.delete(oldest);
    }
  }

  private requireRememberedCapture(captureId: number): RememberedCapture {
    const capture = this.rememberedCaptures.get(captureId);
    if (!capture)
      throw new Error(`Capture ${captureId || '(missing)'} is unavailable or has expired`);
    return capture;
  }

  private analyzeImage(image: CapturedImage): Record<string, unknown> {
    const data = Buffer.from(image.data, 'base64');
    const pixelCount = Math.max(0, image.width * image.height);
    const stride = Math.max(1, Math.ceil(pixelCount / 262_144));
    let samples = 0;
    let invalid = 0;
    let minimum = Number.POSITIVE_INFINITY;
    let maximum = Number.NEGATIVE_INFINITY;
    let total = 0;
    let dark = 0;
    let bright = 0;
    const unique = new Set<number>();
    for (let index = 0; index < pixelCount; index += stride) {
      const values = this.readPixel(data, image.format, index);
      if (!values || values.some((entry) => !Number.isFinite(entry))) {
        ++invalid;
        continue;
      }
      ++samples;
      if (image.channel === 'objectId') {
        unique.add(Math.floor(values[0]));
        total += values[0] === 0 ? 1 : 0;
        continue;
      }
      const scalar = image.channel === 'color'
        ? values[0] * 0.2126 + values[1] * 0.7152 + values[2] * 0.0722
        : image.channel === 'normals'
          ? Math.sqrt(values[0] ** 2 + values[1] ** 2 + values[2] ** 2)
          : values[0];
      minimum = Math.min(minimum, scalar);
      maximum = Math.max(maximum, scalar);
      total += scalar;
      if (scalar <= 0.01) ++dark;
      if (scalar >= 0.99 && image.channel === 'color') ++bright;
    }
    if (image.channel === 'objectId') {
      return {
        width: image.width,
        height: image.height,
        sampledPixels: samples,
        uniqueIds: unique.size,
        backgroundFraction: samples > 0 ? total / samples : 0,
        invalidSamples: invalid,
      };
    }
    return {
      width: image.width,
      height: image.height,
      sampledPixels: samples,
      minimum: samples > 0 ? minimum : null,
      maximum: samples > 0 ? maximum : null,
      average: samples > 0 ? total / samples : null,
      darkFraction: samples > 0 ? dark / samples : 0,
      brightFraction: samples > 0 ? bright / samples : 0,
      invalidSamples: invalid,
    };
  }

  private sampleCapturePixel(capture: RememberedCapture | Record<string, unknown>,
    x: number, y: number): Record<string, unknown> {
    const images = 'images' in capture && Array.isArray(capture.images)
      ? capture.images as CapturedImage[]
      : [];
    const objectsValue = 'objects' in capture && Array.isArray(capture.objects) ? capture.objects : [];
    const objects = new Map<number, string>(
      objectsValue.map((object) => asObject(object))
        .filter((object) => Number.isSafeInteger(object.id) && typeof object.guid === 'string')
        .map((object) => [Number(object.id), String(object.guid)]));
    const reference = images.find((image) => image.channel === 'color') ?? images[0];
    if (!reference) throw new Error('Capture has no image channels to inspect');
    const outputX = Math.max(0, Math.min(reference.width - 1, x));
    const outputY = Math.max(0, Math.min(reference.height - 1, y));
    const channels: Record<string, unknown> = {};
    for (const image of images) {
      const imageX = Math.max(0, Math.min(image.width - 1,
        Math.floor((outputX + 0.5) * image.width / reference.width)));
      const imageY = Math.max(0, Math.min(image.height - 1,
        Math.floor((outputY + 0.5) * image.height / reference.height)));
      const index = imageY * image.width + imageX;
      const values = this.readPixel(Buffer.from(image.data, 'base64'), image.format, index);
      if (!values) {
        channels[image.channel] = { unavailable: true, format: image.format };
        continue;
      }
      if (image.channel === 'objectId') {
        const id = Math.floor(values[0]);
        channels[image.channel] = { id, guid: objects.get(id) ?? null };
      } else {
        channels[image.channel] = { value: values, format: image.format };
      }
    }
    return {
      captureId: Number('captureId' in capture ? capture.captureId : 0),
      frameIndex: Number('frameIndex' in capture ? capture.frameIndex : 0),
      requestedPixel: [x, y],
      sampledPixel: [outputX, outputY],
      outputExtent: [reference.width, reference.height],
      channels,
    };
  }

  private readPixel(data: Buffer, format: string, index: number): number[] | null {
    if (index < 0) return null;
    if (format === 'rgba8' || format === 'bgra8') {
      const offset = index * 4;
      if (offset + 4 > data.length) return null;
      const first = data[offset] / 255;
      const second = data[offset + 1] / 255;
      const third = data[offset + 2] / 255;
      return format === 'bgra8'
        ? [third, second, first, data[offset + 3] / 255]
        : [first, second, third, data[offset + 3] / 255];
    }
    if (format === 'rgba16f') {
      const offset = index * 8;
      if (offset + 8 > data.length) return null;
      return [0, 2, 4, 6].map((component) => this.halfToFloat(data.readUInt16LE(offset + component)));
    }
    if (format === 'r32f') {
      const offset = index * 4;
      return offset + 4 <= data.length ? [data.readFloatLE(offset)] : null;
    }
    if (format === 'r32ui') {
      const offset = index * 4;
      return offset + 4 <= data.length ? [data.readUInt32LE(offset)] : null;
    }
    return null;
  }

  private halfToFloat(bits: number): number {
    const sign = bits >>> 15 ? -1 : 1;
    const exponent = (bits >>> 10) & 0x1f;
    const mantissa = bits & 0x3ff;
    if (exponent === 0) return sign * mantissa * 2 ** -24;
    if (exponent === 31) return mantissa ? Number.NaN : sign * Number.POSITIVE_INFINITY;
    return sign * (1 + mantissa / 1024) * 2 ** (exponent - 15);
  }

  private compareRememberedCaptures(
    baseline: RememberedCapture,
    current: RememberedCapture): Record<string, unknown> {
    const channels: Record<string, unknown> = {};
    for (const baselineImage of baseline.images) {
      const currentImage = current.images.find((image) => image.channel === baselineImage.channel);
      if (!currentImage || currentImage.width !== baselineImage.width ||
          currentImage.height !== baselineImage.height || currentImage.format !== baselineImage.format) {
        channels[baselineImage.channel] = { comparable: false, reason: 'format or extent changed' };
        continue;
      }
      const left = Buffer.from(baselineImage.data, 'base64');
      const right = Buffer.from(currentImage.data, 'base64');
      const pixels = baselineImage.width * baselineImage.height;
      const stride = Math.max(1, Math.ceil(pixels / 262_144));
      let samples = 0;
      let absoluteError = 0;
      let changed = 0;
      for (let index = 0; index < pixels; index += stride) {
        const a = this.readPixel(left, baselineImage.format, index);
        const b = this.readPixel(right, currentImage.format, index);
        if (!a || !b || a.some((entry) => !Number.isFinite(entry)) ||
            b.some((entry) => !Number.isFinite(entry))) continue;
        const components = Math.min(a.length, b.length, baselineImage.channel === 'objectId' ? 1 : 3);
        let pixelError = 0;
        for (let component = 0; component < components; ++component)
          pixelError += Math.abs(a[component] - b[component]);
        pixelError /= components;
        absoluteError += pixelError;
        if (pixelError > (baselineImage.channel === 'objectId' ? 0 : 1 / 255)) ++changed;
        ++samples;
      }
      channels[baselineImage.channel] = {
        comparable: true,
        sampledPixels: samples,
        meanAbsoluteError: samples > 0 ? absoluteError / samples : 0,
        changedFraction: samples > 0 ? changed / samples : 0,
      };
    }
    return {
      baselineCaptureId: baseline.captureId,
      currentCaptureId: current.captureId,
      baselineFrame: baseline.frameIndex,
      currentFrame: current.frameIndex,
      channels,
    };
  }

  private cameraTelemetry(camera: Record<string, unknown>): Record<string, unknown> {
    const vector = (key: string, fallback: number[]): number[] => {
      const value = camera[key];
      return Array.isArray(value) && value.length >= 3
        ? value.slice(0, 3).map((entry) => Number(entry))
        : fallback;
    };
    const forward = vector('forward', [0, 0, -1]);
    const up = vector('up', [0, 1, 0]);
    const length = (value: number[]) => Math.hypot(value[0], value[1], value[2]);
    const forwardLength = length(forward);
    const upLength = length(up);
    const normalizedForward = forward.map((entry) => entry / Math.max(forwardLength, 1e-8));
    const normalizedUp = up.map((entry) => entry / Math.max(upLength, 1e-8));
    const right = [
      normalizedForward[1] * normalizedUp[2] - normalizedForward[2] * normalizedUp[1],
      normalizedForward[2] * normalizedUp[0] - normalizedForward[0] * normalizedUp[2],
      normalizedForward[0] * normalizedUp[1] - normalizedForward[1] * normalizedUp[0],
    ];
    const degrees = 180 / Math.PI;
    return {
      yawDegrees: Math.atan2(normalizedForward[0], -normalizedForward[2]) * degrees,
      pitchDegrees: Math.asin(Math.max(-1, Math.min(1, normalizedForward[1]))) * degrees,
      rollDegrees: Math.asin(Math.max(-1, Math.min(1, right[1] / Math.max(length(right), 1e-8)))) * degrees,
      forwardLength,
      upLength,
      orthogonality: normalizedForward.reduce(
        (sum, entry, index) => sum + entry * normalizedUp[index], 0),
    };
  }

  private cameraAnomalies(camera: Record<string, unknown>): string[] {
    const telemetry = asObject(camera.telemetry);
    const anomalies: string[] = [];
    if (Math.abs(Number(telemetry.rollDegrees) || 0) > 0.1)
      anomalies.push(`Camera has ${Number(telemetry.rollDegrees).toFixed(2)} degrees of roll`);
    if (Math.abs(Number(telemetry.orthogonality) || 0) > 0.001)
      anomalies.push('Camera forward and up axes are not orthogonal');
    const nearPlane = Number(camera.nearPlane);
    const farPlane = Number(camera.farPlane);
    if (nearPlane > 0 && farPlane / nearPlane > 100_000)
      anomalies.push(`Camera far/near ratio is high (${Math.round(farPlane / nearPlane)}:1)`);
    return anomalies;
  }

  private collectDiagnosticAnomalies(
    renderer: Record<string, unknown>,
    capture: Record<string, unknown>): string[] {
    const anomalies: string[] = [];
    const shadows = asObject(renderer.shadows);
    const rendererState = asObject(renderer.renderer);
    const analysis = asObject(asObject(capture.analysis).channels);
    const color = asObject(analysis.color);
    const normals = asObject(analysis.normals);
    if (Number(color.invalidSamples) > 0) anomalies.push('Output color contains non-finite pixels');
    if (Number(color.darkFraction) > 0.85)
      anomalies.push(`Output is predominantly dark (${(Number(color.darkFraction) * 100).toFixed(1)}%)`);
    if (Number(color.brightFraction) > 0.85)
      anomalies.push(`Output is predominantly clipped bright (${(Number(color.brightFraction) * 100).toFixed(1)}%)`);
    if (Number(normals.invalidSamples) > 0) anomalies.push('World-normal capture contains non-finite pixels');
    if (Number(shadows.localEvictions) > 0)
      anomalies.push(`${Number(shadows.localEvictions)} local shadow allocations were evicted`);
    if (Number(shadows.localCacheMisses) > Number(shadows.localCacheHits) &&
        Number(shadows.localCacheMisses) > 4)
      anomalies.push('Local shadow cache is missing more often than it hits');
    if (typeof shadows.fallback === 'string' && shadows.fallback)
      anomalies.push(`Shadow fallback: ${shadows.fallback}`);
    if (Number(rendererState.renderScale) < 0.67)
      anomalies.push(`Render scale is low (${Number(rendererState.renderScale).toFixed(2)})`);
    return anomalies;
  }

  private async beginEdit(clientId: string, params: Record<string, unknown>): Promise<GatewayEditSession> {
    this.requireApproved(clientId);
    if (this.activeEdit) throw new Error(`Another edit session is active for ${this.activeEdit.clientId}`);
    if (this.viewportLease && this.viewportLease.clientId !== clientId)
      throw new Error(`Viewport control is currently leased to ${this.viewportLease.clientId}`);
    const expected = requireRevision(params.expectedSceneRevision);
    const transactionId = this.nextTransactionId++;
    const label = typeof params.label === 'string' && params.label.trim() ? params.label.trim() : 'AI Scene Edit';
    const response = await this.host.command('history.beginTransaction', { id: transactionId, label },
      undefined, expected);
    this.expect(response);
    const timestamp = now();
    this.activeEdit = {
      id: randomUUID(),
      transactionId,
      clientId,
      label,
      startedAt: timestamp,
      lastActivityAt: timestamp,
      expectedSceneRevision: response.sceneRevision,
    };
    this.notify();
    return this.activeEdit;
  }

  private async applyEdit(clientId: string, params: Record<string, unknown>): Promise<unknown> {
    const session = this.requireSession(requireString(params.editSessionId, 'editSessionId'), clientId);
    const expected = requireRevision(params.expectedSceneRevision);
    if (expected !== session.expectedSceneRevision) {
      throw new Error(`Edit session expects scene revision ${session.expectedSceneRevision}`);
    }
    const action = requireString(params.action, 'action');
    const value = asObject(params.value);
    const { type, payload } = await this.editCommand(action, value);
    const response = await this.host.command(type, payload, {
      id: session.transactionId,
      phase: 'update',
      label: session.label,
    }, expected);
    const result = this.expect(response);
    session.expectedSceneRevision = response.sceneRevision;
    session.lastActivityAt = now();
    this.approvedClients.set(clientId, Date.now() + editIdleMilliseconds);
    this.notify();
    return result;
  }

  private async editCommand(action: string, value: Record<string, unknown>):
    Promise<{ type: string; payload: Record<string, unknown> }> {
    if (action === 'create') {
      const payload: Record<string, unknown> = {
        kind: typeof value.kind === 'string' ? value.kind : 'empty',
      };
      if (typeof value.parentGuid === 'string' && value.parentGuid) {
        payload.parent = await this.resolveEntity(value.parentGuid);
      }
      return { type: 'entity.create', payload };
    }
    const guid = requireString(value.guid, 'value.guid');
    const entity = await this.resolveEntity(guid);
    if (action === 'rename') return { type: 'entity.rename', payload: { entity, name: value.name } };
    if (action === 'setActive') return { type: 'entity.setActive', payload: { entity, active: value.active } };
    if (action === 'setTag') return { type: 'entity.setTag', payload: { entity, tag: value.tag } };
    if (action === 'setMobility') return { type: 'entity.setMobility', payload: { entity, mobility: value.mobility } };
    if (action === 'setTransform') return { type: 'entity.setTransform', payload: { entity, transform: value.transform } };
    if (action === 'setMaterial') {
      return {
        type: 'entity.setMaterial',
        payload: { entity, path: requireProjectAssetPath(value.path, 'value.path') },
      };
    }
    if (action === 'delete') return { type: 'entity.delete', payload: { entity } };
    if (action === 'duplicate') return { type: 'entity.duplicate', payload: { entity } };
    if (action === 'reparent') {
      const parent = typeof value.parentGuid === 'string' && value.parentGuid
        ? await this.resolveEntity(value.parentGuid) : undefined;
      return {
        type: 'entity.reparent',
        payload: { entity, parent, preserveWorld: value.preserveWorld !== false },
      };
    }
    if (action === 'patchComponent') {
      const component = requireString(value.component, 'value.component').toLowerCase();
      const fields = asObject(value.fields);
      const snapshot = asObject(this.expect(await this.host.query('gateway.entity', { guid })));
      if (component === 'transform') {
        return {
          type: 'entity.setTransform',
          payload: { entity, transform: { ...asObject(snapshot.transform), ...fields } },
        };
      }
      if (component === 'camera') {
        return {
          type: 'entity.setCamera',
          payload: { entity, camera: { ...asObject(snapshot.camera), ...fields } },
        };
      }
      if (component === 'directionallight' || component === 'pointlight' ||
          component === 'spotlight' || component === 'arealight' || component === 'light') {
        return {
          type: 'entity.setLight',
          payload: { entity, light: { ...asObject(snapshot.light), ...fields } },
        };
      }
      if (component === 'meshrenderer') {
        return {
          type: 'entity.setMeshRenderer',
          payload: { entity, ...asObject(snapshot.meshRenderer), ...fields },
        };
      }
      if (component === 'terrain') {
        return {
          type: 'terrain.update',
          payload: { entity, ...asObject(snapshot.terrain), ...fields },
        };
      }
      if (component === 'worldenvironment') {
        const current = this.expect(await this.host.query('environment.state', { entity }));
        return {
          type: 'environment.update',
          payload: { entity, environment: { ...asObject(current), ...fields } },
        };
      }
      throw new Error(`Component ${String(value.component)} does not have a gateway patch binder`);
    }
    throw new Error(`Unsupported scene edit action: ${action}`);
  }

  private async commitEdit(editSessionId: string, clientId: string, expected: number): Promise<unknown> {
    const session = this.requireSession(editSessionId, clientId);
    if (expected !== session.expectedSceneRevision) {
      throw new Error(`Edit session expects scene revision ${session.expectedSceneRevision}`);
    }
    const response = await this.host.command('history.commitTransaction',
      { id: session.transactionId }, undefined, expected);
    const result = this.expect(response);
    this.lastCommittedEdit = {
      clientId,
      label: session.label,
      sceneRevision: response.sceneRevision,
      committedAt: now(),
    };
    this.activeEdit = null;
    this.notify();
    return result;
  }

  private async cancelEdit(editSessionId: string, clientId: string): Promise<unknown> {
    const session = this.requireSession(editSessionId, clientId);
    const response = await this.host.command('history.cancelTransaction', { id: session.transactionId });
    const result = this.expect(response);
    this.activeEdit = null;
    this.notify();
    return result;
  }

  private requireApproved(clientId: string): void {
    this.expirePermissions();
    const expires = this.approvedClients.get(clientId);
    if (!expires || expires <= Date.now()) {
      throw new Error('scene.edit permission has not been approved or has expired');
    }
  }

  private requireSession(id: string, clientId: string): GatewayEditSession {
    this.requireApproved(clientId);
    if (!this.activeEdit || this.activeEdit.id !== id || this.activeEdit.clientId !== clientId) {
      throw new Error('Edit session is not active for this client');
    }
    return this.activeEdit;
  }

  private async resolveEntity(guid: string): Promise<Record<string, unknown>> {
    const response = await this.host.query('gateway.entity', { guid });
    const entity = asObject(this.expect(response)).entity;
    if (!entity || typeof entity !== 'object') throw new Error(`Entity ${guid} was not found`);
    return entity as Record<string, unknown>;
  }

  private expect(response: GatewayHostResponse): unknown {
    this.sceneRevision = response?.sceneRevision ?? this.sceneRevision;
    this.worldEpoch = response?.worldEpoch ?? this.worldEpoch;
    this.frameRevision = response?.frameRevision ?? this.frameRevision;
    if (!response?.succeeded) throw new Error(response?.error || 'Native host request failed');
    return {
      ...(asObject(response.payload)),
      sceneRevision: response.sceneRevision,
      worldEpoch: response.worldEpoch,
      frameRevision: response.frameRevision,
    };
  }

  private expirePermissions(): boolean {
    const timestamp = Date.now();
    let changed = false;
    for (const [clientId, expires] of this.approvedClients) {
      if (expires <= timestamp) {
        this.approvedClients.delete(clientId);
        changed = true;
      }
    }
    if (this.viewportLease && this.viewportLease.expiresAt <= timestamp) {
      this.viewportLease = null;
      changed = true;
    }
    return changed;
  }

  async expireInactiveAuthority(): Promise<void> {
    const timestamp = Date.now();
    let changed = false;
    const active = this.activeEdit;
    if (active && Date.parse(active.lastActivityAt) + editIdleMilliseconds <= timestamp) {
      try {
        this.expect(await this.host.command('history.cancelTransaction', { id: active.transactionId }));
      } finally {
        this.activeEdit = null;
        this.approvedClients.delete(active.clientId);
        this.audit(active.clientId, 'security', 'edit.expire', true, active.label);
        changed = true;
      }
    }
    changed = this.expirePermissions() || changed;
    for (const [clientId, client] of this.clients) {
      if (Date.parse(client.lastSeenAt) + editIdleMilliseconds > timestamp)
        continue;
      this.clients.delete(clientId);
      this.audit(clientId, 'connection', 'disconnect.timeout', true, '');
      changed = true;
    }
    if (changed) this.notify();
  }

  async undoLastCommittedEdit(): Promise<unknown> {
    const committed = this.lastCommittedEdit;
    if (!committed) throw new Error('There is no committed AI edit to undo');
    const response = await this.host.command('history.undo', {}, undefined, committed.sceneRevision);
    const result = this.expect(response);
    this.audit('editor', 'edit', 'edit.undoLastCommitted', true, committed.label);
    this.lastCommittedEdit = null;
    this.notify();
    return result;
  }

  private claimViewport(clientId: string): void {
    this.expirePermissions();
    if (this.activeEdit && this.activeEdit.clientId !== clientId)
      throw new Error(`Scene editing is currently leased to ${this.activeEdit.clientId}`);
    if (this.viewportLease && this.viewportLease.clientId !== clientId)
      throw new Error(`Viewport control is currently leased to ${this.viewportLease.clientId}`);
    this.viewportLease = { clientId, expiresAt: Date.now() + viewportLeaseMilliseconds };
    this.notify();
  }

  private publicStatus(): Omit<GatewayStatus, 'audit' | 'discoveryFile'> {
    const status = this.status();
    return {
      enabled: status.enabled,
      endpoint: status.endpoint,
      protocolVersion: status.protocolVersion,
      sceneRevision: status.sceneRevision,
      worldEpoch: status.worldEpoch,
      frameRevision: status.frameRevision,
      eventSequence: status.eventSequence,
      clients: status.clients,
      pendingEditRequests: status.pendingEditRequests,
      activeEditSession: status.activeEditSession,
      lastCommittedEdit: status.lastCommittedEdit,
      viewportLease: status.viewportLease,
    };
  }

  private audit(clientId: string, category: GatewayAuditEntry['category'], operation: string,
    succeeded: boolean, detail: string): void {
    this.auditEntries.push({
      sequence: ++this.auditSequence,
      timestamp: now(),
      clientId,
      category,
      operation,
      succeeded,
      detail,
    });
    if (this.auditEntries.length > maximumAuditEntries) {
      this.auditEntries.splice(0, this.auditEntries.length - maximumAuditEntries);
    }
  }

  private appendEvent(event: Omit<GatewayEvent, 'sequence' | 'timestamp'>): void {
    const sequenced: GatewayEvent = {
      ...event,
      sequence: ++this.eventSequence,
      timestamp: now(),
    };
    this.recentEvents.push(sequenced);
    if (this.recentEvents.length > maximumGatewayEvents)
      this.recentEvents.splice(0, this.recentEvents.length - maximumGatewayEvents);
    for (const waiter of this.eventWaiters) waiter(sequenced);
    for (const listener of this.eventListeners) listener(sequenced);
  }

  private notify(): void {
    const status = this.status();
    for (const listener of this.listeners) listener(status);
  }
}

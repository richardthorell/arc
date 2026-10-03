import {
  AI_PROJECT_CONTEXT_SCHEMA_VERSION,
  type AiContextCollectionEvent,
  type AiContextCostEstimate,
  type AiContextJsonValue,
  type AiContextRevision,
  type AiContextSection,
  type AiContextSectionId,
  type AiProjectContextSnapshot,
} from '../../../common/aiContextTypes';

export type AiProjectContextHostEvent = {
  sequence?: number;
  type: string;
  message?: string;
  payload?: unknown;
};

export type AiProjectContextEnvironment = {
  projectSnapshot: () => Promise<unknown>;
  hostQuery: (type: string, payload?: Record<string, unknown>) => Promise<unknown>;
  subscribeHostEvents?: (listener: (event: AiProjectContextHostEvent) => void) => () => void;
  now?: () => number;
};

export type AiContextProviderResult = {
  status?: AiContextSection['status'];
  data?: unknown;
  error?: string;
  revision?: AiContextRevision;
};

export type AiContextProviderContext = {
  environment: AiProjectContextEnvironment;
  projectGuid: string | null;
  projectSnapshot: unknown;
  recentChanges: readonly AiProjectContextHostEvent[];
};

export interface AiContextProvider {
  readonly id: AiContextSectionId;
  collect(context: AiContextProviderContext): Promise<AiContextProviderResult>;
}

export type AiContextLimits = {
  maxDepth: number;
  maxArrayItems: number;
  maxObjectKeys: number;
  maxStringLength: number;
};

export type AiProjectContextServiceOptions = {
  providers?: readonly AiContextProvider[];
  maxAgeMs?: number;
  recentChangeLimit?: number;
  limits?: Partial<AiContextLimits>;
};

export type AiProjectContextCollectOptions = {
  forceRefresh?: boolean;
};

const defaultLimits: AiContextLimits = {
  maxDepth: 8,
  maxArrayItems: 128,
  maxObjectKeys: 128,
  maxStringLength: 4096,
};

const asRecord = (value: unknown): Record<string, unknown> | null =>
  value !== null && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : null;

const stringValue = (value: unknown): string | undefined => (typeof value === 'string' ? value : undefined);
const numberValue = (value: unknown): number | undefined =>
  typeof value === 'number' && Number.isFinite(value) ? value : undefined;

const projectGuidFromSnapshot = (snapshot: unknown): string | null => {
  const activeProject = asRecord(asRecord(snapshot)?.activeProject);
  const descriptor = asRecord(activeProject?.descriptor);
  const guid = stringValue(descriptor?.guid)?.trim();
  return guid || null;
};

const stableAssetReference = (value: unknown): Record<string, unknown> | null => {
  const reference = asRecord(value);
  if (!reference) return null;
  const guid = stringValue(reference.guid);
  const pathHint = stringValue(reference.pathHint);
  if (!guid && !pathHint) return null;
  return {
    guid: guid ?? '',
    expectedType: stringValue(reference.expectedType) ?? '',
    pathHint: pathHint ?? '',
  };
};

const stableProjectMetadata = (snapshot: unknown): unknown => {
  const activeProject = asRecord(asRecord(snapshot)?.activeProject);
  const descriptor = asRecord(activeProject?.descriptor);
  if (!activeProject || !descriptor || !stringValue(descriptor.guid)) return undefined;
  const startupScenes = Array.isArray(descriptor.startupScenes)
    ? descriptor.startupScenes.map(stableAssetReference).filter((value) => value !== null)
    : [];
  return {
    guid: descriptor.guid,
    name: stringValue(descriptor.name) ?? '',
    engineVersion: stringValue(descriptor.engineVersion) ?? '',
    compatibility: stringValue(activeProject.compatibility) ?? '',
    writable: typeof activeProject.writable === 'boolean' ? activeProject.writable : undefined,
    diagnostics: Array.isArray(activeProject.diagnostics) ? activeProject.diagnostics : [],
    defaultScene: stableAssetReference(descriptor.defaultScene),
    startupScenes,
    targetPlatforms: Array.isArray(descriptor.targetPlatforms) ? descriptor.targetPlatforms : [],
    renderer: asRecord(descriptor.renderer) ?? {},
  };
};

const hostRevision = (response: Record<string, unknown>): AiContextRevision | undefined => {
  const revision: AiContextRevision = {
    sceneRevision: numberValue(response.sceneRevision),
    worldEpoch: numberValue(response.worldEpoch),
    frameRevision: numberValue(response.frameRevision),
  };
  return Object.values(revision).some((value) => value !== undefined) ? revision : undefined;
};

const stableHostValue = (value: unknown, seen = new WeakMap<object, unknown>()): unknown => {
  if (Array.isArray(value)) return value.map((entry) => stableHostValue(entry, seen));
  const record = asRecord(value);
  if (!record) return value;
  const existing = seen.get(record);
  if (existing) return existing;
  const result: Record<string, unknown> = {};
  seen.set(record, result);
  const hasStableEntityGuid = typeof record.guid === 'string' && record.guid.length > 0;
  for (const [key, child] of Object.entries(record)) {
    if (hasStableEntityGuid && key === 'entity') continue;
    result[key] = stableHostValue(child, seen);
  }
  return result;
};

const collectHostQuery = async (
  context: AiContextProviderContext,
  queryType: string,
  payload: Record<string, unknown> = {},
): Promise<AiContextProviderResult> => {
  if (!context.projectGuid) return { status: 'unavailable' };
  const raw = await context.environment.hostQuery(queryType, payload);
  const response = asRecord(raw);
  if (!response) return { status: 'unavailable' };
  if (response.succeeded === false) {
    return {
      status: 'error',
      error: stringValue(response.error) ?? `${queryType} query failed`,
      revision: hostRevision(response),
    };
  }
  return {
    status: 'ready',
    data: stableHostValue(response.payload ?? response),
    revision: hostRevision(response),
  };
};

export const createDefaultAiContextProviders = (): AiContextProvider[] => [
  {
    id: 'project',
    collect: async ({ projectSnapshot }) => {
      const data = stableProjectMetadata(projectSnapshot);
      return data === undefined ? { status: 'unavailable' } : { status: 'ready', data };
    },
  },
  {
    id: 'scene',
    collect: (context) => collectHostQuery(context, 'scene.hierarchy'),
  },
  {
    id: 'selection',
    collect: (context) => collectHostQuery(context, 'entity.selected'),
  },
  {
    id: 'workspace',
    collect: (context) => collectHostQuery(context, 'workspace.documents'),
  },
  {
    id: 'diagnostics',
    collect: (context) => collectHostQuery(context, 'gateway.diagnostics'),
  },
  {
    id: 'viewport',
    collect: (context) => collectHostQuery(context, 'viewport.state', { viewportId: 'viewport-1' }),
  },
  {
    id: 'recentChanges',
    collect: async ({ projectGuid, recentChanges }) =>
      projectGuid
        ? {
            status: 'ready',
            data: recentChanges.map((event) => ({
              sequence: event.sequence,
              type: event.type,
              message: event.message,
              payload: stableHostValue(event.payload),
            })),
            revision: recentChanges.length
              ? { eventSequence: recentChanges[recentChanges.length - 1]?.sequence }
              : undefined,
          }
        : { status: 'unavailable' },
  },
];

type SanitizeResult = {
  value: AiContextJsonValue;
  truncated: boolean;
};

export const sanitizeAiContextValue = (input: unknown, limits: Partial<AiContextLimits> = {}): SanitizeResult => {
  const effective = { ...defaultLimits, ...limits };
  const ancestors = new WeakSet<object>();
  let truncated = false;

  const visit = (value: unknown, depth: number): AiContextJsonValue => {
    if (value === null || typeof value === 'boolean') return value;
    if (typeof value === 'number') {
      if (Number.isFinite(value)) return value;
      truncated = true;
      return String(value);
    }
    if (typeof value === 'string') {
      if (value.length <= effective.maxStringLength) return value;
      truncated = true;
      return `${value.slice(0, Math.max(0, effective.maxStringLength - 1))}…`;
    }
    if (typeof value === 'bigint') return value.toString();
    if (value instanceof Date) return value.toISOString();
    if (typeof value !== 'object' || value === undefined) {
      if (value !== undefined) truncated = true;
      return null;
    }
    if (depth >= effective.maxDepth) {
      truncated = true;
      return '[Max depth]';
    }
    if (ancestors.has(value)) {
      truncated = true;
      return '[Circular]';
    }

    ancestors.add(value);
    if (Array.isArray(value)) {
      if (value.length > effective.maxArrayItems) truncated = true;
      const result = value.slice(0, effective.maxArrayItems).map((entry) => visit(entry, depth + 1));
      ancestors.delete(value);
      return result;
    }

    const source = value as Record<string, unknown>;
    const keys = Object.keys(source).sort();
    if (keys.length > effective.maxObjectKeys) truncated = true;
    const result: Record<string, AiContextJsonValue> = {};
    for (const key of keys.slice(0, effective.maxObjectKeys)) {
      if (source[key] === undefined || typeof source[key] === 'function' || typeof source[key] === 'symbol') {
        truncated = true;
        continue;
      }
      result[key] = visit(source[key], depth + 1);
    }
    ancestors.delete(value);
    return result;
  };

  return { value: visit(input, 0), truncated };
};

const estimateCost = (value: AiContextJsonValue | undefined): AiContextCostEstimate => {
  const characters = value === undefined ? 0 : JSON.stringify(value).length;
  return { characters, approximateTokens: Math.ceil(characters / 4) };
};

const mergeRevision = (target: AiContextRevision, source: AiContextRevision | undefined): void => {
  if (!source) return;
  for (const key of ['sceneRevision', 'worldEpoch', 'frameRevision', 'eventSequence'] as const) {
    const value = source[key];
    if (value === undefined) continue;
    target[key] = Math.max(target[key] ?? 0, value);
  }
};

const normalizedEventType = (type: string): string => type.toLocaleLowerCase().replaceAll('_', '.');

const shouldRecordChange = (event: AiProjectContextHostEvent): boolean => {
  const type = normalizedEventType(event.type);
  return !type.includes('frame.ready') && !type.includes('profiler') && !type.includes('tick.completed');
};

const invalidatedProviderIds = (event: AiProjectContextHostEvent): AiContextSectionId[] | null => {
  const type = normalizedEventType(event.type);
  if (type.includes('project.open') || type.includes('project.close')) return null;

  const ids = new Set<AiContextSectionId>(['recentChanges']);
  if (type.includes('scene') || type.includes('entity') || type.includes('component')) {
    ids.add('scene');
    ids.add('selection');
  }
  if (type.includes('asset')) {
    ids.add('workspace');
    ids.add('diagnostics');
  }
  if (type.includes('viewport')) ids.add('viewport');
  if (type.includes('failed') || type.includes('fault') || type.includes('error')) ids.add('diagnostics');
  return [...ids];
};

export class AiProjectContextService {
  private readonly providers: readonly AiContextProvider[];
  private readonly limits: AiContextLimits;
  private readonly maxAgeMs: number;
  private readonly recentChangeLimit: number;
  private readonly cache = new Map<
    AiContextSectionId,
    { projectGuid: string | null; capturedAtMs: number; section: AiContextSection }
  >();
  private readonly listeners = new Set<(event: AiContextCollectionEvent) => void>();
  private recentChanges: AiProjectContextHostEvent[] = [];
  private lastProjectGuid: string | null | undefined;
  private latestEventSequence: number | undefined;
  private nextCollectionId = 1;
  private unsubscribeHostEvents: (() => void) | undefined;

  constructor(
    private readonly environment: AiProjectContextEnvironment,
    options: AiProjectContextServiceOptions = {},
  ) {
    this.providers = options.providers ?? createDefaultAiContextProviders();
    this.maxAgeMs = Math.max(0, options.maxAgeMs ?? 750);
    this.recentChangeLimit = Math.max(1, options.recentChangeLimit ?? 24);
    this.limits = { ...defaultLimits, ...options.limits };
    this.unsubscribeHostEvents = environment.subscribeHostEvents?.((event) => this.handleHostEvent(event));
  }

  subscribe(listener: (event: AiContextCollectionEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  invalidate(providerIds: readonly AiContextSectionId[] | null = null, reason = 'manual'): void {
    if (providerIds === null) this.cache.clear();
    else for (const id of providerIds) this.cache.delete(id);
    this.emit({
      type: 'context.invalidated',
      providerIds: providerIds === null ? null : [...providerIds],
      reason,
      timestamp: new Date(this.now()).toISOString(),
    });
  }

  async collect(options: AiProjectContextCollectOptions = {}): Promise<AiProjectContextSnapshot> {
    const startedAt = this.now();
    const projectSnapshot = await this.environment.projectSnapshot();
    const projectGuid = projectGuidFromSnapshot(projectSnapshot);
    if (this.lastProjectGuid !== undefined && this.lastProjectGuid !== projectGuid) {
      this.recentChanges = [];
      this.latestEventSequence = undefined;
      this.invalidate(null, 'project.changed');
    }
    this.lastProjectGuid = projectGuid;

    const collectionId = `${projectGuid ?? 'no-project'}:${this.nextCollectionId++}`;
    this.emit({
      type: 'collection.started',
      collectionId,
      projectGuid,
      timestamp: new Date(startedAt).toISOString(),
    });

    const providerContext: AiContextProviderContext = {
      environment: this.environment,
      projectGuid,
      projectSnapshot,
      recentChanges: [...this.recentChanges],
    };
    const sections = await Promise.all(
      this.providers.map((provider) =>
        this.collectProvider(provider, providerContext, collectionId, options.forceRefresh),
      ),
    );
    const revision: AiContextRevision = {};
    let characters = 0;
    let approximateTokens = 0;
    for (const section of sections) {
      mergeRevision(revision, section.freshness.revision);
      characters += section.estimatedCost.characters;
      approximateTokens += section.estimatedCost.approximateTokens;
    }
    if (this.latestEventSequence !== undefined) {
      revision.eventSequence = Math.max(revision.eventSequence ?? 0, this.latestEventSequence);
    }
    const estimatedCost = { characters, approximateTokens };
    const completedAt = this.now();
    const snapshot: AiProjectContextSnapshot = {
      schemaVersion: AI_PROJECT_CONTEXT_SCHEMA_VERSION,
      collectionId,
      projectGuid,
      capturedAt: new Date(completedAt).toISOString(),
      revision,
      sections,
      estimatedCost,
    };
    this.emit({
      type: 'collection.completed',
      collectionId,
      projectGuid,
      estimatedCost,
      durationMs: Math.max(0, completedAt - startedAt),
      timestamp: snapshot.capturedAt,
    });
    return snapshot;
  }

  dispose(): void {
    this.unsubscribeHostEvents?.();
    this.unsubscribeHostEvents = undefined;
    this.cache.clear();
    this.listeners.clear();
  }

  private async collectProvider(
    provider: AiContextProvider,
    context: AiContextProviderContext,
    collectionId: string,
    forceRefresh = false,
  ): Promise<AiContextSection> {
    const startedAt = this.now();
    const cached = this.cache.get(provider.id);
    if (
      !forceRefresh &&
      cached &&
      cached.projectGuid === context.projectGuid &&
      startedAt - cached.capturedAtMs <= this.maxAgeMs
    ) {
      const section: AiContextSection = {
        ...cached.section,
        freshness: {
          ...cached.section.freshness,
          ageMs: Math.max(0, startedAt - cached.capturedAtMs),
          cache: 'cached',
        },
      };
      this.emitProviderCompleted(collectionId, provider.id, section, 0);
      return section;
    }

    let result: AiContextProviderResult;
    try {
      result = await provider.collect(context);
    } catch (error) {
      result = {
        status: 'error',
        error: error instanceof Error ? error.message : String(error),
      };
    }
    const completedAt = this.now();
    const status = result.status ?? (result.data === undefined ? 'unavailable' : 'ready');
    const sanitized = result.data === undefined ? undefined : sanitizeAiContextValue(result.data, this.limits);
    const section: AiContextSection = {
      id: provider.id,
      status,
      data: sanitized?.value,
      error: result.error,
      truncated: sanitized?.truncated ?? false,
      freshness: {
        capturedAt: new Date(completedAt).toISOString(),
        ageMs: 0,
        cache: 'live',
        revision: result.revision,
      },
      estimatedCost: estimateCost(sanitized?.value),
    };
    if (status !== 'error') {
      this.cache.set(provider.id, { projectGuid: context.projectGuid, capturedAtMs: completedAt, section });
    }
    this.emitProviderCompleted(collectionId, provider.id, section, Math.max(0, completedAt - startedAt));
    return section;
  }

  private handleHostEvent(event: AiProjectContextHostEvent): void {
    if (typeof event.sequence === 'number' && Number.isFinite(event.sequence)) {
      this.latestEventSequence = Math.max(this.latestEventSequence ?? 0, event.sequence);
    }
    const type = normalizedEventType(event.type);
    if (type.includes('project.open') || type.includes('project.close')) this.recentChanges = [];
    if (shouldRecordChange(event)) {
      this.recentChanges.push(event);
      if (this.recentChanges.length > this.recentChangeLimit) {
        this.recentChanges.splice(0, this.recentChanges.length - this.recentChangeLimit);
      }
    }
    this.invalidate(invalidatedProviderIds(event), `host.${event.type}`);
  }

  private emitProviderCompleted(
    collectionId: string,
    providerId: AiContextSectionId,
    section: AiContextSection,
    durationMs: number,
  ): void {
    this.emit({
      type: 'provider.completed',
      collectionId,
      providerId,
      status: section.status,
      durationMs,
      cache: section.freshness.cache,
      timestamp: new Date(this.now()).toISOString(),
    });
  }

  private emit(event: AiContextCollectionEvent): void {
    for (const listener of this.listeners) listener(event);
  }

  private now(): number {
    return this.environment.now?.() ?? Date.now();
  }
}

export const createWindowAiProjectContextEnvironment = (): AiProjectContextEnvironment => ({
  projectSnapshot: () => window.arc.projects.snapshot(),
  hostQuery: (type, payload = {}) => window.arc.host?.query?.(type, payload) ?? Promise.resolve(undefined),
  subscribeHostEvents: (listener) =>
    window.arc.host?.onEvent?.((event) =>
      listener({
        sequence: event.sequence,
        type: event.type,
        message: event.message,
        payload: event.payload,
      }),
    ) ?? (() => undefined),
});

export const createWindowAiProjectContextService = (
  options: AiProjectContextServiceOptions = {},
): AiProjectContextService => new AiProjectContextService(createWindowAiProjectContextEnvironment(), options);

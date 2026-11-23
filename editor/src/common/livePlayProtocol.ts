import { z } from 'zod';

/**
 * L1 control/inspection plane. This is NOT C9 cooked-content transfer, the
 * local EditorHost protocol, or gameplay networking. See docs/live-play-control-protocol.md.
 */
export const LIVE_PLAY_PROTOCOL = 'arc.live-play';
export const LIVE_PLAY_VERSION = 1;
export const LIVE_PLAY_MAX_FRAME_BYTES = 64 * 1024;

const id = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z0-9._:-]+$/u);
const guid = z.string().regex(/^(?:[a-f\d]{32}|[a-f\d]{8}(?:-[a-f\d]{4}){3}-[a-f\d]{12})$/iu);
const revision = z.number().int().nonnegative().max(Number.MAX_SAFE_INTEGER);
const positiveSequence = revision.refine((value) => value > 0, 'Sequence must be positive');
const schemaHash = z.string().regex(/^[a-f\d]{64}$/u);

export const livePlayCapabilitySchema = z.enum([
  'session.observe',
  'session.control',
  'entity.inspect',
  'entity.edit',
  'content.observe',
]);
export type LivePlayCapability = z.infer<typeof livePlayCapabilitySchema>;

export const livePlayEntitySchema = z.discriminatedUnion('kind', [
  z.object({ kind: z.literal('authored'), entityGuid: guid }).strict(),
  z.object({ kind: z.literal('runtime'), runtimeEntityId: id }).strict(),
]);
export type LivePlayEntity = z.infer<typeof livePlayEntitySchema>;

/** Renderer-to-main inspection requests are strictly read-only and whitelist known native schemas. */
export const livePlayInspectPayloadSchema = z
  .object({
    entity: livePlayEntitySchema,
    component: z
      .object({
        typeId: z.enum(['arc.transform', 'arc.scene.name', 'arc.scene.tag', 'arc.scene.active']),
        schemaVersion: z.literal(1),
      })
      .strict(),
  })
  .strict();

const component = z.object({ typeId: id, schemaVersion: z.number().int().positive().max(65535) }).strict();
const propertyPath = z
  .string()
  .min(1)
  .max(128)
  .regex(/^[A-Za-z][A-Za-z0-9_.]*$/u);
const value = z.union([
  z.string().max(2048),
  z.number().finite(),
  z.boolean(),
  z.null(),
  z.array(z.number().finite()).max(16),
]);
const hierarchyEntity = z
  .object({
    entity: livePlayEntitySchema,
    parent: livePlayEntitySchema.nullable(),
    name: z.string().max(256),
  })
  .strict();
const errorCode = z.enum([
  'unauthenticated',
  'unsupported_version',
  'identity_mismatch',
  'schema_mismatch',
  'permission_denied',
  'unsupported_operation',
  'unknown_entity',
  'stale_revision',
  'restart_required',
  'invalid_request',
  'session_expired',
  'resync_required',
  'busy',
]);
const error = z.object({ code: errorCode, message: z.string().max(512) }).strict();
const capabilities = z
  .array(livePlayCapabilitySchema)
  .max(5)
  .refine((items) => new Set(items).size === items.length);

const envelope = z
  .object({
    protocol: z.literal(LIVE_PLAY_PROTOCOL),
    protocolVersion: z.literal(LIVE_PLAY_VERSION),
    sender: z.enum(['editor', 'runtime']),
    projectGuid: guid,
    buildId: id,
    targetProfile: id,
    sessionId: id,
    instanceId: id,
    messageId: id,
    sequence: positiveSequence,
  })
  .strict();

const message = <T extends string, P extends z.ZodTypeAny>(type: T, payload: P) =>
  envelope.extend({ type: z.literal(type), payload }).strict();

export const livePlayMessageSchema = z.discriminatedUnion('type', [
  message(
    'hello',
    z
      .object({
        schemaHash,
        lastReceivedSequence: revision,
        capabilities,
      })
      .strict(),
  ),
  message('hello.accepted', z.object({ capabilities, lastReceivedSequence: revision }).strict()),
  message('hello.rejected', z.object({ error }).strict()),
  message('heartbeat', z.object({ lastReceivedSequence: revision }).strict()),
  message(
    'session.state',
    z
      .object({
        state: z.enum(['starting', 'running', 'paused', 'stopped', 'faulted']),
        tick: revision,
        revision,
      })
      .strict(),
  ),
  message(
    'session.command',
    z
      .object({
        action: z.enum(['pause', 'resume', 'stop', 'step']),
        ticks: z.number().int().min(1).max(256).optional(),
      })
      .strict(),
  ),
  message(
    'entity.property.set',
    z
      .object({
        entity: livePlayEntitySchema,
        component,
        property: propertyPath,
        value,
        authoringRevision: revision,
        expectedRuntimeRevision: revision,
        phase: z.enum(['begin', 'update', 'commit', 'cancel']),
      })
      .strict(),
  ),
  message(
    'entity.create',
    z
      .object({
        entityGuid: guid,
        parent: livePlayEntitySchema.nullable(),
        archetype: id,
        authoringRevision: revision,
      })
      .strict(),
  ),
  message('entity.delete', z.object({ entity: livePlayEntitySchema, expectedRuntimeRevision: revision }).strict()),
  message(
    'entity.reparent',
    z
      .object({
        entity: livePlayEntitySchema,
        parent: livePlayEntitySchema.nullable(),
        expectedRuntimeRevision: revision,
      })
      .strict(),
  ),
  message(
    'hierarchy.snapshot',
    z
      .object({
        worldRevision: revision,
        entities: z.array(hierarchyEntity).max(128),
        complete: z.boolean(),
        // L1.5d: optional bounded pages; a page is NEVER independently authoritative.
        pageIndex: z.number().int().min(0).max(63).optional(),
        pageCount: z.number().int().min(1).max(64).optional(),
      })
      .strict(),
  ),
  message(
    'hierarchy.changed',
    z
      .object({
        // The receiver must have this exact base. Unknown bases require a new snapshot.
        baseRevision: revision.optional(),
        worldRevision: revision,
        changes: z
          .array(
            z.discriminatedUnion('kind', [
              z.object({ kind: z.literal('upsert'), value: hierarchyEntity }).strict(),
              z.object({ kind: z.literal('removed'), entity: livePlayEntitySchema }).strict(),
            ]),
          )
          .max(128),
      })
      .strict(),
  ),
  message('component.inspect.request', z.object({ entity: livePlayEntitySchema, component }).strict()),
  message(
    'component.inspect.result',
    z
      .object({
        entity: livePlayEntitySchema,
        component,
        requestId: id.optional(),
        runtimeRevision: revision,
        properties: z.array(z.object({ path: propertyPath, value }).strict()).max(64),
      })
      .strict(),
  ),
  message(
    'log.record',
    z.object({ level: z.enum(['debug', 'info', 'warning', 'error']), text: z.string().max(4096) }).strict(),
  ),
  message(
    'content.generation',
    z.object({ generation: revision, status: z.enum(['available', 'active', 'rejected']) }).strict(),
  ),
  // L1.3b1: metadata only. The editor must NOT send this until C9 verifies a
  // target-compatible cooked release. No source paths, bytes or tokens.
  message(
    'scene.baseline.offer',
    z
      .object({
        sceneGuid: guid,
        authoringRevision: revision,
        authoringSha256: schemaHash,
        cookedGeneration: revision,
        cookedReleaseSha256: schemaHash,
      })
      .strict(),
  ),
  message(
    'scene.baseline.status',
    z
      .object({
        requestId: id,
        sceneGuid: guid,
        cookedGeneration: revision,
        cookedReleaseSha256: schemaHash,
        status: z.enum(['active', 'rejected']),
        reason: z.enum(['none', 'unavailable', 'incompatible', 'corrupt', 'unsupported']),
      })
      .strict()
      .refine(
        (value) => (value.status === 'active') === (value.reason === 'none'),
        'An active baseline must have no rejection reason',
      ),
  ),
  message(
    'session.resync.request',
    z.object({ lastReceivedSequence: revision, reason: z.enum(['reconnected', 'gap', 'overflow']) }).strict(),
  ),
  message(
    'command.ack',
    z.discriminatedUnion('status', [
      z.object({ status: z.literal('applied'), requestId: id, appliedRevision: revision }).strict(),
      z.object({ status: z.literal('rejected'), requestId: id, error }).strict(),
    ]),
  ),
]);
export type LivePlayMessage = z.infer<typeof livePlayMessageSchema>;
export type LivePlayMessageType = LivePlayMessage['type'];

export type LivePlayAuthority = Pick<
  LivePlayMessage,
  'projectGuid' | 'buildId' | 'targetProfile' | 'sessionId' | 'instanceId'
>;

export type LivePlayReceiveContext = {
  authority: LivePlayAuthority;
  /** Connection authentication must complete before ANY frame is dispatched. */
  authenticated: boolean;
  /** Expected native component schema hash, pinned by the trusted launch context. */
  expectedSchemaHash: string;
  /** Role of the peer sending this frame, verified at connection handshake. */
  peerRole: 'editor' | 'runtime';
  /** Sender-specific highest accepted sequence; retained across reconnect. */
  lastSequence: number;
  /** Permissions granted by the session owner, never by a received hello. */
  grantedCapabilities: readonly LivePlayCapability[];
};

export type LivePlayRejectCode =
  | 'frame_too_large'
  | 'unsupported_version'
  | 'schema_mismatch'
  | 'invalid_frame'
  | 'unauthenticated'
  | 'identity_mismatch'
  | 'wrong_direction'
  | 'replayed_sequence'
  | 'permission_denied';

export type LivePlayReceiveResult =
  { accepted: true; message: LivePlayMessage; nextSequence: number } | { accepted: false; code: LivePlayRejectCode };

const editorOnly = new Set<LivePlayMessageType>([
  'session.command',
  'entity.property.set',
  'entity.create',
  'entity.delete',
  'entity.reparent',
  'component.inspect.request',
  'session.resync.request',
  'scene.baseline.offer',
]);
const runtimeOnly = new Set<LivePlayMessageType>([
  'session.state',
  'hierarchy.snapshot',
  'hierarchy.changed',
  'component.inspect.result',
  'log.record',
  'content.generation',
  'scene.baseline.status',
]);

function capabilityFor(type: LivePlayMessageType): LivePlayCapability | undefined {
  if (
    type === 'entity.property.set' ||
    type === 'entity.create' ||
    type === 'entity.delete' ||
    type === 'entity.reparent'
  )
    return 'entity.edit';
  if (type === 'component.inspect.request' || type === 'session.resync.request') return 'entity.inspect';
  if (type === 'session.command') return 'session.control';
  if (type === 'scene.baseline.offer' || type === 'scene.baseline.status') return 'content.observe';
  return undefined;
}

/** Parse and authorize without mutating caller state or dispatching native commands. */
export function receiveLivePlayFrame(frame: string, context: LivePlayReceiveContext): LivePlayReceiveResult {
  if (
    frame.length > LIVE_PLAY_MAX_FRAME_BYTES ||
    new TextEncoder().encode(frame).byteLength > LIVE_PLAY_MAX_FRAME_BYTES
  )
    return { accepted: false, code: 'frame_too_large' };
  if (!context.authenticated) return { accepted: false, code: 'unauthenticated' };

  let raw: unknown;
  try {
    raw = JSON.parse(frame) as unknown;
  } catch {
    return { accepted: false, code: 'invalid_frame' };
  }
  if (raw && typeof raw === 'object' && 'protocolVersion' in raw && raw.protocolVersion !== LIVE_PLAY_VERSION)
    return { accepted: false, code: 'unsupported_version' };
  const parsed = livePlayMessageSchema.safeParse(raw);
  if (!parsed.success) return { accepted: false, code: 'invalid_frame' };

  const item = parsed.data;
  if (item.type === 'hello' && item.payload.schemaHash !== context.expectedSchemaHash)
    return { accepted: false, code: 'schema_mismatch' };
  if (
    item.type === 'hello.accepted' &&
    item.payload.capabilities.some((cap) => !context.grantedCapabilities.includes(cap))
  )
    return { accepted: false, code: 'permission_denied' };
  const expected = context.authority;
  if (
    item.projectGuid.toLowerCase() !== expected.projectGuid.toLowerCase() ||
    item.buildId !== expected.buildId ||
    item.targetProfile !== expected.targetProfile ||
    item.sessionId !== expected.sessionId ||
    item.instanceId !== expected.instanceId
  )
    return { accepted: false, code: 'identity_mismatch' };
  if (item.sender !== context.peerRole) return { accepted: false, code: 'wrong_direction' };
  if (
    (item.sender === 'editor' && runtimeOnly.has(item.type)) ||
    (item.sender === 'runtime' && editorOnly.has(item.type))
  )
    return { accepted: false, code: 'wrong_direction' };
  if (!Number.isSafeInteger(context.lastSequence) || item.sequence <= context.lastSequence)
    return { accepted: false, code: 'replayed_sequence' };
  const required = capabilityFor(item.type);
  if (required && !context.grantedCapabilities.includes(required))
    return { accepted: false, code: 'permission_denied' };
  return { accepted: true, message: item, nextSequence: item.sequence };
}

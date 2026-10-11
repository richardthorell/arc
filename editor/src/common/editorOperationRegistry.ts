import { z } from 'zod';

import type { AiJsonObject } from './aiRuntimeTypes';

export type EditorOperationErrorCode =
  | 'unknown_operation'
  | 'unsupported_version'
  | 'stale_catalog'
  | 'unavailable'
  | 'permission_denied'
  | 'invalid_arguments'
  | 'invalid_result';

export class EditorOperationError extends Error {
  constructor(
    readonly code: EditorOperationErrorCode,
    message: string,
    readonly operation?: string,
  ) {
    super(message);
    this.name = 'EditorOperationError';
  }
}

export type EditorOperationContext = {
  capabilities: ReadonlySet<string>;
};

export type EditorOperationMetadata = {
  id: string;
  version: number;
  description: string;
  domain: string;
  permissions: readonly string[];
  requiredCapabilities: readonly string[];
  effects: { reads: readonly string[]; writes: readonly string[] };
  concurrency: 'independent-read' | 'serial';
  cache: 'revision' | 'never';
  transaction: 'none' | 'control' | 'participant' | 'legacy';
  references: readonly { field: string; resource: string; temporary: boolean }[];
  creates: readonly { field: string; resource: string }[];
};

export type EditorOperationDefinition<
  TContext extends EditorOperationContext,
  TInput = unknown,
  TResult = unknown,
> = EditorOperationMetadata & {
  input: z.ZodType<TInput>;
  output: z.ZodType<TResult>;
  available?: (context: TContext) => string | undefined;
  execute: (input: TInput, context: TContext) => TResult | Promise<TResult>;
};

export type EditorOperationDescriptor = EditorOperationMetadata & {
  available: boolean;
  unavailableReason?: string;
  inputSchema: AiJsonObject;
  outputSchema: AiJsonObject;
};

export const operationListRequest = z
  .object({
    domain: z.string().min(1).optional(),
    offset: z.number().int().nonnegative().default(0),
    limit: z.number().int().min(1).max(100).default(50),
    catalogRevision: z.string().min(1).optional(),
  })
  .strict();

export const operationDescribeRequest = z
  .object({
    ids: z.array(z.string().min(1)).min(1).max(16),
    catalogRevision: z.string().min(1).optional(),
  })
  .strict();

export const operationInvocationContract = z
  .object({
    version: z.number().int().positive(),
    catalogRevision: z.string().min(1),
  })
  .strict();

const operationId = /^[a-z][A-Za-z0-9]*(?:\.[a-z][A-Za-z0-9]*)+$/;
const schemaJson = (schema: z.ZodType, io: 'input' | 'output' = 'output'): AiJsonObject =>
  JSON.parse(JSON.stringify(z.toJSONSchema(schema, { io }))) as AiJsonObject;

export const operationWireSchema = (input: z.ZodType) => {
  if (!(input instanceof z.ZodObject)) throw new Error('Operation input must be an object schema');
  return input.safeExtend({
    operationContract: operationInvocationContract.optional(),
    clientName: z.string().optional(),
  });
};

/** Trusted registrations only. Serialized descriptors never carry executable code. */
export class EditorOperationRegistry<TContext extends EditorOperationContext = EditorOperationContext> {
  private readonly operations = new Map<
    string,
    {
      definition: EditorOperationDefinition<TContext>;
      descriptor: Omit<EditorOperationDescriptor, 'available' | 'unavailableReason'>;
    }
  >();
  private generation = 0;
  private readonly epoch = globalThis.crypto.randomUUID();

  constructor(
    private readonly authorize: (operation: EditorOperationMetadata, context: TContext) => void | Promise<void>,
  ) {}

  get revision(): string {
    return `${this.epoch}:${this.generation}`;
  }

  register<TInput, TResult>(definition: EditorOperationDefinition<TContext, TInput, TResult>): () => void {
    if (!operationId.test(definition.id)) throw new Error(`Operation id must be namespaced: ${definition.id}`);
    if (this.operations.has(definition.id)) throw new Error(`Operation already registered: ${definition.id}`);
    if (
      !Number.isSafeInteger(definition.version) ||
      definition.version < 1 ||
      !definition.domain.trim() ||
      !definition.description.trim()
    )
      throw new Error('Invalid operation metadata');
    if (
      definition.effects.writes.length &&
      (definition.concurrency !== 'serial' || definition.cache !== 'never' || !definition.permissions.length)
    )
      throw new Error('Writing operations require permissions, serial execution, and no read cache');
    const { input, output, available: _available, execute: _execute, ...metadata } = definition;
    void _available;
    void _execute;
    if (!(input instanceof z.ZodObject)) throw new Error('Operation input must be an object schema');
    if ('operationContract' in input.shape || 'clientName' in input.shape)
      throw new Error('Operation input uses a reserved transport envelope field');
    const stored = {
      definition: { ...definition } as EditorOperationDefinition<TContext>,
      descriptor: structuredClone({
        ...metadata,
        inputSchema: schemaJson(input, 'input'),
        outputSchema: schemaJson(output),
      }),
    };
    this.operations.set(definition.id, stored);
    ++this.generation;
    return () => {
      if (this.operations.get(definition.id) === stored) {
        this.operations.delete(definition.id);
        ++this.generation;
      }
    };
  }

  has(id: string): boolean {
    return this.operations.has(id);
  }

  checkRevision(revision?: string): void {
    if (revision !== undefined && revision !== this.revision)
      throw new EditorOperationError('stale_catalog', 'Operation catalog changed; rediscover the operation schema');
  }

  describe(id: string, context: TContext): EditorOperationDescriptor {
    const entry = this.require(id);
    const missing = entry.descriptor.requiredCapabilities.filter((capability) => !context.capabilities.has(capability));
    const reason = missing.length
      ? `Requires capabilities: ${missing.join(', ')}`
      : entry.definition.available?.(context);
    return {
      ...structuredClone(entry.descriptor),
      available: !reason,
      ...(reason ? { unavailableReason: reason } : {}),
    };
  }

  list(raw: unknown, context: TContext) {
    const request = operationListRequest.parse(raw);
    this.checkRevision(request.catalogRevision);
    const entries = [...this.operations.keys()]
      .sort()
      .map((id) => this.describe(id, context))
      .filter((entry) => !request.domain || entry.domain === request.domain);
    const operations = entries
      .slice(request.offset, request.offset + request.limit)
      .map(({ inputSchema: _input, outputSchema: _output, ...summary }) => summary);
    return {
      catalogRevision: this.revision,
      total: entries.length,
      operations,
      nextOffset: request.offset + operations.length < entries.length ? request.offset + operations.length : null,
    };
  }

  describeMany(raw: unknown, context: TContext) {
    const request = operationDescribeRequest.parse(raw);
    this.checkRevision(request.catalogRevision);
    return { catalogRevision: this.revision, operations: request.ids.map((id) => this.describe(id, context)) };
  }

  /** For trusted transport adapters; schemas still come from the registration. */
  transportEntries(context: TContext) {
    return [...this.operations.keys()].sort().map((id) => ({
      descriptor: this.describe(id, context),
      input: operationWireSchema(this.require(id).definition.input),
    }));
  }

  async execute(
    id: string,
    raw: unknown,
    context: TContext,
    contract?: z.infer<typeof operationInvocationContract>,
  ): Promise<unknown> {
    this.checkRevision(contract?.catalogRevision);
    const entry = this.require(id);
    const descriptor = this.describe(id, context);
    if ((contract?.version ?? 1) !== descriptor.version)
      throw new EditorOperationError('unsupported_version', `Unsupported version for ${id}`, id);
    if (!descriptor.available) throw new EditorOperationError('unavailable', descriptor.unavailableReason!, id);
    const input = entry.definition.input.safeParse(raw);
    if (!input.success)
      throw new EditorOperationError(
        'invalid_arguments',
        `Invalid arguments for ${id}: ${input.error.issues[0]?.message}`,
        id,
      );
    await this.authorize(descriptor, context);
    // Authorization may yield. Never execute a removed/replaced registration or a stale schema.
    this.checkRevision(contract?.catalogRevision);
    if (this.operations.get(id) !== entry)
      throw new EditorOperationError('stale_catalog', 'Operation registration changed', id);
    const current = this.describe(id, context);
    if (!current.available) throw new EditorOperationError('unavailable', current.unavailableReason!, id);
    const result = await entry.definition.execute(input.data, context);
    const parsed = entry.definition.output.safeParse(result);
    if (!parsed.success) throw new EditorOperationError('invalid_result', `Invalid result for ${id}`, id);
    return parsed.data;
  }

  private require(id: string) {
    const entry = this.operations.get(id);
    if (!entry) throw new EditorOperationError('unknown_operation', `Unknown editor operation: ${id}`, id);
    return entry;
  }
}

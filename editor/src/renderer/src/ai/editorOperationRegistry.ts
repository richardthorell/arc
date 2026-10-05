export type EditorOperationSchema<TInput> = {
  parse: (value: unknown) => TInput;
};

export type EditorOperationContext = {
  capabilities: ReadonlySet<string>;
};

export type EditorOperationDefinition<TInput = unknown, TResult = unknown> = {
  id: string;
  description: string;
  schema: EditorOperationSchema<TInput>;
  mutating: boolean;
  batchable: boolean;
  requiredCapabilities?: readonly string[];
  owner: string;
  execute: (input: TInput, context: EditorOperationContext) => TResult | Promise<TResult>;
};

export type EditorOperationSummary = Omit<EditorOperationDefinition<unknown, unknown>, 'schema' | 'execute'>;

const OPERATION_ID = /^[a-z][a-z0-9]*(?:\.[a-z][a-z0-9]*)+$/;

export class EditorOperationRegistry {
  private readonly operations = new Map<string, EditorOperationDefinition<unknown, unknown>>();

  register<TInput, TResult>(definition: EditorOperationDefinition<TInput, TResult>): () => void {
    if (!OPERATION_ID.test(definition.id)) {
      throw new Error(`Editor operation id must be a stable namespaced id: ${definition.id}`);
    }
    if (!definition.owner.trim()) throw new Error(`Editor operation ${definition.id} must declare an owner`);
    if (!definition.description.trim()) {
      throw new Error(`Editor operation ${definition.id} must declare a description`);
    }
    if (this.operations.has(definition.id)) {
      throw new Error(`Editor operation is already registered: ${definition.id}`);
    }

    const stored = definition as EditorOperationDefinition<unknown, unknown>;
    this.operations.set(definition.id, stored);
    return () => {
      if (this.operations.get(definition.id) === stored) this.operations.delete(definition.id);
    };
  }

  has(id: string): boolean {
    return this.operations.has(id);
  }

  list(): EditorOperationSummary[] {
    return [...this.operations.values()]
      .map(({ schema: _schema, execute: _execute, ...summary }) => ({ ...summary }))
      .sort((left, right) => left.id.localeCompare(right.id));
  }

  listAvailable(capabilities: ReadonlySet<string>): EditorOperationSummary[] {
    return this.list().filter((operation) =>
      (operation.requiredCapabilities ?? []).every((capability) => capabilities.has(capability)),
    );
  }

  async execute(id: string, input: unknown, context: EditorOperationContext): Promise<unknown> {
    const operation = this.operations.get(id);
    if (!operation) throw new Error(`Unknown editor operation: ${id}`);

    const missing = (operation.requiredCapabilities ?? []).filter(
      (capability) => !context.capabilities.has(capability),
    );
    if (missing.length > 0) {
      throw new Error(`Editor operation ${id} requires capabilities: ${missing.join(', ')}`);
    }

    const parsed = operation.schema.parse(input);
    return operation.execute(parsed, context);
  }
}

import { z } from 'zod';

import type { AiJsonObject, AiJsonValue, AiToolDefinition } from '../common/aiRuntimeTypes';
import { assertAiToolInvocationAllowed, type AiToolSecurityDescriptor } from '../common/aiSecurityPolicy';
import type { BuiltInAgentCapabilities, BuiltInAgentToolExecutionResult } from '../common/builtInAgentTypes';
import { type AgentEditAction, type AgentHarnessMethod } from './agentHarnessContract';
import { validateAgentEditValue } from './agentEditToolSchemas';
import type { BuiltInAgentAdapter } from './builtInAgentAdapter';
import type { EditorOperationDescriptor } from '../common/editorOperationRegistry';

export const BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES = 64 * 1024;
const maximumResultPreviewCharacters = 16 * 1024;

import { registryEntries, type RegistryEntry } from './agentOperationCatalog';

const entryByMethod = new Map<AgentHarnessMethod, RegistryEntry>(registryEntries.map((entry) => [entry.method, entry]));

const entryByName: ReadonlyMap<string, RegistryEntry> = new Map(registryEntries.map((entry) => [entry.method, entry]));

const toJsonValue = (value: unknown, path = 'result'): AiJsonValue => {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error(`${path} contains a non-finite number`);
    return value;
  }
  if (Array.isArray(value)) return value.map((entry, index) => toJsonValue(entry, `${path}[${String(index)}]`));
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, nested]) => [
        key,
        toJsonValue(nested, `${path}.${key}`),
      ]),
    );
  }
  throw new Error(`${path} contains a non-serializable ${typeof value}`);
};

const toJsonObject = (value: unknown, label: string): AiJsonObject => {
  const json = toJsonValue(value, label);
  if (!json || typeof json !== 'object' || Array.isArray(json)) throw new Error(`${label} must be an object`);
  return json as AiJsonObject;
};

const jsonSchema = (schema: z.ZodType<unknown>): AiJsonObject =>
  toJsonObject(JSON.parse(JSON.stringify(z.toJSONSchema(schema))) as unknown, 'tool schema');

const securityDescriptor = (entry: RegistryEntry): AiToolSecurityDescriptor => ({
  name: entry.method,
  boundary: 'harness',
  harnessOperation: entry.method,
  ...(entry.mutating ? { mutating: true, requiresHarnessApproval: true } : {}),
});

const definition = (entry: RegistryEntry, capabilities: BuiltInAgentCapabilities): AiToolDefinition => {
  const inputSchema = jsonSchema(entry.schema);
  if (entry.method === 'edit.apply') {
    const properties = inputSchema.properties;
    if (properties && typeof properties === 'object' && !Array.isArray(properties)) {
      const action = (properties as Record<string, unknown>).action;
      if (action && typeof action === 'object' && !Array.isArray(action))
        (action as Record<string, unknown>).enum = [...capabilities.editActions];
    }
  }
  return {
    name: entry.method,
    description: entry.description,
    inputSchema,
  };
};

export const builtInAgentToolDefinitions = (capabilities: BuiltInAgentCapabilities): AiToolDefinition[] => {
  const operations = new Set(capabilities.operations);
  return registryEntries
    .filter((entry) => operations.has(entry.method))
    .map((entry) => definition(entry, capabilities));
};

const serializeToolResult = (name: string, operation: string, value: unknown): BuiltInAgentToolExecutionResult => {
  const normalized = toJsonValue(value);
  const serialized = JSON.stringify(normalized);
  const originalBytes = Buffer.byteLength(serialized, 'utf8');
  if (originalBytes <= BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES)
    return { name, operation, content: serialized, truncated: false, originalBytes };

  const content = JSON.stringify({
    truncated: true,
    originalBytes,
    maximumBytes: BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES,
    preview: serialized.slice(0, maximumResultPreviewCharacters),
  });
  return { name, operation, content, truncated: true, originalBytes };
};

export class BuiltInAgentToolRegistry {
  constructor(private readonly adapter: Pick<BuiltInAgentAdapter, 'capabilities' | 'invoke'>) {}

  async definitions(): Promise<AiToolDefinition[]> {
    const capabilities = await this.adapter.capabilities();
    if (typeof capabilities.catalogRevision !== 'string') return builtInAgentToolDefinitions(capabilities);
    const definitions: AiToolDefinition[] = [];
    for (let offset = 0; offset < capabilities.operations.length; offset += 16) {
      const descriptors = await this.describe(
        capabilities.operations.slice(offset, offset + 16),
        capabilities.catalogRevision,
      );
      for (const descriptor of descriptors) {
        if (descriptor.available)
          definitions.push({
            name: descriptor.id,
            description: descriptor.description,
            inputSchema: descriptor.inputSchema,
            operationContract: { version: descriptor.version, catalogRevision: capabilities.catalogRevision },
          });
      }
    }
    return definitions;
  }

  private async describe(ids: readonly string[], catalogRevision: string): Promise<EditorOperationDescriptor[]> {
    const result = (await this.adapter.invoke('agent.operations.describe', { ids, catalogRevision })) as {
      operations: EditorOperationDescriptor[];
    };
    return result.operations;
  }

  async invoke(name: string, rawArguments: unknown = {}): Promise<BuiltInAgentToolExecutionResult> {
    const entry = entryByName.get(name);
    const capabilities = await this.adapter.capabilities();
    if (typeof capabilities.catalogRevision === 'string') {
      if (!capabilities.operations.includes(name)) throw new Error(`Unknown built-in ARC AI tool: ${name}`);
      const [descriptor] = await this.describe([name], capabilities.catalogRevision);
      assertAiToolInvocationAllowed(
        {
          name,
          boundary: 'harness',
          harnessOperation: name,
          mutating: descriptor.permissions.includes('scene.edit'),
          requiresHarnessApproval: descriptor.permissions.includes('scene.edit'),
        },
        { harnessCapabilities: new Set(capabilities.operations) },
      );
      const params = toJsonObject(rawArguments ?? {}, 'tool arguments');
      return serializeToolResult(
        name,
        name,
        await this.adapter.invoke(name, {
          ...params,
          operationContract: params.operationContract ?? {
            version: descriptor.version,
            catalogRevision: capabilities.catalogRevision,
          },
        }),
      );
    }
    if (!entry) throw new Error(`Unknown built-in ARC AI tool: ${name}`);
    const capabilitySet = new Set(capabilities.operations);
    assertAiToolInvocationAllowed(securityDescriptor(entry), { harnessCapabilities: capabilitySet });

    const parsed = entry.schema.safeParse(rawArguments ?? {});
    if (!parsed.success) {
      const issue = parsed.error.issues[0];
      const path = issue?.path.length ? ` at ${issue.path.join('.')}` : '';
      throw new Error(`Invalid arguments for ${name}${path}: ${issue?.message ?? 'schema validation failed'}`);
    }
    const params = toJsonObject(parsed.data, 'tool arguments');
    let invokeParams = params;
    if (entry.method === 'edit.apply') {
      const action = params.action;
      if (typeof action !== 'string' || !capabilities.editActions.includes(action))
        throw new Error(`Edit action '${String(action)}' is not available from EditorAgentHarness`);
      try {
        invokeParams = {
          ...params,
          value: toJsonValue(validateAgentEditValue(action as AgentEditAction, params.value), 'tool arguments.value'),
        };
      } catch (error) {
        if (error instanceof z.ZodError) {
          const issue = error.issues[0];
          const path = issue?.path.length ? `.${issue.path.join('.')}` : '';
          throw new Error(
            `Invalid arguments for edit.apply at value${path}: ${issue?.message ?? 'schema validation failed'}`,
          );
        }
        throw error;
      }
    }

    return serializeToolResult(name, entry.method, await this.adapter.invoke(entry.method, invokeParams));
  }
}

export const registeredBuiltInAgentMethods = (): AgentHarnessMethod[] => [...entryByMethod.keys()];

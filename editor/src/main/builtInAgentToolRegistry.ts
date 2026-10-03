import { z } from 'zod';

import type { AiJsonObject, AiJsonValue, AiToolDefinition } from '../common/aiRuntimeTypes';
import { assertAiToolInvocationAllowed, type AiToolSecurityDescriptor } from '../common/aiSecurityPolicy';
import type {
  BuiltInAgentCapabilities,
  BuiltInAgentToolExecutionResult,
} from '../common/builtInAgentTypes';
import { agentEditActions, type AgentHarnessMethod } from './agentHarnessContract';
import { BuiltInAgentAdapter } from './builtInAgentAdapter';

export const BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES = 64 * 1024;
const maximumResultPreviewCharacters = 16 * 1024;

const vector3 = z.tuple([z.number(), z.number(), z.number()]);
const empty = z.object({}).strict();
const captureOptions = z
  .object({
    color: z.boolean().optional(),
    depth: z.boolean().optional(),
    objectId: z.boolean().optional(),
    normals: z.boolean().optional(),
    sceneColor: z.boolean().optional(),
    baseColor: z.boolean().optional(),
    materialProperties: z.boolean().optional(),
    emissive: z.boolean().optional(),
    indirectDiffuse: z.boolean().optional(),
    reflections: z.boolean().optional(),
    traceSource: z.boolean().optional(),
    distanceField: z.boolean().optional(),
    temporalConfidence: z.boolean().optional(),
    waitFrames: z.number().int().min(0).max(120).optional(),
    maxWidth: z.number().int().min(1).max(1920).optional(),
    maxHeight: z.number().int().min(1).max(1080).optional(),
    samplePixels: z
      .array(z.object({ x: z.number().int().nonnegative(), y: z.number().int().nonnegative() }).strict())
      .max(64)
      .optional(),
  })
  .strict();
const cameraMove = z
  .object({
    action: z.enum(['orbit', 'look', 'pan', 'dolly', 'frame', 'place']),
    x: z.number().optional(),
    y: z.number().optional(),
    amount: z.number().optional(),
    guid: z.string().min(1).optional(),
    position: vector3.optional(),
    target: vector3.optional(),
    waitFrames: z.number().int().min(0).max(120).optional(),
    maxWidth: z.number().int().min(1).max(1920).optional(),
    maxHeight: z.number().int().min(1).max(1080).optional(),
  })
  .strict();
const environmentOptions = z
  .object({
    sky: z.boolean().optional(),
    fog: z.boolean().optional(),
    terrain: z.boolean().optional(),
    water: z.boolean().optional(),
    vegetation: z.boolean().optional(),
    decals: z.boolean().optional(),
  })
  .strict();
const renderOptions = z
  .object({
    renderMode: z.enum(['shaded', 'wireframe']).optional(),
    visualization: z.string().min(1).optional(),
    overlay: z.enum(['none', 'selectedWireframe', 'allWireframe']).optional(),
    selectionOutline: z.boolean().optional(),
    hoverOutline: z.boolean().optional(),
    selectionBounds: z.boolean().optional(),
    componentGizmos: z.boolean().optional(),
    selectionHierarchy: z.boolean().optional(),
    shadows: z.boolean().optional(),
    grid: z.boolean().optional(),
    environment: environmentOptions.optional(),
    waitFrames: z.number().int().min(0).max(120).optional(),
  })
  .strict();

type RegistryEntry = Readonly<{
  method: AgentHarnessMethod;
  description: string;
  schema: z.ZodType<unknown>;
  mutating?: boolean;
}>;

const registryEntries = [
  {
    method: 'agent.capabilities',
    description: 'Describe the editor operations and edit actions currently available to the built-in ARC agent.',
    schema: empty,
  },
  {
    method: 'scene.overview',
    description: 'Read the current ARC scene hierarchy, document state, and authority revisions.',
    schema: empty,
  },
  {
    method: 'scene.findEntities',
    description: 'Find scene entities by name or persistent GUID.',
    schema: z
      .object({
        search: z.string().optional(),
        offset: z.number().int().nonnegative().optional(),
        limit: z.number().int().min(1).max(200).optional(),
      })
      .strict(),
  },
  {
    method: 'scene.getEntity',
    description: 'Inspect one scene entity and its components using a persistent GUID.',
    schema: z.object({ guid: z.string().min(1) }).strict(),
  },
  {
    method: 'scene.componentSchemas',
    description: 'Read reflected ARC component and field schemas.',
    schema: empty,
  },
  {
    method: 'scene.spatialQuery',
    description: 'Raycast or find nearby, bounds-overlapping, or frustum-visible entities.',
    schema: z
      .object({
        kind: z.enum(['raycast', 'nearby', 'bounds', 'frustum']),
        origin: vector3.optional(),
        direction: vector3.optional(),
        center: vector3.optional(),
        extent: vector3.optional(),
        radius: z.number().nonnegative().optional(),
        limit: z.number().int().min(1).max(500).optional(),
      })
      .strict(),
  },
  {
    method: 'scene.changes',
    description: 'Read scene changes since a known scene revision, with full-snapshot fallback metadata.',
    schema: z.object({ sinceSceneRevision: z.number().int().nonnegative() }).strict(),
  },
  {
    method: 'assets.list',
    description: 'List project assets available for validated scene and material bindings.',
    schema: empty,
  },
  {
    method: 'viewport.state',
    description: 'Read the live ARC viewport dimensions, camera, frame revision, and render state.',
    schema: empty,
  },
  {
    method: 'viewport.move',
    description: 'Move, frame, or place the editor camera without changing persistent scene content.',
    schema: cameraMove,
  },
  {
    method: 'viewport.setRenderOptions',
    description: 'Set non-persistent viewport visualization, overlays, shadows, and environment visibility.',
    schema: renderOptions,
  },
  {
    method: 'viewport.pick',
    description: 'Pick an entity at output viewport pixel coordinates.',
    schema: z.object({ x: z.number().int().nonnegative(), y: z.number().int().nonnegative() }).strict(),
  },
  {
    method: 'viewport.observe',
    description: 'Capture coherent viewport channels and optional pixel samples from the live renderer.',
    schema: captureOptions,
  },
  {
    method: 'viewport.debug',
    description: 'Configure, settle, capture, and diagnose the viewport atomically.',
    schema: z
      .object({
        renderOptions: renderOptions.omit({ waitFrames: true }).optional(),
        camera: cameraMove.omit({ waitFrames: true, maxWidth: true, maxHeight: true }).optional(),
        capture: captureOptions.omit({ waitFrames: true, samplePixels: true }).optional(),
        samplePixels: z
          .array(z.object({ x: z.number().int().nonnegative(), y: z.number().int().nonnegative() }).strict())
          .max(64)
          .optional(),
        baselineCaptureId: z.number().int().positive().optional(),
        waitFrames: z.number().int().min(1).max(120).optional(),
      })
      .strict(),
  },
  {
    method: 'viewport.inspectPixel',
    description: 'Inspect color, depth, ObjectID/entity GUID, and normal values at one viewport pixel.',
    schema: z
      .object({
        x: z.number().int().nonnegative(),
        y: z.number().int().nonnegative(),
        captureId: z.number().int().positive().optional(),
        waitFrames: z.number().int().min(0).max(120).optional(),
      })
      .strict(),
  },
  {
    method: 'viewport.compare',
    description: 'Compare coherent viewport captures using per-channel error and changed-pixel fractions.',
    schema: z
      .object({
        baselineCaptureId: z.number().int().positive(),
        currentCaptureId: z.number().int().positive().optional(),
        waitFrames: z.number().int().min(0).max(120).optional(),
      })
      .strict(),
  },
  {
    method: 'diagnostics.get',
    description: 'Collect current scene, viewport, renderer, render-graph, shadow, and history diagnostics.',
    schema: empty,
  },
  {
    method: 'events.wait',
    description: 'Wait for a newer scene, frame, selection, or diagnostic event.',
    schema: z
      .object({
        kind: z.enum(['scene', 'frame', 'selection', 'diagnostic']),
        afterSequence: z.number().int().nonnegative().optional(),
        afterFrameRevision: z.number().int().nonnegative().optional(),
        timeoutMs: z.number().int().min(1).max(30_000).optional(),
      })
      .strict(),
  },
  {
    method: 'edit.request',
    description: 'Request user approval for a temporary ARC editor mutation scope.',
    schema: z.object({ label: z.string().optional() }).strict(),
  },
  {
    method: 'edit.begin',
    description: 'Begin one transactional editor action after user approval.',
    schema: z.object({ label: z.string().min(1), expectedSceneRevision: z.number().int().positive() }).strict(),
    mutating: true,
  },
  {
    method: 'edit.apply',
    description: 'Apply one validated scene operation or stage an authored asset inside an active transaction.',
    schema: z
      .object({
        editSessionId: z.string().min(1),
        expectedSceneRevision: z.number().int().positive(),
        action: z.enum(agentEditActions),
        value: z.record(z.string(), z.unknown()),
      })
      .strict(),
    mutating: true,
  },
  {
    method: 'edit.commit',
    description: 'Commit an active approved ARC edit transaction.',
    schema: z
      .object({ editSessionId: z.string().min(1), expectedSceneRevision: z.number().int().positive() })
      .strict(),
    mutating: true,
  },
  {
    method: 'edit.cancel',
    description: 'Cancel an active ARC edit transaction without committing staged changes.',
    schema: z.object({ editSessionId: z.string().min(1) }).strict(),
    mutating: true,
  },
  {
    method: 'history.undo',
    description: 'Undo one validated in-memory scene history operation after edit access is approved.',
    schema: z.object({ expectedSceneRevision: z.number().int().positive() }).strict(),
    mutating: true,
  },
  {
    method: 'history.redo',
    description: 'Redo one validated in-memory scene history operation after edit access is approved.',
    schema: z.object({ expectedSceneRevision: z.number().int().positive() }).strict(),
    mutating: true,
  },
] as const satisfies readonly RegistryEntry[];

const entryByMethod = new Map<AgentHarnessMethod, RegistryEntry>(registryEntries.map((entry) => [entry.method, entry]));

export const builtInAgentToolName = (method: AgentHarnessMethod): string =>
  `arc_${method
    .replaceAll('.', '_')
    .replaceAll(/([a-z0-9])([A-Z])/gu, '$1_$2')
    .toLocaleLowerCase()}`;

const entryByName = new Map(registryEntries.map((entry) => [builtInAgentToolName(entry.method), entry]));

const toJsonValue = (value: unknown, path = 'result'): AiJsonValue => {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return value;
  if (typeof value === 'number') {
    if (!Number.isFinite(value)) throw new Error(`${path} contains a non-finite number`);
    return value;
  }
  if (Array.isArray(value)) return value.map((entry, index) => toJsonValue(entry, `${path}[${String(index)}]`));
  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value as Record<string, unknown>).map(([key, nested]) => [key, toJsonValue(nested, `${path}.${key}`)]),
    );
  }
  throw new Error(`${path} contains a non-serializable ${typeof value}`);
};

const toJsonObject = (value: unknown, label: string): AiJsonObject => {
  const json = toJsonValue(value, label);
  if (!json || typeof json !== 'object' || Array.isArray(json)) throw new Error(`${label} must be an object`);
  return json;
};

const jsonSchema = (schema: z.ZodType<unknown>): AiJsonObject =>
  toJsonObject(JSON.parse(JSON.stringify(z.toJSONSchema(schema))) as unknown, 'tool schema');

const securityDescriptor = (entry: RegistryEntry): AiToolSecurityDescriptor => ({
  name: builtInAgentToolName(entry.method),
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
    name: builtInAgentToolName(entry.method),
    operationId: entry.method,
    description: entry.description,
    inputSchema,
  };
};

export const builtInAgentToolDefinitions = (capabilities: BuiltInAgentCapabilities): AiToolDefinition[] => {
  const operations = new Set(capabilities.operations);
  return registryEntries.filter((entry) => operations.has(entry.method)).map((entry) => definition(entry, capabilities));
};

const serializeToolResult = (
  name: string,
  operation: AgentHarnessMethod,
  value: unknown,
): BuiltInAgentToolExecutionResult => {
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
  constructor(private readonly adapter: BuiltInAgentAdapter) {}

  async definitions(): Promise<AiToolDefinition[]> {
    return builtInAgentToolDefinitions(await this.adapter.capabilities());
  }

  async invoke(name: string, rawArguments: unknown = {}): Promise<BuiltInAgentToolExecutionResult> {
    const entry = entryByName.get(name);
    if (!entry) throw new Error(`Unknown built-in ARC AI tool: ${name}`);

    const capabilities = await this.adapter.capabilities();
    const capabilitySet = new Set(capabilities.operations);
    assertAiToolInvocationAllowed(securityDescriptor(entry), { harnessCapabilities: capabilitySet });

    const parsed = entry.schema.safeParse(rawArguments ?? {});
    if (!parsed.success) {
      const issue = parsed.error.issues[0];
      const path = issue?.path.length ? ` at ${issue.path.join('.')}` : '';
      throw new Error(`Invalid arguments for ${name}${path}: ${issue?.message ?? 'schema validation failed'}`);
    }
    const params = toJsonObject(parsed.data, 'tool arguments');
    if (entry.method === 'edit.apply') {
      const action = params.action;
      if (typeof action !== 'string' || !capabilities.editActions.includes(action))
        throw new Error(`Edit action '${String(action)}' is not available from EditorAgentHarness`);
    }

    return serializeToolResult(name, entry.method, await this.adapter.invoke(entry.method, params));
  }
}

export const registeredBuiltInAgentMethods = (): AgentHarnessMethod[] => [...entryByMethod.keys()];

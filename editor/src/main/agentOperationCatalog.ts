import { z } from 'zod';
import { agentEditorBatchRequestSchema } from './agentEditorBatch';
import { agentEditActions, type AgentHarnessMethod } from './agentHarnessContract';
import { agentEditValueSchema, validateAgentEditValue } from './agentEditToolSchemas';
import {
  operationListRequest,
  operationDescribeRequest,
  type EditorOperationMetadata,
} from '../common/editorOperationRegistry';

const agentChoiceOption = z
  .object({
    uri: z.string().regex(/^arc:\/\/asset\/[^/?#]+$/u),
    label: z.string().min(1).max(120),
    reason: z.string().min(1).max(240).optional(),
  })
  .strict();
const agentChoices = z
  .object({
    title: z.string().min(1).max(120),
    prompt: z.string().min(1).max(280).optional(),
    selection: z.literal('single').default('single'),
    options: z.array(agentChoiceOption).min(2).max(8),
  })
  .strict();

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

export type RegistryEntry = Readonly<{
  method: AgentHarnessMethod;
  description: string;
  schema: z.ZodType<unknown>;
  mutating?: boolean;
}>;

export const registryEntries = [
  {
    method: 'agent.capabilities',
    description: 'Describe the editor operations and edit actions currently available to the built-in ARC agent.',
    schema: empty,
  },
  {
    method: 'agent.operations.list',
    description:
      'Discover bounded operation summaries, permissions, effects, and availability. Request exact schemas with agent.operations.describe.',
    schema: operationListRequest,
  },
  {
    method: 'agent.operations.describe',
    description:
      'Read exact versioned input/output schemas for up to 16 registered operations. Use the returned catalog revision when invoking.',
    schema: operationDescribeRequest,
  },
  {
    method: 'agent.presentChoices',
    description:
      'Present 2-8 visual project asset choices to the user when their selection is needed before continuing. Each option must use an authoritative arc://asset/<guid> reference from project context or tool results. Do not invent GUIDs. After calling this tool, wait for the user selection before taking dependent action.',
    schema: agentChoices,
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
    method: 'selection.set',
    description: 'Replace the editor selection with one scene entity using its persistent GUID.',
    schema: z.object({ guid: z.string().min(1) }).strict(),
  },
  {
    method: 'selection.clear',
    description: 'Clear the current editor entity selection without mutating scene content.',
    schema: empty,
  },
  {
    method: 'assets.list',
    description:
      'Search the authoritative ARC asset inventory, including engine built-ins and project content. Use this before authoring or assigning materials, textures, functions, shaders, prefabs, or other reusable assets. Results are bounded and include stable identity, readiness, provenance, dependency relations, and typed capability hints derived from the registered asset kind. For scene construction or appearance changes, inspect the current inventory before planning edits; do not assume the project lacks assets or rely on hardcoded material/function catalogs. Use returned references and distinguish capabilities from available assets.',
    schema: z
      .object({
        search: z.string().max(240).optional(),
        kinds: z.array(z.string().min(1).max(64)).max(16).optional(),
        scopes: z
          .array(z.enum(['builtin', 'project', 'user', 'organization', 'procedural']))
          .max(5)
          .optional(),
        statuses: z.array(z.string().min(1).max(32)).max(12).optional(),
        offset: z.number().int().nonnegative().optional(),
        limit: z.number().int().min(1).max(200).optional(),
      })
      .strict(),
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
        value: agentEditValueSchema,
      })
      .strict()
      .superRefine((request, context) => {
        try {
          validateAgentEditValue(request.action, request.value);
        } catch (error) {
          if (!(error instanceof z.ZodError)) throw error;
          for (const issue of error.issues) context.addIssue({ ...issue, path: ['value', ...issue.path] });
        }
      }),
    mutating: true,
  },
  {
    method: 'editor.applyBatch',
    description:
      'Apply multiple validated editor mutations in one active transaction. Use tempId to reference entities created earlier in the same batch. Prefer existing asset bindings and lightweight overrides; include material.create only when no suitable project asset or override can satisfy the request, or when a new reusable asset was explicitly requested. Inspect assets.list for appearance edits, and verify resolved material assignments and rendered results before claiming success; a successful batch alone does not prove the requested appearance changed.',
    schema: agentEditorBatchRequestSchema,
    mutating: true,
  },
  {
    method: 'edit.commit',
    description: 'Commit an active approved ARC edit transaction.',
    schema: z.object({ editSessionId: z.string().min(1), expectedSceneRevision: z.number().int().positive() }).strict(),
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

// Compatibility adapters opt into caching/concurrency only after exposing read sets.
export const operationMetadata = (entry: RegistryEntry): EditorOperationMetadata => {
  const viewportWrite = ['viewport.move', 'viewport.setRenderOptions', 'viewport.debug', 'viewport.pick'].includes(
    entry.method,
  );
  const selectionWrite = entry.method.startsWith('selection.');
  const control = entry.method.startsWith('edit.') && entry.method !== 'edit.apply';
  return {
    id: entry.method,
    version: 1,
    description: entry.description,
    domain: entry.method.split('.')[0],
    permissions: entry.mutating ? ['scene.edit'] : ['editor.read'],
    requiredCapabilities: [],
    effects: {
      reads: ['editor'],
      writes: entry.mutating
        ? ['scene', 'assets', 'edit-session']
        : viewportWrite
          ? ['viewport']
          : selectionWrite
            ? ['selection']
            : entry.method === 'edit.request'
              ? ['edit-session']
              : entry.method === 'agent.presentChoices'
                ? ['agent-ui']
                : [],
    },
    concurrency: 'serial',
    cache: 'never',
    transaction: control ? 'control' : entry.mutating ? 'legacy' : 'none',
    references:
      entry.method === 'scene.getEntity' || entry.method === 'selection.set'
        ? [{ field: 'guid', resource: 'entity', temporary: false }]
        : [],
    creates: [],
  };
};

export const legacyOperationOutput = z
  .looseObject({})
  .describe(
    'Compatibility response: an extensible native host snapshot. Domain adapters introduce precise versioned output contracts.',
  );

const legacyMcpNames: Readonly<Record<string, string>> = {
  'agent.capabilities': 'arc_agent_capabilities',
  'scene.overview': 'arc_scene_overview',
  'scene.findEntities': 'arc_find_entities',
  'scene.getEntity': 'arc_get_entity',
  'scene.componentSchemas': 'arc_component_schemas',
  'scene.spatialQuery': 'arc_spatial_query',
  'scene.changes': 'arc_scene_changes',
  'assets.list': 'arc_list_assets',
  'viewport.state': 'arc_viewport_state',
  'viewport.move': 'arc_move_viewport',
  'viewport.setRenderOptions': 'arc_set_viewport_render_options',
  'viewport.pick': 'arc_pick_viewport',
  'viewport.observe': 'arc_observe_viewport',
  'viewport.debug': 'arc_debug_viewport',
  'viewport.inspectPixel': 'arc_inspect_viewport_pixel',
  'viewport.compare': 'arc_compare_viewport_captures',
  'diagnostics.get': 'arc_diagnose_viewport',
  'events.wait': 'arc_wait_for_event',
  'edit.request': 'arc_request_edit_access',
  'edit.begin': 'arc_begin_edit',
  'edit.apply': 'arc_apply_edit',
};

export const operationMcpName = (id: string): string => legacyMcpNames[id] ?? `arc_${id.replaceAll('.', '_')}`;

import {
  assetInspectInput,
  assetInspectOutput,
  materialDescribeInput,
  materialDescribeOutput,
  materialEntityInspectInput,
  materialEditInputs,
  materialEditOutput,
  materialFunctionListInput,
  materialFunctionListOutput,
} from '../common/materialOperationContract';
import type { EditorOperationMetadata } from '../common/editorOperationRegistry';
import type { EditorAgentHarness } from './editorAgentHarness';
import type { AgentHarnessHost } from './editorAgentHarnessCore';

export function registerMaterialReadOperations(
  harness: EditorAgentHarness,
  host: AgentHarnessHost,
  edit: (clientId: string, params: Record<string, unknown>) => Promise<unknown>,
): void {
  const readMetadata = {
    version: 1,
    permissions: ['editor.read'],
    requiredCapabilities: [],
    effects: { reads: ['asset-catalog', 'saved-material-definition'], writes: [] },
    concurrency: 'serial',
    cache: 'never',
    transaction: 'none',
    references: [{ field: 'guid', resource: 'asset', temporary: false }],
    creates: [],
  } as const satisfies Omit<EditorOperationMetadata, 'id' | 'domain' | 'description'>;
  const query = async (method: string, params: Record<string, unknown>): Promise<unknown> => {
    const response = await host.query(method, params);
    if (!response.succeeded) throw new Error(response.error || `${method} failed`);
    return response.payload;
  };
  harness.registerOperation({
    ...readMetadata,
    id: 'assets.inspect',
    domain: 'assets',
    description:
      'Inspect one registered asset by GUID, including source identity, dependencies, scope and readiness. Does not modify or save assets.',
    input: assetInspectInput,
    output: assetInspectOutput,
    execute: async (params) => assetInspectOutput.parse(await query('assets.inspect', params)),
  });
  harness.registerOperation({
    ...readMetadata,
    id: 'materials.describe',
    domain: 'materials',
    description:
      'Inspect a saved Material, Material Instance or Material Function by GUID using native compiler reflection. Returns stable parameter and slot IDs, dependency revision, inherited values and diagnostics. Does not apply dirty documents or prove runtime/visual readiness.',
    input: materialDescribeInput,
    output: materialDescribeOutput,
    execute: async (params) => materialDescribeOutput.parse(await query('materials.describe', params)),
  });
  harness.registerOperation({
    ...readMetadata,
    id: 'materials.inspectEntity',
    domain: 'materials',
    effects: { reads: ['asset-catalog', 'saved-material-definition', 'scene', 'runtime-material'], writes: [] },
    description:
      'Inspect an entity material binding by entity GUID, including inherited values, local overrides, selected functions and runtime resource handle. Resource existence is not visual verification.',
    references: [{ field: 'entityGuid', resource: 'entity', temporary: false }],
    input: materialEntityInspectInput,
    output: materialDescribeOutput,
    execute: async (params) => materialDescribeOutput.parse(await query('materials.describe', params)),
  });
  harness.registerOperation({
    ...readMetadata,
    id: 'materials.listFunctions',
    domain: 'materials',
    effects: { reads: ['asset-catalog', 'saved-material-definition', 'scene'], writes: [] },
    description:
      'Page compiler-compatible functions for a slot on an asset or entity. Use nextOffset and discoveryRevision unchanged for continuation; restart on stale revision. Reads saved sources and local selections, never saves or edits.',
    references: [
      { field: 'guid', resource: 'asset', temporary: false },
      { field: 'entityGuid', resource: 'entity', temporary: false },
    ],
    input: materialFunctionListInput,
    output: materialFunctionListOutput,
    execute: async ({ slotId, offset, limit, ...identity }) =>
      materialFunctionListOutput.parse(
        await query('materials.describe', {
          ...identity,
          functionSlotId: slotId,
          functionOffset: offset,
          functionLimit: limit,
        }),
      ),
  });
  for (const [operation, input] of Object.entries(materialEditInputs)) {
    harness.registerOperation<Record<string, unknown>, ReturnType<typeof materialEditOutput.parse>>({
      ...readMetadata,
      id: `materials.${operation}`,
      domain: 'materials',
      description: `${operation === 'clearTexture' ? 'Set an explicit empty texture override; unlike resetOverride this does not restore the inherited texture.' : operation === 'resetFunction' ? 'Remove an entity-local function selection and its local inputs, restoring the inherited function.' : `${operation} an entity-local material through native validation and scene undo.`} Requires approved edit session, current world/scene authority and inspected definitionRevision. Does not save shared assets or verify rendered appearance.`,
      permissions: ['scene.edit'],
      effects: { reads: ['asset-catalog', 'saved-material-definition'], writes: ['scene'] },
      transaction: 'participant',
      references: [{ field: 'entityGuid', resource: 'entity', temporary: false }],
      input,
      output: materialEditOutput,
      execute: async (params, context) =>
        materialEditOutput.parse(await edit(context.clientId, { ...params, operation })),
    });
  }
}

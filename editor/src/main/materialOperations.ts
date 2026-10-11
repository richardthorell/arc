import {
  assetInspectInput,
  assetInspectOutput,
  materialDescribeInput,
  materialDescribeOutput,
} from '../common/materialOperationContract';
import type { EditorOperationMetadata } from '../common/editorOperationRegistry';
import type { EditorAgentHarness } from './editorAgentHarness';
import type { AgentHarnessHost } from './editorAgentHarnessCore';

export function registerMaterialReadOperations(harness: EditorAgentHarness, host: AgentHarnessHost): void {
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
  const query = async (method: string, params: { guid: string }): Promise<unknown> => {
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
}

import { describe, expect, it, vi } from 'vitest';
import { EditorAgentHarness, type AgentHarnessHost, type AgentHostResponse } from './editorAgentHarness';
import { materialDescribeOutput } from '../common/materialOperationContract';

const guid = '11111111-1111-4111-8111-111111111111';
const description = {
  guid,
  kind: 'material',
  provenance: 'savedSource',
  definitionRevision: 'a'.repeat(64),
  parentGuid: null,
  dependencies: [],
  parameters: [
    {
      id: '18446744073709551615',
      name: 'Custom artist control',
      type: 'vec3',
      defaultValue: [0.2, 0.3, 0.4],
      resolvedValue: [0.2, 0.3, 0.4],
      range: null,
      valueStatus: 'resolved',
    },
  ],
  functionSlots: [],
  functionInterface: null,
  diagnostics: [],
  definitionReady: true,
  runtimeStatus: 'notInspected',
};
function setup(payload: unknown = description) {
  const query = vi.fn(async (): Promise<AgentHostResponse> => ({
    kind: 'response',
    requestId: 1,
    succeeded: true,
    error: '',
    payload,
    sceneRevision: 1,
    worldEpoch: 1,
    frameRevision: 1,
  }));
  const command = vi.fn<AgentHarnessHost['command']>();
  return { query, command, harness: new EditorAgentHarness({ query, command }) };
}

describe('native material discovery operations', () => {
  it('discovers exact schemas through the shared registry and transport projections', async () => {
    const { harness } = setup();
    const discovery = await harness.invoke(
      'agent.operations.describe',
      { ids: ['materials.describe', 'assets.inspect'] },
      'agent',
    );
    expect(JSON.stringify(discovery)).toContain('definitionRevision');
    expect(harness.operationTransportEntries('agent').map((entry) => entry.descriptor.id)).toEqual(
      expect.arrayContaining(['materials.describe', 'assets.inspect']),
    );
  });

  it('preserves native custom metadata and 64-bit IDs without scripts, saving or scene edits', async () => {
    const { harness, query, command } = setup();
    expect(await harness.invoke('materials.describe', { guid }, 'agent')).toEqual(description);
    expect(query).toHaveBeenCalledExactlyOnceWith('materials.describe', { guid });
    expect(command).not.toHaveBeenCalled();
  });

  it('rejects paths and unregistered input fields before native dispatch', async () => {
    const { harness, query } = setup();
    await expect(harness.invoke('materials.describe', { guid: 'Content/material.arcmat' }, 'agent')).rejects.toThrow();
    await expect(harness.invoke('materials.describe', { guid, source: '{}' }, 'agent')).rejects.toThrow();
    expect(query).not.toHaveBeenCalled();
  });

  it('does not turn native compilation diagnostics into runtime or visual success', async () => {
    const result = {
      ...description,
      definitionReady: false,
      diagnostics: [{ code: 'missing-function', message: 'Repair the saved function binding' }],
    };
    const { harness } = setup(result);
    expect(await harness.invoke('materials.describe', { guid }, 'agent')).toEqual(result);
    expect(materialDescribeOutput.safeParse({ ...description, runtimeStatus: 'ready' }).success).toBe(false);
    expect(
      materialDescribeOutput.safeParse({ ...description, parameters: [{ ...description.parameters[0], id: 123 }] })
        .success,
    ).toBe(false);
  });

  it('returns catalog identity, scope and revisions for an asset without claiming compilation', async () => {
    const asset = {
      guid,
      typeId: 'texture',
      title: 'Renamed texture',
      sourcePath: 'Content/moved.png',
      scope: 'project',
      readOnly: false,
      generation: '9007199254740993',
      revision: '7',
      sourceMissing: false,
      hasLastGood: true,
      dependencies: [],
      diagnostics: [],
    };
    const { harness, query } = setup(asset);
    expect(await harness.invoke('assets.inspect', { guid }, 'agent')).toEqual(asset);
    expect(query).toHaveBeenCalledWith('assets.inspect', { guid });
  });

  it('propagates host failure and never guesses metadata', async () => {
    const { harness, query } = setup();
    query.mockResolvedValueOnce({
      kind: 'response',
      requestId: 1,
      succeeded: false,
      error: 'Asset GUID was not found',
      payload: {},
      sceneRevision: 1,
      worldEpoch: 1,
      frameRevision: 1,
    });
    await expect(harness.invoke('materials.describe', { guid }, 'agent')).rejects.toThrow('Asset GUID was not found');
  });
});

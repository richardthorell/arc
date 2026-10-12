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
      inheritedValue: [0.2, 0.3, 0.4],
      overridden: false,
      slotId: '',
      semantic: 'none',
      presentation: {
        editorKind: 'vector',
        groupId: 'surface',
        groupLabel: 'Artist controls',
        groupOrder: 2,
        order: 3,
      },
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
  entity: null,
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
  it('preserves authored presentation independently of material semantics', () => {
    const parameter = {
      ...description.parameters[0],
      type: 'vec4',
      semantic: 'none',
      defaultValue: [1, 0, 0, 1],
      inheritedValue: [1, 0, 0, 1],
      resolvedValue: [1, 0, 0, 1],
      localValue: null,
      presentation: { ...description.parameters[0].presentation, editorKind: 'color' },
    };
    const result = materialDescribeOutput.parse({ ...description, parameters: [parameter] });
    expect(result.parameters[0]).toEqual(parameter);
    expect(
      materialDescribeOutput.safeParse({
        ...description,
        parameters: [{ ...parameter, presentation: { ...parameter.presentation, order: Infinity } }],
      }).success,
    ).toBe(false);
  });

  const editParams = {
    editSessionId: 'unapproved',
    entityGuid: guid,
    expectedSceneRevision: 1,
    worldEpoch: 1,
    definitionRevision: 'a'.repeat(64),
    parameterId: '18446744073709551615',
    value: [0.2, 0.3, 0.4],
  };

  async function approved() {
    const fixture = setup();
    fixture.command.mockResolvedValue({
      kind: 'response',
      requestId: 2,
      succeeded: true,
      error: '',
      payload: { entityGuid: guid, applied: true, visualVerified: false },
      sceneRevision: 1,
      worldEpoch: 1,
      frameRevision: 1,
    });
    const request = fixture.harness.requestEdit('agent', 'Edit a local override');
    fixture.harness.approveEdit(request.id);
    const session = (await fixture.harness.invoke(
      'edit.begin',
      { label: 'Edit a local override', expectedSceneRevision: 1 },
      'agent',
    )) as { id: string };
    fixture.command.mockClear();
    return { ...fixture, params: { ...editParams, editSessionId: session.id } };
  }

  it('requires approval before dispatching any typed material mutation', async () => {
    const { harness, command } = setup();
    await expect(harness.invoke('materials.setOverride', editParams, 'agent')).rejects.toThrow(/approv|permission/i);
    expect(command).not.toHaveBeenCalled();
  });

  it('dispatches through the native scene transaction and does not claim visual success', async () => {
    const { harness, command, params } = await approved();
    expect(await harness.invoke('materials.setOverride', params, 'agent')).toEqual({
      entityGuid: guid,
      applied: true,
      visualVerified: false,
      sceneRevision: 1,
      worldEpoch: 1,
      frameRevision: 1,
    });
    expect(command).toHaveBeenCalledWith(
      'materials.edit',
      {
        operation: 'setOverride',
        entityGuid: guid,
        parameterId: params.parameterId,
        value: params.value,
        expectedSceneRevision: 1,
        worldEpoch: 1,
        definitionRevision: params.definitionRevision,
      },
      expect.objectContaining({ phase: 'update' }),
      1,
    );
  });

  it('rejects stale session revisions and malformed values before native mutation', async () => {
    const { harness, command, params } = await approved();
    await expect(
      harness.invoke('materials.setOverride', { ...params, expectedSceneRevision: 2 }, 'agent'),
    ).rejects.toThrow(/revision/i);
    await expect(harness.invoke('materials.setOverride', { ...params, parameterId: 123 }, 'agent')).rejects.toThrow();
    await expect(
      harness.invoke('materials.setOverride', { ...params, value: { path: 'guess.png' } }, 'agent'),
    ).rejects.toThrow();
    expect(command).not.toHaveBeenCalled();
  });

  it('propagates native failed compilation without reporting an applied result', async () => {
    const { harness, command, params } = await approved();
    command.mockResolvedValueOnce({
      kind: 'response',
      requestId: 3,
      succeeded: false,
      error: 'Function compilation failed; previous binding retained',
      payload: {},
      sceneRevision: 1,
      worldEpoch: 1,
      frameRevision: 1,
    });
    await expect(harness.invoke('materials.setOverride', params, 'agent')).rejects.toThrow(/previous binding retained/);
  });
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
      catalogRevision: '11',
      activeProvider: 'arc.project.source',
      providers: [{ id: 'arc.project.source', kind: 'project-source', active: true, readOnly: false, generation: '1' }],
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

import { describe, expect, it } from 'vitest';

import { EditorAgentHarness, type AgentHarnessHost, type AgentHostResponse } from './editorAgentHarness';

const response = (payload: unknown = {}, sceneRevision = 4): AgentHostResponse => ({
  kind: 'response',
  requestId: 1,
  succeeded: true,
  error: '',
  payload,
  sceneRevision,
  worldEpoch: 2,
  frameRevision: 12,
});

class BatchHost implements AgentHarnessHost {
  readonly commands: Array<{
    type: string;
    payload: Record<string, unknown>;
    edit?: Record<string, unknown>;
    revision?: number;
  }> = [];

  async command(
    type: string,
    payload: Record<string, unknown> = {},
    edit?: Record<string, unknown>,
    revision?: number,
  ): Promise<AgentHostResponse> {
    this.commands.push({ type, payload, edit, revision });
    if (type === 'entity.create')
      return response({ entity: { index: 9, generation: 1 }, guid: 'created-cube-guid' }, 5);
    if (type === 'entity.setTransform') return response({ entity: { index: 9, generation: 1 } }, 6);
    if (type === 'entity.rename') return response({ entity: { index: 9, generation: 1 } }, 7);
    return response({}, revision ?? 4);
  }

  async query(type: string, payload: Record<string, unknown> = {}): Promise<AgentHostResponse> {
    if (type === 'gateway.entity') {
      return response({
        entity: payload.guid === 'created-cube-guid' ? { index: 9, generation: 1 } : { index: 7, generation: 3 },
        guid: payload.guid,
      });
    }
    return response({ entities: [], totalEntityCount: 0 });
  }
}

const beginApprovedEdit = async (harness: EditorAgentHarness) => {
  const request = harness.requestEdit('writer', 'Batch scene edit');
  harness.approveEdit(request.id);
  return (await harness.invoke('edit.begin', { label: 'Batch scene edit', expectedSceneRevision: 4 }, 'writer')) as {
    id: string;
  };
};

describe('EditorAgentHarness editor.applyBatch', () => {
  it('creates an entity and applies follow-up edits through one batch call', async () => {
    const host = new BatchHost();
    const harness = new EditorAgentHarness(host);
    const session = await beginApprovedEdit(harness);

    const result = (await harness.invoke(
      'editor.applyBatch',
      {
        editSessionId: session.id,
        expectedSceneRevision: 4,
        operations: [
          { type: 'entity.create', tempId: 'cube', kind: 'cube' },
          {
            type: 'entity.setTransform',
            target: { tempId: 'cube' },
            transform: {
              position: [0, 0, 0],
              rotation: [0, 0, 0, 1],
              scale: [4, 4, 4],
            },
          },
          { type: 'entity.rename', target: { tempId: 'cube' }, name: 'Large Cube' },
        ],
      },
      'writer',
    )) as Record<string, unknown>;

    expect(result).toMatchObject({
      operationCount: 3,
      expectedSceneRevision: 7,
      sceneRevision: 7,
      created: { cube: { kind: 'entity', guid: 'created-cube-guid' } },
    });
    expect(host.commands.map((command) => command.type)).toEqual([
      'history.beginTransaction',
      'entity.create',
      'entity.setTransform',
      'entity.rename',
    ]);
    expect(host.commands[1].revision).toBe(4);
    expect(host.commands[2].revision).toBe(5);
    expect(host.commands[3].revision).toBe(6);
    expect(host.commands[2].edit).toMatchObject({ phase: 'update', label: 'Batch scene edit' });

    await harness.invoke('edit.commit', { editSessionId: session.id, expectedSceneRevision: 7 }, 'writer');
    expect(host.commands.at(-1)?.type).toBe('history.commitTransaction');
  });

  it('validates the complete tempId dependency graph before mutating', async () => {
    const host = new BatchHost();
    const harness = new EditorAgentHarness(host);
    const session = await beginApprovedEdit(harness);

    await expect(
      harness.invoke(
        'editor.applyBatch',
        {
          editSessionId: session.id,
          expectedSceneRevision: 4,
          operations: [
            { type: 'entity.rename', target: { tempId: 'cube' }, name: 'Too Early' },
            { type: 'entity.create', tempId: 'cube', kind: 'cube' },
          ],
        },
        'writer',
      ),
    ).rejects.toThrow(/tempId/);

    expect(host.commands.map((command) => command.type)).toEqual(['history.beginTransaction']);
    await harness.invoke('edit.cancel', { editSessionId: session.id }, 'writer');
  });
});

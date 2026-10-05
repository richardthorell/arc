import { describe, expect, it } from 'vitest';

import {
  EditorAgentHarness,
  type AgentAssetWorkspace,
  type AgentHarnessHost,
  type AgentHostResponse,
} from './editorAgentHarness';

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
    const nextRevision = (revision ?? 4) + 1;
    if (type === 'entity.create')
      return response({ entity: { index: 9, generation: 1 }, guid: 'created-cube-guid' }, nextRevision);
    if (type === 'entity.setTransform' || type === 'entity.rename' || type === 'entity.setMaterial')
      return response({ entity: { index: 9, generation: 1 } }, nextRevision);
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

class MemoryAssetWorkspace implements AgentAssetWorkspace {
  readonly files = new Map<string, string>();

  async exists(path: string): Promise<boolean> {
    return this.files.has(path);
  }

  async create(path: string, contents: string): Promise<void> {
    this.files.set(path, contents);
  }

  async remove(path: string): Promise<void> {
    this.files.delete(path);
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

  it('creates a green scaled capsule by overriding the existing material base color', async () => {
    const host = new BatchHost();
    const harness = new EditorAgentHarness(host);
    const session = await beginApprovedEdit(harness);

    const result = (await harness.invoke(
      'editor.applyBatch',
      {
        editSessionId: session.id,
        expectedSceneRevision: 4,
        operations: [
          { type: 'entity.create', tempId: 'capsule', kind: 'capsule' },
          { type: 'entity.rename', target: { tempId: 'capsule' }, name: 'Green Capsule' },
          {
            type: 'entity.setTransform',
            target: { tempId: 'capsule' },
            transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [5, 5, 5] },
          },
          { type: 'entity.setBaseColor', target: { tempId: 'capsule' }, color: [0.1, 0.8, 0.1] },
        ],
      },
      'writer',
    )) as Record<string, unknown>;

    expect(result).toMatchObject({
      operationCount: 4,
      expectedSceneRevision: 8,
      created: { capsule: { kind: 'entity', guid: 'created-cube-guid' } },
    });
    expect(host.commands.map((command) => command.type)).toEqual([
      'history.beginTransaction',
      'entity.create',
      'entity.rename',
      'entity.setTransform',
      'entity.setMaterial',
    ]);

    const materialCommand = host.commands.at(-1)!;
    const path = String(materialCommand.payload.path);
    const prefix = '__arc_primitive_parameter__/__arc_material_parameter__';
    expect(path.startsWith(prefix)).toBe(true);
    expect(path.endsWith('/0')).toBe(true);
    const encoded = path.slice(prefix.length, -2);
    expect(JSON.parse(Buffer.from(encoded, 'hex').toString('utf8'))).toEqual({
      name: 'Base Color',
      type: 'vec3',
      kind: 'color',
      value: [0.1, 0.8, 0.1],
    });

    await harness.invoke('edit.commit', { editSessionId: session.id, expectedSceneRevision: 8 }, 'writer');
    expect(host.commands.at(-1)?.type).toBe('history.commitTransaction');
  });

  it('creates and binds a semantic material in the same batch and keeps it on commit', async () => {
    const host = new BatchHost();
    const assets = new MemoryAssetWorkspace();
    const harness = new EditorAgentHarness(host, { assets });
    const session = await beginApprovedEdit(harness);

    const result = (await harness.invoke(
      'editor.applyBatch',
      {
        editSessionId: session.id,
        expectedSceneRevision: 4,
        operations: [
          { type: 'entity.create', tempId: 'capsule', kind: 'capsule' },
          { type: 'entity.rename', target: { tempId: 'capsule' }, name: 'Red Capsule' },
          {
            type: 'entity.setTransform',
            target: { tempId: 'capsule' },
            transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [5, 5, 5] },
          },
          {
            type: 'material.create',
            tempId: 'redMaterial',
            path: 'materials/red_capsule.arcmat',
            baseColor: [1, 0, 0, 1],
          },
          {
            type: 'entity.setMaterial',
            target: { tempId: 'capsule' },
            material: { tempId: 'redMaterial' },
          },
        ],
      },
      'writer',
    )) as Record<string, unknown>;

    expect(result).toMatchObject({
      operationCount: 5,
      expectedSceneRevision: 8,
      created: {
        capsule: { kind: 'entity', guid: 'created-cube-guid' },
        redMaterial: { kind: 'material', path: 'materials/red_capsule.arcmat' },
      },
    });

    const material = JSON.parse(assets.files.get('materials/red_capsule.arcmat') ?? '{}') as {
      graph?: { nodes?: Array<{ id?: string; type?: string; values?: { value?: unknown } }> };
    };
    expect(material.graph?.nodes?.find((node) => node.id === 'base-color')).toMatchObject({
      type: 'colorRgba',
      values: { value: [1, 0, 0, 1] },
    });
    const materialCommand = host.commands.find((command) => command.type === 'entity.setMaterial');
    expect(materialCommand?.payload).toMatchObject({
      entity: { index: 9, generation: 1 },
      path: 'materials/red_capsule.arcmat',
    });

    await harness.invoke('edit.commit', { editSessionId: session.id, expectedSceneRevision: 8 }, 'writer');
    expect(assets.files.has('materials/red_capsule.arcmat')).toBe(true);
  });

  it('removes a batch-created material when the transaction is cancelled', async () => {
    const host = new BatchHost();
    const assets = new MemoryAssetWorkspace();
    const harness = new EditorAgentHarness(host, { assets });
    const session = await beginApprovedEdit(harness);

    await harness.invoke(
      'editor.applyBatch',
      {
        editSessionId: session.id,
        expectedSceneRevision: 4,
        operations: [
          {
            type: 'material.create',
            tempId: 'temporaryMaterial',
            path: 'materials/temporary.arcmat',
            baseColor: [0, 1, 0, 1],
          },
        ],
      },
      'writer',
    );
    expect(assets.files.has('materials/temporary.arcmat')).toBe(true);

    await harness.invoke('edit.cancel', { editSessionId: session.id }, 'writer');
    expect(assets.files.has('materials/temporary.arcmat')).toBe(false);
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

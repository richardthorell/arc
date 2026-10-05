export * from './editorAgentHarnessCore';

import {
  type AgentBatchEntityTarget,
  type AgentBatchMaterialTarget,
  type AgentEditorBatchOperation,
  parseAgentEditorBatchRequest,
} from './agentEditorBatch';
import {
  EditorAgentHarness as EditorAgentHarnessCore,
  type AgentHarnessHost,
  type AgentHarnessOptions,
  type AgentHostResponse,
} from './editorAgentHarnessCore';

type HostEntityId = { index: number; generation: number };

type SelectionSnapshot = Record<string, unknown> & {
  sceneRevision: number;
  worldEpoch: number;
  frameRevision: number;
};

const asObject = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const requireGuid = (value: unknown): string => {
  if (typeof value !== 'string' || value.trim() === '') throw new Error('guid must be a non-empty string');
  return value;
};

const requireSceneRevision = (value: unknown): number => {
  const revision = Number(value);
  if (!Number.isSafeInteger(revision) || revision <= 0)
    throw new Error('Batch operation did not return a valid sceneRevision');
  return revision;
};

const hostEntityId = (value: unknown, guid: string): HostEntityId => {
  const entity = asObject(value);
  const index = Number(entity.index);
  const generation = Number(entity.generation);
  if (!Number.isSafeInteger(index) || index < 0 || !Number.isSafeInteger(generation) || generation < 0)
    throw new Error(`Entity ${guid} did not resolve to a live host entity`);
  return { index, generation };
};

const expectHostResponse = (response: AgentHostResponse): Record<string, unknown> => {
  if (!response.succeeded) throw new Error(response.error || 'Native host request failed');
  return asObject(response.payload);
};

const resolveBatchTarget = (target: AgentBatchEntityTarget, created: ReadonlyMap<string, string>): string => {
  if ('guid' in target) return target.guid;
  const guid = created.get(target.tempId);
  if (!guid) throw new Error(`Batch tempId '${target.tempId}' has not been created`);
  return guid;
};

const resolveBatchMaterial = (target: AgentBatchMaterialTarget, created: ReadonlyMap<string, string>): string => {
  if ('path' in target) return target.path;
  const path = created.get(target.tempId);
  if (!path) throw new Error(`Batch material tempId '${target.tempId}' has not been created`);
  return path;
};

const semanticMaterialDefinition = (
  name: string,
  baseColor: readonly [number, number, number, number],
  metallic = 0,
  roughness = 0.62,
): Record<string, unknown> => ({
  version: 4,
  name,
  domain: 'surface',
  blendMode: 'opaque',
  shadingModel: 'standard',
  doubleSided: false,
  graph: {
    version: 1,
    nodes: [
      {
        id: 'base-color',
        type: 'colorRgba',
        position: [80, 120],
        values: { value: [...baseColor] },
        parameter: { exposed: true, name: 'Base Color' },
      },
      {
        id: 'metallic',
        type: 'constant',
        position: [80, 290],
        values: { value: metallic },
        parameter: { exposed: true, name: 'Metallic' },
      },
      {
        id: 'roughness',
        type: 'constant',
        position: [80, 420],
        values: { value: roughness },
        parameter: { exposed: true, name: 'Roughness' },
      },
      { id: 'material-output', type: 'output', position: [520, 210], values: {} },
    ],
    connections: [
      {
        id: 'base-color-output',
        from: { nodeId: 'base-color', pin: 'rgb' },
        to: { nodeId: 'material-output', pin: 'baseColor' },
      },
      {
        id: 'metallic-output',
        from: { nodeId: 'metallic', pin: 'value' },
        to: { nodeId: 'material-output', pin: 'metallic' },
      },
      {
        id: 'roughness-output',
        from: { nodeId: 'roughness', pin: 'value' },
        to: { nodeId: 'material-output', pin: 'roughness' },
      },
    ],
    viewport: { x: 40, y: 40, zoom: 1 },
  },
});

const assetName = (path: string): string => {
  const filename = path.slice(path.lastIndexOf('/') + 1);
  return filename.slice(0, filename.lastIndexOf('.')) || 'New Material';
};

const batchOperationToEditApply = (
  operation: AgentEditorBatchOperation,
  createdEntities: ReadonlyMap<string, string>,
  createdMaterials: ReadonlyMap<string, string>,
): { action: string; value: Record<string, unknown> } => {
  if (operation.type === 'entity.create') {
    return {
      action: 'create',
      value: {
        ...(operation.kind ? { kind: operation.kind } : {}),
        ...(operation.parent ? { parentGuid: resolveBatchTarget(operation.parent, createdEntities) } : {}),
      },
    };
  }
  if (operation.type === 'material.create') {
    return {
      action: 'createAsset',
      value: {
        kind: 'material',
        path: operation.path,
        definition: semanticMaterialDefinition(
          operation.name ?? assetName(operation.path),
          operation.baseColor,
          operation.metallic,
          operation.roughness,
        ),
      },
    };
  }

  const guid = resolveBatchTarget(operation.target, createdEntities);
  switch (operation.type) {
    case 'entity.rename':
      return { action: 'rename', value: { guid, name: operation.name } };
    case 'entity.setActive':
      return { action: 'setActive', value: { guid, active: operation.active } };
    case 'entity.setTag':
      return { action: 'setTag', value: { guid, tag: operation.tag } };
    case 'entity.setMobility':
      return { action: 'setMobility', value: { guid, mobility: operation.mobility } };
    case 'entity.setTransform':
      return { action: 'setTransform', value: { guid, transform: operation.transform } };
    case 'entity.setRenderLayer':
      return { action: 'setRenderLayer', value: { guid, renderLayerMask: operation.renderLayerMask } };
    case 'entity.setMaterial':
      return { action: 'setMaterial', value: { guid, path: resolveBatchMaterial(operation.material, createdMaterials) } };
    case 'entity.setFlow':
      return {
        action: 'setFlow',
        value: {
          guid,
          ...(operation.path !== undefined ? { path: operation.path } : {}),
          ...(operation.assetGuid !== undefined ? { assetGuid: operation.assetGuid } : {}),
          ...(operation.enabled !== undefined ? { enabled: operation.enabled } : {}),
        },
      };
    case 'entity.snapToFloor':
      return { action: 'snapToFloor', value: { guid } };
    case 'entity.delete':
      return { action: 'delete', value: { guid } };
    case 'entity.duplicate':
      return { action: 'duplicate', value: { guid } };
    case 'entity.reparent':
      return {
        action: 'reparent',
        value: {
          guid,
          ...(operation.parent ? { parentGuid: resolveBatchTarget(operation.parent, createdEntities) } : {}),
          ...(operation.preserveWorld !== undefined ? { preserveWorld: operation.preserveWorld } : {}),
        },
      };
    case 'entity.patchComponent':
      return {
        action: 'patchComponent',
        value: { guid, component: operation.component, fields: operation.fields },
      };
  }
};

/**
 * Production harness boundary for editor-only state and higher-level editor tools.
 *
 * The core harness remains responsible for scene reads, viewport state, and
 * transactional edits. This boundary adds UI-state selection and model-facing
 * batching without creating a second mutation implementation: batches are
 * validated up front and then reuse the authoritative edit.apply path.
 */
export class EditorAgentHarness extends EditorAgentHarnessCore {
  constructor(
    private readonly selectionHost: AgentHarnessHost,
    options: AgentHarnessOptions = {},
  ) {
    super(selectionHost, options);
  }

  override async invoke(method: string, rawParams: unknown, clientId: string): Promise<unknown> {
    if (method === 'selection.set') return this.setSelection(rawParams, clientId);
    if (method === 'selection.clear') return this.clearSelection(clientId);
    if (method === 'editor.applyBatch') return this.applyBatch(rawParams, clientId);
    return super.invoke(method, rawParams, clientId);
  }

  private async applyBatch(rawParams: unknown, clientId: string): Promise<Record<string, unknown>> {
    const request = parseAgentEditorBatchRequest(rawParams);
    const createdEntities = new Map<string, string>();
    const createdMaterials = new Map<string, string>();
    const createdResources: Record<string, { kind: 'entity'; guid: string } | { kind: 'material'; path: string }> = {};
    const results: Array<Record<string, unknown>> = [];
    let expectedSceneRevision = request.expectedSceneRevision;
    let authority: Record<string, unknown> = {};

    for (const [index, operation] of request.operations.entries()) {
      const edit = batchOperationToEditApply(operation, createdEntities, createdMaterials);
      const result = asObject(
        await super.invoke(
          'edit.apply',
          {
            editSessionId: request.editSessionId,
            expectedSceneRevision,
            action: edit.action,
            value: edit.value,
          },
          clientId,
        ),
      );
      expectedSceneRevision = requireSceneRevision(result.sceneRevision);
      authority = result;

      if (operation.type === 'entity.create' && operation.tempId) {
        const guid = requireGuid(result.guid);
        createdEntities.set(operation.tempId, guid);
        createdResources[operation.tempId] = { kind: 'entity', guid };
      } else if (operation.type === 'material.create') {
        createdMaterials.set(operation.tempId, operation.path);
        createdResources[operation.tempId] = { kind: 'material', path: operation.path };
      }

      results.push({ index, type: operation.type, result });
    }

    return {
      editSessionId: request.editSessionId,
      operationCount: request.operations.length,
      results,
      created: createdResources,
      expectedSceneRevision,
      sceneRevision: expectedSceneRevision,
      worldEpoch: authority.worldEpoch,
      frameRevision: authority.frameRevision,
    };
  }

  private async setSelection(rawParams: unknown, clientId: string): Promise<SelectionSnapshot> {
    const guid = requireGuid(asObject(rawParams).guid);
    const resolved = asObject(await super.invoke('scene.getEntity', { guid }, clientId));
    const entity = hostEntityId(resolved.entity, guid);

    expectHostResponse(
      await this.selectionHost.command('entity.select', {
        entity,
        additive: false,
        toggle: false,
      }),
    );
    return this.selectionSnapshot(clientId);
  }

  private async clearSelection(clientId: string): Promise<SelectionSnapshot> {
    expectHostResponse(await this.selectionHost.command('entity.clearSelection', {}));
    return this.selectionSnapshot(clientId);
  }

  private async selectionSnapshot(clientId: string): Promise<SelectionSnapshot> {
    const selected = expectHostResponse(await this.selectionHost.query('entity.selected'));
    // Feed one authoritative host response back through the core so its public
    // status/capabilities revisions stay synchronized with this UI-state change.
    const authority = asObject(await super.invoke('scene.overview', {}, clientId));
    return {
      ...selected,
      sceneRevision: Number(authority.sceneRevision ?? 0),
      worldEpoch: Number(authority.worldEpoch ?? 0),
      frameRevision: Number(authority.frameRevision ?? 0),
    };
  }
}

export * from './editorAgentHarnessCore';

import {
  type AgentBatchEntityTarget,
  type AgentBatchMaterialTarget,
  type AgentEditorBatchOperation,
  parseAgentEditorBatchRequest,
} from './agentEditorBatch';
import {
  EditorAgentHarness as EditorAgentHarnessCore,
  type AgentAssetWorkspace,
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

type BatchCreatedAssets = {
  clientId: string;
  paths: string[];
};

const materialParameterCommandPrefix = '__arc_primitive_parameter__/__arc_material_parameter__';

const asObject = (value: unknown): Record<string, unknown> =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};

const requireGuid = (value: unknown): string => {
  if (typeof value !== 'string' || value.trim() === '') throw new Error('guid must be a non-empty string');
  return value;
};

const requireEditSessionId = (value: unknown): string => {
  const id = asObject(value).editSessionId;
  if (typeof id !== 'string' || id.trim() === '') throw new Error('editSessionId must be a non-empty string');
  return id;
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

const materialParameterPath = (
  name: string,
  type: string,
  kind: string,
  value: readonly number[],
): string => {
  const payload = JSON.stringify({ name, type, kind, value: [...value] });
  return `${materialParameterCommandPrefix}${Buffer.from(payload, 'utf8').toString('hex')}/0`;
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

const materialContents = (operation: Extract<AgentEditorBatchOperation, { type: 'material.create' }>): string => {
  const filename = operation.path.slice(operation.path.lastIndexOf('/') + 1);
  const inferredName = filename.slice(0, filename.lastIndexOf('.')) || 'New Material';
  const definition = semanticMaterialDefinition(
    operation.name ?? inferredName,
    operation.baseColor,
    operation.metallic,
    operation.roughness,
  );
  return `${JSON.stringify(definition, null, 2)}\n`;
};

const batchOperationToEditApply = (
  operation: Exclude<AgentEditorBatchOperation, { type: 'material.create' }>,
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
    case 'entity.setMaterial': {
      const path = operation.material ? resolveBatchMaterial(operation.material, createdMaterials) : operation.path;
      if (!path) throw new Error('entity.setMaterial requires a material path or material tempId');
      return { action: 'setMaterial', value: { guid, path } };
    }
    case 'entity.setBaseColor':
      return {
        action: 'setMaterial',
        value: { guid, path: materialParameterPath('Base Color', 'vec3', 'color', operation.color) },
      };
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
 * batching without creating a second scene mutation implementation. Semantic
 * batch-created assets are materialized immediately so later operations in the
 * same approved transaction can reference them, and are removed on cancel.
 */
export class EditorAgentHarness extends EditorAgentHarnessCore {
  private readonly assetWorkspace: AgentAssetWorkspace | undefined;
  private readonly batchCreatedAssets = new Map<string, BatchCreatedAssets>();

  constructor(
    private readonly selectionHost: AgentHarnessHost,
    options: AgentHarnessOptions = {},
  ) {
    super(selectionHost, options);
    this.assetWorkspace = options.assets;
  }

  override async disconnectClient(clientId: string): Promise<void> {
    await super.disconnectClient(clientId);
    const sessions = [...this.batchCreatedAssets.entries()]
      .filter(([, assets]) => assets.clientId === clientId)
      .map(([editSessionId]) => editSessionId);
    for (const editSessionId of sessions) await this.removeBatchCreatedAssets(editSessionId);
  }

  override async invoke(method: string, rawParams: unknown, clientId: string): Promise<unknown> {
    if (method === 'selection.set') return this.setSelection(rawParams, clientId);
    if (method === 'selection.clear') return this.clearSelection(clientId);
    if (method === 'editor.applyBatch') return this.applyBatch(rawParams, clientId);
    if (method === 'edit.cancel') {
      const editSessionId = requireEditSessionId(rawParams);
      const result = await super.invoke(method, rawParams, clientId);
      await this.removeBatchCreatedAssets(editSessionId);
      return result;
    }
    if (method === 'edit.commit') {
      const editSessionId = requireEditSessionId(rawParams);
      const result = await super.invoke(method, rawParams, clientId);
      this.batchCreatedAssets.delete(editSessionId);
      return result;
    }
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
      if (operation.type === 'material.create') {
        const workspace = this.assetWorkspace;
        if (!workspace) throw new Error('Asset authoring is not available in this editor session');
        if (await workspace.exists(operation.path)) throw new Error(`Asset already exists: ${operation.path}`);
        await workspace.create(operation.path, materialContents(operation));
        this.trackBatchCreatedAsset(request.editSessionId, clientId, operation.path);
        createdMaterials.set(operation.tempId, operation.path);
        createdResources[operation.tempId] = { kind: 'material', path: operation.path };
        if (Object.keys(authority).length === 0)
          authority = asObject(await super.invoke('scene.overview', {}, clientId));
        const result = {
          created: true,
          asset: { kind: 'material', path: operation.path },
          sceneRevision: expectedSceneRevision,
          worldEpoch: authority.worldEpoch,
          frameRevision: authority.frameRevision,
        };
        results.push({ index, type: operation.type, result });
        continue;
      }

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

  private trackBatchCreatedAsset(editSessionId: string, clientId: string, path: string): void {
    const current = this.batchCreatedAssets.get(editSessionId) ?? { clientId, paths: [] };
    if (current.clientId !== clientId) throw new Error('Batch asset edit session belongs to another client');
    current.paths.push(path);
    this.batchCreatedAssets.set(editSessionId, current);
  }

  private async removeBatchCreatedAssets(editSessionId: string): Promise<void> {
    const created = this.batchCreatedAssets.get(editSessionId);
    if (!created) return;
    const workspace = this.assetWorkspace;
    if (workspace) {
      for (const path of [...created.paths].reverse()) await workspace.remove(path);
    }
    this.batchCreatedAssets.delete(editSessionId);
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

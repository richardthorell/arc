export * from './editorAgentHarnessCore';

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

/**
 * Production harness boundary for editor-only selection state.
 *
 * The core harness remains responsible for scene reads, viewport state, and
 * transactional edits. Selection deliberately sits outside edit sessions: it
 * resolves persistent identity through the existing GUID query, then invokes
 * the same native select/clear commands used by normal editor interaction.
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
    return super.invoke(method, rawParams, clientId);
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

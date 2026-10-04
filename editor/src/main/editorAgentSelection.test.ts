import { describe, expect, it } from 'vitest';

import { EditorAgentHarness, type AgentHarnessHost, type AgentHostResponse } from './editorAgentHarness';

const reply = (
  payload: unknown = {},
  overrides: Partial<Pick<AgentHostResponse, 'succeeded' | 'error'>> = {},
): AgentHostResponse => ({
  kind: 'response',
  requestId: 1,
  succeeded: overrides.succeeded ?? true,
  error: overrides.error ?? '',
  payload,
  sceneRevision: 9,
  worldEpoch: 3,
  frameRevision: 14,
});

class SelectionHost implements AgentHarnessHost {
  readonly commands: Array<{ type: string; payload?: Record<string, unknown> }> = [];
  selectedGuid = '';
  staleGuid = '';

  async command(type: string, payload?: Record<string, unknown>): Promise<AgentHostResponse> {
    this.commands.push({ type, payload });
    if (type === 'entity.select') {
      const entity = payload?.entity as { index?: number; generation?: number } | undefined;
      const guid = entity?.index === 7 && entity?.generation === 3 ? 'floor-guid' : '';
      if (!guid || guid === this.staleGuid)
        return reply({}, { succeeded: false, error: 'Cannot select a missing entity' });
      this.selectedGuid = guid;
      return reply({ entity });
    }
    if (type === 'entity.clearSelection') {
      this.selectedGuid = '';
      return reply();
    }
    return reply();
  }

  async query(type: string, payload?: Record<string, unknown>): Promise<AgentHostResponse> {
    if (type === 'gateway.entity') {
      if (payload?.guid !== 'floor-guid')
        return reply({}, { succeeded: false, error: `Entity ${String(payload?.guid)} was not found` });
      return reply({
        entity: { index: 7, generation: 3 },
        guid: 'floor-guid',
        name: 'Floor',
      });
    }
    if (type === 'entity.selected') {
      return reply(
        this.selectedGuid
          ? {
              entity: { index: 7, generation: 3 },
              selectionCount: 1,
              selectedGuids: [this.selectedGuid],
              guid: this.selectedGuid,
              name: 'Floor',
            }
          : {
              entity: { index: 0xffffffff, generation: 0 },
              selectionCount: 0,
              selectedGuids: [],
              guid: '',
              name: '',
            },
      );
    }
    if (type === 'gateway.sceneEntities') {
      return reply({
        entities: [
          {
            entity: { index: 7, generation: 3 },
            guid: 'floor-guid',
            name: 'Floor',
            selected: this.selectedGuid === 'floor-guid',
          },
        ],
        totalEntityCount: 1,
      });
    }
    return reply();
  }
}

describe('EditorAgentHarness entity selection', () => {
  it('selects a discovered entity by persistent GUID through the native editor selection path', async () => {
    const host = new SelectionHost();
    const harness = new EditorAgentHarness(host);

    const capabilities = (await harness.invoke('agent.capabilities', {}, 'agent')) as { operations: string[] };
    expect(capabilities.operations).toEqual(expect.arrayContaining(['selection.set', 'selection.clear']));

    const result = await harness.invoke('selection.set', { guid: 'floor-guid' }, 'agent');

    expect(host.commands).toEqual([
      {
        type: 'entity.select',
        payload: { entity: { index: 7, generation: 3 }, additive: false, toggle: false },
      },
    ]);
    expect(result).toMatchObject({
      selectionCount: 1,
      selectedGuids: ['floor-guid'],
      guid: 'floor-guid',
      sceneRevision: 9,
      worldEpoch: 3,
      frameRevision: 14,
    });
    expect(harness.status()).toMatchObject({ sceneRevision: 9, worldEpoch: 3, frameRevision: 14 });
    expect(host.commands.some((command) => command.type.startsWith('history.'))).toBe(false);
  });

  it('clears editor selection without opening an edit transaction', async () => {
    const host = new SelectionHost();
    host.selectedGuid = 'floor-guid';
    const harness = new EditorAgentHarness(host);

    const result = await harness.invoke('selection.clear', {}, 'agent');

    expect(host.commands).toEqual([{ type: 'entity.clearSelection', payload: {} }]);
    expect(result).toMatchObject({
      selectionCount: 0,
      selectedGuids: [],
      sceneRevision: 9,
      worldEpoch: 3,
      frameRevision: 14,
    });
  });

  it('fails explicitly for missing and stale persistent GUIDs', async () => {
    const host = new SelectionHost();
    const harness = new EditorAgentHarness(host);

    await expect(harness.invoke('selection.set', { guid: 'missing-guid' }, 'agent')).rejects.toThrow(
      'Entity missing-guid was not found',
    );
    expect(host.commands).toHaveLength(0);

    host.staleGuid = 'floor-guid';
    await expect(harness.invoke('selection.set', { guid: 'floor-guid' }, 'agent')).rejects.toThrow(
      'Cannot select a missing entity',
    );
  });
});

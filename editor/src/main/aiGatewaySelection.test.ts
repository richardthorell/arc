import { mkdtempSync, rmSync } from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { afterEach, describe, expect, it } from 'vitest';

import { AiGatewayServer } from './aiGatewayServer';
import { EditorAgentHarness, type AgentHarnessHost, type AgentHostResponse } from './editorAgentHarness';

const temporaryDirectories: string[] = [];
afterEach(() => {
  for (const directory of temporaryDirectories.splice(0)) rmSync(directory, { recursive: true, force: true });
});

const reply = (payload: unknown = {}): AgentHostResponse => ({
  kind: 'response',
  requestId: 1,
  succeeded: true,
  error: '',
  payload,
  sceneRevision: 6,
  worldEpoch: 2,
  frameRevision: 11,
});

class GatewaySelectionHost implements AgentHarnessHost {
  selected = false;
  readonly commands: string[] = [];

  async command(type: string): Promise<AgentHostResponse> {
    this.commands.push(type);
    if (type === 'entity.select') this.selected = true;
    if (type === 'entity.clearSelection') this.selected = false;
    return reply();
  }

  async query(type: string, payload?: Record<string, unknown>): Promise<AgentHostResponse> {
    if (type === 'gateway.entity') {
      if (payload?.guid !== 'floor-guid') return { ...reply(), succeeded: false, error: 'Entity was not found' };
      return reply({ entity: { index: 4, generation: 2 }, guid: 'floor-guid', name: 'Floor' });
    }
    if (type === 'entity.selected') {
      return reply({
        entity: this.selected ? { index: 4, generation: 2 } : { index: 0xffffffff, generation: 0 },
        selectionCount: this.selected ? 1 : 0,
        selectedGuids: this.selected ? ['floor-guid'] : [],
        guid: this.selected ? 'floor-guid' : '',
      });
    }
    if (type === 'gateway.sceneEntities') return reply({ entities: [], totalEntityCount: 0 });
    return reply();
  }
}

describe('AI Gateway editor selection', () => {
  it('exposes selection.set and selection.clear with the same harness semantics', async () => {
    const directory = mkdtempSync(path.join(os.tmpdir(), 'arc-ai-selection-'));
    temporaryDirectories.push(directory);
    const host = new GatewaySelectionHost();
    const server = new AiGatewayServer(new EditorAgentHarness(host), { appDataPath: directory });
    await server.start();

    const endpoint = server.status().endpoint;
    const headers = {
      authorization: `Bearer ${server.token}`,
      'content-type': 'application/json',
      'x-arc-client-id': 'selection-test',
    };
    try {
      const selected = await fetch(`${endpoint}/api/v1/selection/set`, {
        method: 'POST',
        headers,
        body: JSON.stringify({ guid: 'floor-guid' }),
      });
      expect(selected.status).toBe(200);
      expect(await selected.json()).toMatchObject({
        result: {
          selectionCount: 1,
          selectedGuids: ['floor-guid'],
          sceneRevision: 6,
          frameRevision: 11,
        },
      });

      const cleared = await fetch(`${endpoint}/api/v1/selection/clear`, {
        method: 'POST',
        headers,
        body: '{}',
      });
      expect(cleared.status).toBe(200);
      expect(await cleared.json()).toMatchObject({ result: { selectionCount: 0, selectedGuids: [] } });
      expect(host.commands).toEqual(['entity.select', 'entity.clearSelection']);

      const openApi = await fetch(`${endpoint}/openapi.json`, { headers });
      const document = (await openApi.json()) as { paths?: Record<string, unknown> };
      expect(document.paths?.['/api/v1/selection/set']).toBeDefined();
      expect(document.paths?.['/api/v1/selection/clear']).toBeDefined();
    } finally {
      await server.stop();
    }
  });
});

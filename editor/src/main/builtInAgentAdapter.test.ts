import { describe, expect, it } from 'vitest';

import { BUILT_IN_AGENT_CLIENT_ID, BUILT_IN_AGENT_CLIENT_NAME } from '../common/builtInAgentTypes';
import { BuiltInAgentAdapter } from './builtInAgentAdapter';
import {
  EditorAgentHarness,
  type AgentHarnessHost,
  type AgentHostResponse,
} from './editorAgentHarness';

const response = (payload: unknown = {}): AgentHostResponse => ({
  kind: 'response',
  requestId: 1,
  succeeded: true,
  error: '',
  payload,
  sceneRevision: 4,
  worldEpoch: 2,
  frameRevision: 12,
});

class MockHost implements AgentHarnessHost {
  async command(): Promise<AgentHostResponse> {
    return response();
  }

  async query(): Promise<AgentHostResponse> {
    return response({ entities: [], totalEntityCount: 0 });
  }
}

describe('BuiltInAgentAdapter', () => {
  it('owns one stable harness identity and discovers capabilities through the harness', async () => {
    const harness = new EditorAgentHarness(new MockHost());
    const adapter = new BuiltInAgentAdapter(harness);

    expect(adapter.clientId).toBe(BUILT_IN_AGENT_CLIENT_ID);
    expect(adapter.clientName).toBe(BUILT_IN_AGENT_CLIENT_NAME);
    expect(harness.status().clients).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: BUILT_IN_AGENT_CLIENT_ID, name: BUILT_IN_AGENT_CLIENT_NAME }),
      ]),
    );

    const capabilities = await adapter.capabilities();
    expect(capabilities.operations).toEqual(expect.arrayContaining(['agent.capabilities', 'scene.overview', 'edit.begin']));
    expect(capabilities.editActions).toEqual(expect.arrayContaining(['create', 'setTransform', 'createAsset']));
  });

  it('routes operations through harness approval and audit policy instead of bypassing it', async () => {
    const harness = new EditorAgentHarness(new MockHost());
    const adapter = new BuiltInAgentAdapter(harness);

    const request = (await adapter.invoke('edit.request', { label: 'AI edit' })) as { id: string };
    expect(harness.status().pendingEditRequests).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: request.id, clientId: BUILT_IN_AGENT_CLIENT_ID, label: 'AI edit' }),
      ]),
    );
    await expect(
      adapter.invoke('edit.begin', { label: 'AI edit', expectedSceneRevision: 4 }),
    ).rejects.toThrow();
    expect(harness.status().audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ clientId: BUILT_IN_AGENT_CLIENT_ID, operation: 'edit.request', succeeded: true }),
        expect.objectContaining({ clientId: BUILT_IN_AGENT_CLIENT_ID, operation: 'edit.begin', succeeded: false }),
      ]),
    );
  });

  it('forwards harness events and disconnects the built-in client when disposed', async () => {
    const harness = new EditorAgentHarness(new MockHost());
    const adapter = new BuiltInAgentAdapter(harness);
    const events: string[] = [];
    const unsubscribe = adapter.onEvent((event) => events.push(event.type));

    harness.recordHostEvent({ type: 'scene.changed', message: 'Scene changed', payload: { revision: 5 } });
    expect(events).toEqual(['scene.changed']);

    unsubscribe();
    harness.recordHostEvent({ type: 'scene.changed', message: 'Scene changed again', payload: { revision: 6 } });
    expect(events).toEqual(['scene.changed']);

    await adapter.dispose();
    expect(harness.status().clients.some((client) => client.id === BUILT_IN_AGENT_CLIENT_ID)).toBe(false);
    await expect(adapter.capabilities()).rejects.toThrow('disposed');
  });
});

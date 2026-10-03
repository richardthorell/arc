import { describe, expect, it, vi } from 'vitest';

import type { BuiltInAgentCapabilities } from '../common/builtInAgentTypes';
import { agentHarnessMethods } from './agentHarnessContract';
import {
  BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES,
  BuiltInAgentToolRegistry,
  builtInAgentToolDefinitions,
  registeredBuiltInAgentMethods,
} from './builtInAgentToolRegistry';

const capabilities = (
  operations: readonly string[] = agentHarnessMethods,
  editActions: readonly string[] = ['rename', 'setTransform'],
): BuiltInAgentCapabilities => ({ operations, editActions });

const fakeAdapter = (snapshot: BuiltInAgentCapabilities, result: unknown = { ok: true }) => {
  const invoke = vi.fn(async () => result);
  return {
    adapter: {
      capabilities: vi.fn(async () => snapshot),
      invoke,
    },
    invoke,
  };
};

describe('BuiltInAgentToolRegistry', () => {
  it('keeps the registry aligned with the authoritative harness operation catalog', () => {
    expect(registeredBuiltInAgentMethods()).toEqual(agentHarnessMethods);
  });

  it('advertises only runtime-supported harness operations with stable operation names', async () => {
    const snapshot = capabilities(['scene.getEntity', 'viewport.state']);
    const { adapter } = fakeAdapter(snapshot);
    const registry = new BuiltInAgentToolRegistry(adapter);

    const definitions = await registry.definitions();
    expect(definitions.map((tool) => tool.name)).toEqual(['scene.getEntity', 'viewport.state']);
    expect(definitions[0]).toMatchObject({
      name: 'scene.getEntity',
      inputSchema: {
        type: 'object',
        required: ['guid'],
        additionalProperties: false,
      },
    });
  });

  it('validates arguments before invoking the harness operation', async () => {
    const { adapter, invoke } = fakeAdapter(capabilities(['scene.getEntity']));
    const registry = new BuiltInAgentToolRegistry(adapter);

    await expect(registry.invoke('scene.getEntity', {})).rejects.toThrow('Invalid arguments for scene.getEntity');
    expect(invoke).not.toHaveBeenCalled();

    const result = await registry.invoke('scene.getEntity', { guid: 'entity-guid' });
    expect(invoke).toHaveBeenCalledWith('scene.getEntity', { guid: 'entity-guid' });
    expect(JSON.parse(result.content)).toEqual({ ok: true });
    expect(result).toMatchObject({
      name: 'scene.getEntity',
      operation: 'scene.getEntity',
      truncated: false,
    });
  });

  it('rejects operations and edit actions removed from the live harness capability snapshot', async () => {
    const snapshot = capabilities(['edit.apply'], ['rename']);
    const { adapter, invoke } = fakeAdapter(snapshot);
    const registry = new BuiltInAgentToolRegistry(adapter);

    await expect(registry.invoke('viewport.state')).rejects.toThrow(/tool_capability_missing/);
    await expect(
      registry.invoke('edit.apply', {
        editSessionId: 'edit-1',
        expectedSceneRevision: 4,
        action: 'setTransform',
        value: {},
      }),
    ).rejects.toThrow("Edit action 'setTransform' is not available from EditorAgentHarness");
    expect(invoke).not.toHaveBeenCalled();

    const definition = builtInAgentToolDefinitions(snapshot).find((tool) => tool.name === 'edit.apply');
    const properties = definition?.inputSchema.properties as Record<string, unknown> | undefined;
    expect(properties?.action).toMatchObject({ enum: ['rename'] });
  });

  it('serializes results deterministically and truncates oversized payloads at the registry boundary', async () => {
    const oversized = { text: 'x'.repeat(BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES + 1024) };
    const { adapter } = fakeAdapter(capabilities(['viewport.state']), oversized);
    const registry = new BuiltInAgentToolRegistry(adapter);

    const result = await registry.invoke('viewport.state');
    expect(result.truncated).toBe(true);
    expect(result.originalBytes).toBeGreaterThan(BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES);
    expect(Buffer.byteLength(result.content, 'utf8')).toBeLessThan(BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES);
    expect(JSON.parse(result.content)).toMatchObject({
      truncated: true,
      maximumBytes: BUILT_IN_AGENT_TOOL_RESULT_MAX_BYTES,
    });
  });

  it('rejects non-JSON harness results instead of silently changing their meaning', async () => {
    const { adapter } = fakeAdapter(capabilities(['viewport.state']), { value: undefined });
    const registry = new BuiltInAgentToolRegistry(adapter);

    await expect(registry.invoke('viewport.state')).rejects.toThrow('non-serializable undefined');
  });
});

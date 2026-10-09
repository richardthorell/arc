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

const transform = {
  position: [0, 0, 0],
  rotation: [0, 0, 0, 1],
  scale: [4, 4, 4],
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

  it('advertises explicit value shapes for edit actions instead of an untyped record', () => {
    const definition = builtInAgentToolDefinitions(capabilities(['edit.apply'], ['rename', 'setTransform'])).find(
      (tool) => tool.name === 'edit.apply',
    );
    const properties = definition?.inputSchema.properties as Record<string, unknown> | undefined;
    const value = properties?.value as { anyOf?: Array<Record<string, unknown>> } | undefined;

    expect(properties?.action).toMatchObject({ enum: ['rename', 'setTransform'] });
    expect(value?.anyOf?.length).toBeGreaterThan(2);
    expect(value?.anyOf).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          description: expect.stringContaining('setTransform'),
          required: ['guid', 'transform'],
          additionalProperties: false,
        }),
        expect.objectContaining({
          description: expect.stringContaining('rename'),
          required: ['guid', 'name'],
          additionalProperties: false,
        }),
      ]),
    );
  });

  it('describes authoritative asset discovery and reuse before authoring', () => {
    const definitions = builtInAgentToolDefinitions(
      capabilities(['assets.list', 'edit.apply', 'editor.applyBatch'], ['setMaterial', 'createAsset']),
    );
    const assets = definitions.find((tool) => tool.name === 'assets.list');
    const edit = definitions.find((tool) => tool.name === 'edit.apply');
    const batch = definitions.find((tool) => tool.name === 'editor.applyBatch');

    expect(assets?.description).toContain('authoritative ARC asset inventory');
    expect(assets?.description).toContain('engine built-ins and project content');
    expect(assets?.description).toContain('before authoring or assigning materials');
    expect(batch?.description).toContain('Prefer existing asset bindings and lightweight overrides');
    expect(batch?.description).toContain('material.create only when no suitable project asset or override');

    expect(assets?.inputSchema).toMatchObject({
      type: 'object',
      additionalProperties: false,
      properties: expect.objectContaining({
        search: expect.objectContaining({ type: 'string' }),
        kinds: expect.objectContaining({ type: 'array' }),
        scopes: expect.objectContaining({ type: 'array' }),
        offset: expect.objectContaining({ type: 'integer', minimum: 0 }),
        limit: expect.objectContaining({ type: 'integer', minimum: 1, maximum: 200 }),
      }),
    });

    const editSchema = JSON.stringify(edit?.inputSchema);
    expect(editSchema).toContain('prefer binding an existing project material');
    expect(editSchema).toContain(
      'only after suitable project-local assets and lightweight overrides have been considered',
    );
  });

  it('validates visual asset choice payloads before exposing them to chat', async () => {
    const snapshot = capabilities(['agent.presentChoices']);
    const { adapter, invoke } = fakeAdapter(snapshot);
    const registry = new BuiltInAgentToolRegistry(adapter);

    const definition = (await registry.definitions())[0];
    expect(definition).toMatchObject({
      name: 'agent.presentChoices',
      description: expect.stringContaining('visual project asset choices'),
    });

    await expect(
      registry.invoke('agent.presentChoices', {
        title: 'Choose a rock',
        options: [{ uri: 'arc://asset/rock-a', label: 'Only one' }],
      }),
    ).rejects.toThrow('Invalid arguments for agent.presentChoices');

    await expect(
      registry.invoke('agent.presentChoices', {
        title: 'Choose a rock',
        options: [
          { uri: 'arc://entity/entity-a', label: 'Wrong kind' },
          { uri: 'arc://asset/rock-b', label: 'Rock B' },
        ],
      }),
    ).rejects.toThrow('Invalid arguments for agent.presentChoices');
    expect(invoke).not.toHaveBeenCalled();

    await registry.invoke('agent.presentChoices', {
      title: 'Choose a rock',
      prompt: 'Pick one before placement.',
      options: [
        { uri: 'arc://asset/rock-a', label: 'Rock A', reason: 'Closest silhouette.' },
        { uri: 'arc://asset/rock-b', label: 'Rock B' },
      ],
    });
    expect(invoke).toHaveBeenCalledWith('agent.presentChoices', {
      title: 'Choose a rock',
      prompt: 'Pick one before placement.',
      selection: 'single',
      options: [
        { uri: 'arc://asset/rock-a', label: 'Rock A', reason: 'Closest silhouette.' },
        { uri: 'arc://asset/rock-b', label: 'Rock B' },
      ],
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

  it('rejects malformed action values before they reach edit handlers', async () => {
    const snapshot = capabilities(['edit.apply'], ['rename', 'setTransform']);
    const { adapter, invoke } = fakeAdapter(snapshot);
    const registry = new BuiltInAgentToolRegistry(adapter);
    const common = { editSessionId: 'edit-1', expectedSceneRevision: 3, action: 'setTransform' as const };

    await expect(
      registry.invoke('edit.apply', {
        ...common,
        value: { guid: 'entity-guid', position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [4, 4, 4] },
      }),
    ).rejects.toThrow('Invalid arguments for edit.apply');
    await expect(
      registry.invoke('edit.apply', {
        ...common,
        value: { entity: 'entity-guid', transform },
      }),
    ).rejects.toThrow('Invalid arguments for edit.apply');
    expect(invoke).not.toHaveBeenCalled();

    await registry.invoke('edit.apply', {
      ...common,
      value: { guid: 'entity-guid', transform },
    });
    expect(invoke).toHaveBeenCalledWith('edit.apply', {
      ...common,
      value: { guid: 'entity-guid', transform },
    });
  });

  it('validates the value against the selected edit action', async () => {
    const snapshot = capabilities(['edit.apply'], ['rename', 'setTransform']);
    const { adapter, invoke } = fakeAdapter(snapshot);
    const registry = new BuiltInAgentToolRegistry(adapter);

    await expect(
      registry.invoke('edit.apply', {
        editSessionId: 'edit-1',
        expectedSceneRevision: 3,
        action: 'rename',
        value: { guid: 'entity-guid', transform },
      }),
    ).rejects.toThrow('Invalid arguments for edit.apply at value');
    expect(invoke).not.toHaveBeenCalled();
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
        value: { guid: 'entity-guid', transform },
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

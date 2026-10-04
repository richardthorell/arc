import { describe, expect, it, vi } from 'vitest';

import type { BuiltInAgentCapabilities } from '../common/builtInAgentTypes';
import { BuiltInAgentToolRegistry, builtInAgentToolDefinitions } from './builtInAgentToolRegistry';

describe('built-in agent entity selection tools', () => {
  const capabilities: BuiltInAgentCapabilities = {
    operations: ['selection.set', 'selection.clear'],
    editActions: [],
  };

  it('publishes selection tools as non-transactional harness operations', () => {
    const definitions = builtInAgentToolDefinitions(capabilities);
    expect(definitions.map((definition) => definition.name)).toEqual(['selection.set', 'selection.clear']);
    expect(definitions[0]?.inputSchema).toMatchObject({
      type: 'object',
      required: ['guid'],
      additionalProperties: false,
    });
    expect(definitions[1]?.inputSchema).toMatchObject({ type: 'object', additionalProperties: false });
  });

  it('invokes the same transport-neutral selection operation and validates GUIDs', async () => {
    const invoke = vi.fn(async (method: string) => ({ method, selectedGuids: ['floor-guid'] }));
    const registry = new BuiltInAgentToolRegistry({
      capabilities: vi.fn(async () => capabilities),
      invoke,
    });

    await expect(registry.invoke('selection.set', {})).rejects.toThrow('Invalid arguments for selection.set');
    expect(invoke).not.toHaveBeenCalled();

    const selected = await registry.invoke('selection.set', { guid: 'floor-guid' });
    expect(invoke).toHaveBeenCalledWith('selection.set', { guid: 'floor-guid' });
    expect(JSON.parse(selected.content)).toEqual({ method: 'selection.set', selectedGuids: ['floor-guid'] });

    await registry.invoke('selection.clear', {});
    expect(invoke).toHaveBeenLastCalledWith('selection.clear', {});
  });
});

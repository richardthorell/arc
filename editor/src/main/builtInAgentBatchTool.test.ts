import { describe, expect, it, vi } from 'vitest';

import type { BuiltInAgentCapabilities } from '../common/builtInAgentTypes';
import { BuiltInAgentToolRegistry, builtInAgentToolDefinitions } from './builtInAgentToolRegistry';

const capabilities: BuiltInAgentCapabilities = {
  operations: ['editor.applyBatch'],
  editActions: ['create', 'setTransform'],
};

describe('editor.applyBatch built-in tool', () => {
  it('projects explicit typed batch operation schemas to the model', () => {
    const definition = builtInAgentToolDefinitions(capabilities)[0];
    expect(definition?.name).toBe('editor.applyBatch');
    expect(definition?.inputSchema).toMatchObject({
      type: 'object',
      required: ['editSessionId', 'expectedSceneRevision', 'operations'],
      additionalProperties: false,
    });

    const serialized = JSON.stringify(definition?.inputSchema);
    expect(serialized).toContain('entity.create');
    expect(serialized).toContain('entity.setTransform');
    expect(serialized).toContain('tempId');
  });

  it('rejects invalid tempId dependencies before invoking the harness', async () => {
    const invoke = vi.fn(async () => ({ ok: true }));
    const registry = new BuiltInAgentToolRegistry({
      capabilities: vi.fn(async () => capabilities),
      invoke,
    });

    await expect(
      registry.invoke('editor.applyBatch', {
        editSessionId: 'edit-1',
        expectedSceneRevision: 4,
        operations: [
          { type: 'entity.rename', target: { tempId: 'cube' }, name: 'Too Early' },
          { type: 'entity.create', tempId: 'cube', kind: 'cube' },
        ],
      }),
    ).rejects.toThrow(/tempId/);
    expect(invoke).not.toHaveBeenCalled();
  });

  it('forwards one validated batch invocation to the harness', async () => {
    const invoke = vi.fn(async () => ({ sceneRevision: 6 }));
    const registry = new BuiltInAgentToolRegistry({
      capabilities: vi.fn(async () => capabilities),
      invoke,
    });
    const arguments_ = {
      editSessionId: 'edit-1',
      expectedSceneRevision: 4,
      operations: [
        { type: 'entity.create' as const, tempId: 'cube', kind: 'cube' },
        {
          type: 'entity.setTransform' as const,
          target: { tempId: 'cube' },
          transform: { position: [0, 0, 0], rotation: [0, 0, 0, 1], scale: [4, 4, 4] },
        },
      ],
    };

    await registry.invoke('editor.applyBatch', arguments_);
    expect(invoke).toHaveBeenCalledWith('editor.applyBatch', arguments_);
  });
});

import { describe, expect, it, vi } from 'vitest';

import { EditorOperationRegistry } from './editorOperationRegistry';

const schema = {
  parse(value: unknown) {
    if (!value || typeof value !== 'object' || typeof (value as { name?: unknown }).name !== 'string') {
      throw new Error('name is required');
    }
    return { name: (value as { name: string }).name };
  },
};

describe('EditorOperationRegistry', () => {
  it('registers namespaced operations and exposes deterministic summaries', () => {
    const registry = new EditorOperationRegistry();
    registry.register({
      id: 'terrain.scatter',
      description: 'Scatter terrain instances',
      schema,
      mutating: true,
      batchable: true,
      owner: 'terrain',
      execute: () => undefined,
    });
    registry.register({
      id: 'asset.inspect',
      description: 'Inspect an asset',
      schema,
      mutating: false,
      batchable: false,
      owner: 'assets',
      execute: () => undefined,
    });

    expect(registry.list().map((operation) => operation.id)).toEqual(['asset.inspect', 'terrain.scatter']);
  });

  it('rejects duplicate, unnamespaced, and unknown operations', async () => {
    const registry = new EditorOperationRegistry();
    const definition = {
      id: 'plugin.doThing',
      description: 'Do a thing',
      schema,
      mutating: false,
      batchable: false,
      owner: 'test-plugin',
      execute: () => undefined,
    };
    registry.register(definition);

    expect(() => registry.register(definition)).toThrow('already registered');
    expect(() => registry.register({ ...definition, id: 'invalid' })).toThrow('namespaced');
    await expect(registry.execute('missing.operation', {}, { capabilities: new Set() })).rejects.toThrow(
      'Unknown editor operation',
    );
  });

  it('validates input and capabilities before executing a registered operation', async () => {
    const registry = new EditorOperationRegistry();
    const execute = vi.fn(({ name }: { name: string }) => `hello ${name}`);
    registry.register({
      id: 'plugin.greet',
      description: 'Greet through a plugin operation',
      schema,
      mutating: false,
      batchable: true,
      requiredCapabilities: ['project.read'],
      owner: 'test-plugin',
      execute,
    });

    expect(registry.listAvailable(new Set())).toEqual([]);
    expect(registry.listAvailable(new Set(['project.read']))).toHaveLength(1);
    await expect(
      registry.execute('plugin.greet', { name: 'ARC' }, { capabilities: new Set(['project.read']) }),
    ).resolves.toBe('hello ARC');
    expect(execute).toHaveBeenCalledWith({ name: 'ARC' }, { capabilities: new Set(['project.read']) });
    await expect(registry.execute('plugin.greet', { name: 'ARC' }, { capabilities: new Set() })).rejects.toThrow(
      'requires capabilities',
    );
    await expect(registry.execute('plugin.greet', {}, { capabilities: new Set(['project.read']) })).rejects.toThrow(
      'name is required',
    );
  });

  it('supports plugin lifecycle unregister without removing a replacement', () => {
    const registry = new EditorOperationRegistry();
    const unregister = registry.register({
      id: 'plugin.action',
      description: 'Plugin action',
      schema,
      mutating: true,
      batchable: true,
      owner: 'test-plugin',
      execute: () => undefined,
    });

    expect(registry.has('plugin.action')).toBe(true);
    unregister();
    expect(registry.has('plugin.action')).toBe(false);
  });
});

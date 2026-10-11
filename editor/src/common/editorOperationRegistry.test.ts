import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import {
  EditorOperationRegistry,
  type EditorOperationDefinition,
  type EditorOperationContext,
} from './editorOperationRegistry';

const context = { capabilities: new Set<string>() };
const definition = (
  id = 'test.inspect',
): EditorOperationDefinition<EditorOperationContext, { name: string }, { name: string }> => ({
  id,
  version: 1,
  domain: 'test',
  description: 'Inspect a test resource',
  permissions: ['editor.read'],
  requiredCapabilities: [],
  effects: { reads: ['test-resource'], writes: [] },
  concurrency: 'independent-read',
  cache: 'revision',
  transaction: 'none',
  references: [],
  creates: [],
  input: z.object({ name: z.string() }).strict(),
  output: z.object({ name: z.string() }).strict(),
  execute: (input) => input,
});

describe('shared editor operation contract', () => {
  it('discovers bounded sorted summaries and exact schemas from the same registration', () => {
    const registry = new EditorOperationRegistry(() => {});
    registry.register(definition('test.zebra'));
    registry.register(definition('test.alpha'));
    const page = registry.list({ limit: 1 }, context);
    expect(page.operations.map((entry) => entry.id)).toEqual(['test.alpha']);
    expect(page.nextOffset).toBe(1);
    expect(page.operations[0]).not.toHaveProperty('inputSchema');
    const details = registry.describeMany({ ids: ['test.zebra'], catalogRevision: page.catalogRevision }, context);
    expect(details.operations[0].inputSchema).toMatchObject({
      type: 'object',
      required: ['name'],
      additionalProperties: false,
    });
    expect(details.operations[0].outputSchema).toMatchObject({ type: 'object', required: ['name'] });
    expect(() => registry.list({ limit: 101 }, context)).toThrow();
    expect(() => registry.describeMany({ ids: Array(17).fill('test.zebra') }, context)).toThrow();
  });

  it('rejects missing operations, stale catalogs, unsupported versions and invalid inputs before execution', async () => {
    const registry = new EditorOperationRegistry(() => {});
    const execute = vi.fn((input: { name: string }) => input);
    registry.register({ ...definition(), execute });
    const revision = registry.revision;
    await expect(registry.execute('test.missing', {}, context)).rejects.toMatchObject({ code: 'unknown_operation' });
    await expect(registry.execute('test.inspect', {}, context)).rejects.toMatchObject({ code: 'invalid_arguments' });
    await expect(
      registry.execute('test.inspect', { name: 'ARC' }, context, { version: 2, catalogRevision: revision }),
    ).rejects.toMatchObject({ code: 'unsupported_version' });
    registry.register(definition('test.another'));
    await expect(
      registry.execute('test.inspect', { name: 'ARC' }, context, { version: 1, catalogRevision: revision }),
    ).rejects.toMatchObject({ code: 'stale_catalog' });
    expect(execute).not.toHaveBeenCalled();
    expect(() => registry.list({ catalogRevision: revision }, context)).toThrow('catalog changed');
  });

  it('checks live capabilities, authorization, and output schemas', async () => {
    const authorize = vi.fn();
    const registry = new EditorOperationRegistry(authorize);
    registry.register({ ...definition(), requiredCapabilities: ['test.read'], execute: () => ({ name: 4 }) as never });
    expect(registry.describe('test.inspect', context)).toMatchObject({ available: false });
    await expect(registry.execute('test.inspect', { name: 'ARC' }, context)).rejects.toMatchObject({
      code: 'unavailable',
    });
    expect(authorize).not.toHaveBeenCalled();
    await expect(
      registry.execute('test.inspect', { name: 'ARC' }, { capabilities: new Set(['test.read']) }),
    ).rejects.toMatchObject({ code: 'invalid_result' });
    expect(authorize).toHaveBeenCalledTimes(1);
  });

  it('does not execute after failed authorization or unregister during authorization', async () => {
    const execute = vi.fn((input: { name: string }) => input);
    let unregister = () => {};
    const registry = new EditorOperationRegistry(() => unregister());
    unregister = registry.register({ ...definition(), execute });
    await expect(registry.execute('test.inspect', { name: 'ARC' }, context)).rejects.toMatchObject({
      code: 'stale_catalog',
    });
    expect(execute).not.toHaveBeenCalled();
    const denied = new EditorOperationRegistry(() => {
      throw new Error('denied');
    });
    denied.register({ ...definition(), execute });
    await expect(denied.execute('test.inspect', { name: 'ARC' }, context)).rejects.toThrow('denied');
    expect(execute).not.toHaveBeenCalled();
  });

  it('isolates descriptors and lifecycle generations, including across registry restarts', () => {
    const registry = new EditorOperationRegistry(() => {});
    const unregister = registry.register(definition());
    const revision = registry.revision;
    const view = registry.describe('test.inspect', context);
    view.effects = { reads: [], writes: ['fake'] };
    expect(registry.describe('test.inspect', context).effects.writes).toEqual([]);
    expect(() => registry.register(definition())).toThrow('already registered');
    unregister();
    registry.register(definition());
    unregister();
    expect(registry.has('test.inspect')).toBe(true);
    expect(registry.revision).not.toBe(revision);
    expect(new EditorOperationRegistry(() => {}).revision).not.toBe(registry.revision);
  });

  it('rejects unsafe write scheduling metadata', () => {
    const registry = new EditorOperationRegistry(() => {});
    expect(() => registry.register({ ...definition(), effects: { reads: [], writes: ['document'] } })).toThrow(
      'Writing operations require',
    );
  });
});

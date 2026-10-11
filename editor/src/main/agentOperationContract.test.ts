import { describe, expect, it, vi } from 'vitest';
import { z } from 'zod';
import { EditorAgentHarness, type AgentHarnessHost, type AgentHostResponse } from './editorAgentHarness';
import { BuiltInAgentAdapter } from './builtInAgentAdapter';
import { BuiltInAgentToolRegistry } from './builtInAgentToolRegistry';

const response = (): AgentHostResponse => ({
  kind: 'response',
  requestId: 1,
  succeeded: true,
  error: '',
  payload: {},
  sceneRevision: 1,
  worldEpoch: 1,
  frameRevision: 1,
});
const host: AgentHarnessHost = { query: async () => response(), command: async () => response() };

const testOperation = {
  id: 'test.inspect',
  version: 1,
  domain: 'test',
  description: 'Inspect a trusted test domain',
  permissions: ['editor.read'],
  requiredCapabilities: [],
  effects: { reads: ['test'], writes: [] },
  concurrency: 'independent-read' as const,
  cache: 'revision' as const,
  transaction: 'none' as const,
  references: [],
  creates: [],
  input: z.object({ value: z.number() }).strict(),
  output: z.object({ doubled: z.number() }).strict(),
  execute: ({ value }: { value: number }) => ({ doubled: value * 2 }),
};

describe('harness operation contract', () => {
  it('rejects ambiguous provider and legacy MCP names at registration', () => {
    const harness = new EditorAgentHarness(host);
    harness.registerOperation({ ...testOperation, id: 'test.fooBar' });
    expect(() => harness.registerOperation({ ...testOperation, id: 'test.foo.bar' })).toThrow('collision');
    expect(() => harness.registerOperation({ ...testOperation, id: 'finish.edit' })).toThrow('collision');
  });
  it('discovers and invokes a new domain through the unchanged built-in adapter', async () => {
    const harness = new EditorAgentHarness(host);
    const unregister = harness.registerOperation(testOperation);
    const adapter = new BuiltInAgentAdapter(harness);
    const tools = new BuiltInAgentToolRegistry(adapter);
    expect((await tools.definitions()).find((entry) => entry.name === 'test.inspect')).toMatchObject({
      inputSchema: { type: 'object', properties: { value: { type: 'number' } } },
    });
    expect(JSON.parse((await tools.invoke('test.inspect', { value: 3 })).content)).toEqual({ doubled: 6 });
    await expect(tools.invoke('test.inspect', { value: 'wrong' })).rejects.toMatchObject({ code: 'invalid_arguments' });
    const { catalogRevision } = await adapter.capabilities();
    unregister();
    harness.registerOperation(testOperation);
    await expect(
      tools.invoke('test.inspect', { value: 3, operationContract: { version: 1, catalogRevision } }),
    ).rejects.toMatchObject({ code: 'stale_catalog' });
    await expect(
      adapter.invoke('test.inspect', { value: 3, operationContract: { version: 1, catalogRevision } }),
    ).rejects.toMatchObject({ code: 'stale_catalog' });
    await adapter.dispose();
  });

  it('advertises unavailable operations but does not expose them as callable built-in tools', async () => {
    const harness = new EditorAgentHarness(host);
    const execute = vi.fn(testOperation.execute);
    harness.registerOperation({ ...testOperation, requiredCapabilities: ['missing.backend'], execute });
    const snapshot = await harness.invoke('agent.operations.describe', { ids: ['test.inspect'] }, 'test');
    expect(snapshot).toMatchObject({
      operations: [{ available: false, unavailableReason: expect.stringContaining('missing.backend') }],
    });
    expect(
      (await new BuiltInAgentToolRegistry(new BuiltInAgentAdapter(harness)).definitions()).some(
        (entry) => entry.name === 'test.inspect',
      ),
    ).toBe(false);
    await expect(harness.invoke('test.inspect', { value: 1 }, 'test')).rejects.toMatchObject({ code: 'unavailable' });
    expect(execute).not.toHaveBeenCalled();
  });

  it('enforces permissions and version checks for trusted mutation extensions', async () => {
    const harness = new EditorAgentHarness(host);
    const execute = vi.fn(testOperation.execute);
    harness.registerOperation({
      ...testOperation,
      execute,
      permissions: ['scene.edit'],
      effects: { reads: [], writes: ['test'] },
      concurrency: 'serial',
      cache: 'never',
    });
    await expect(harness.invoke('test.inspect', { value: 3 }, 'test')).rejects.toMatchObject({
      code: 'permission_denied',
    });
    expect(execute).not.toHaveBeenCalled();
    const request = harness.requestEdit('test', 'Test mutation');
    harness.approveEdit(request.id);
    await expect(harness.invoke('test.inspect', { value: 3 }, 'test')).resolves.toEqual({ doubled: 6 });
    expect(harness.status().audit).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ operation: 'test.inspect', succeeded: false }),
        expect.objectContaining({ operation: 'test.inspect', succeeded: true }),
      ]),
    );
  });

  it('keeps transient viewport and selection operations callable without scene edit permission', async () => {
    const harness = new EditorAgentHarness(host);
    const tools = new BuiltInAgentToolRegistry(new BuiltInAgentAdapter(harness));
    await expect(tools.invoke('selection.clear')).resolves.toMatchObject({ name: 'selection.clear' });
  });
});

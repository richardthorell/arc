import { describe, expect, it, vi } from 'vitest';

import type { BuiltInAgentToolExecutionResult } from '../../../common/builtInAgentTypes';
import type { AiToolCall } from '../../../common/aiRuntimeTypes';
import { AiAgentApprovalCoordinator } from './aiAgentApproval';

const call: AiToolCall = {
  id: 'call-1',
  name: 'edit.request',
  arguments: { label: 'Create cube' },
};

const pendingResult = (): BuiltInAgentToolExecutionResult => {
  const content = JSON.stringify({
    id: 'request-1',
    clientId: 'arc.builtin-ai',
    clientName: 'ARC Built-in AI',
    label: 'Create cube',
    requestedAt: '2026-10-04T05:40:00.000Z',
    state: 'pending',
  });
  return {
    name: 'edit.request',
    operation: 'edit.request',
    content,
    truncated: false,
    originalBytes: new TextEncoder().encode(content).byteLength,
  };
};

describe('AiAgentApprovalCoordinator', () => {
  it('keeps the tool call pending until the exact harness request is approved', async () => {
    const approve = vi.fn(async () => true);
    const coordinator = new AiAgentApprovalCoordinator({
      invokeTool: vi.fn(async () => pendingResult()),
      approve,
      deny: vi.fn(async () => true),
    });

    let settled = false;
    const resultPromise = coordinator.invokeTool(call).then((result) => {
      settled = true;
      return result;
    });
    await Promise.resolve();
    expect(settled).toBe(false);

    await expect(coordinator.approve('request-1')).resolves.toBe(true);
    const result = await resultPromise;
    expect(approve).toHaveBeenCalledTimes(1);
    expect(approve).toHaveBeenCalledWith('request-1');
    expect(JSON.parse(result.content)).toMatchObject({ id: 'request-1', state: 'approved' });
  });

  it('returns a denied decision to the model instead of allowing the edit turn to disappear', async () => {
    const coordinator = new AiAgentApprovalCoordinator({
      invokeTool: vi.fn(async () => pendingResult()),
      approve: vi.fn(async () => true),
      deny: vi.fn(async () => true),
    });

    const resultPromise = coordinator.invokeTool(call);
    await Promise.resolve();
    await expect(coordinator.deny('request-1')).resolves.toBe(true);
    const result = await resultPromise;
    expect(result).toMatchObject({ operation: 'edit.request' });
    expect(JSON.parse(result.content)).toMatchObject({ state: 'denied' });
  });

  it('auto-approves through the harness without bypassing edit.request', async () => {
    const invokeTool = vi.fn(async () => pendingResult());
    const approve = vi.fn(async () => true);
    const coordinator = new AiAgentApprovalCoordinator({
      invokeTool,
      approve,
      deny: vi.fn(async () => true),
    });
    coordinator.setMode('auto');

    const result = await coordinator.invokeTool(call);
    expect(invokeTool).toHaveBeenCalledWith(call, undefined);
    expect(approve).toHaveBeenCalledWith('request-1');
    expect(JSON.parse(result.content)).toMatchObject({ state: 'approved' });
  });

  it('cancels and denies a pending harness approval when the agent turn is stopped', async () => {
    const deny = vi.fn(async () => true);
    const coordinator = new AiAgentApprovalCoordinator({
      invokeTool: vi.fn(async () => pendingResult()),
      approve: vi.fn(async () => true),
      deny,
    });
    const controller = new AbortController();
    const resultPromise = coordinator.invokeTool(call, controller.signal);
    await Promise.resolve();
    controller.abort();

    await expect(resultPromise).rejects.toThrow('cancelled');
    expect(deny).toHaveBeenCalledWith('request-1');
  });
});

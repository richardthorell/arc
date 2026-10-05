import { describe, expect, it, vi } from 'vitest';

import type { AiRuntimeRequest, AiRuntimeStreamEvent, AiToolCall } from '../../../common/aiRuntimeTypes';
import { runAiAgentToolLoop } from './aiAgentToolLoop';
import { recordConversationTaskUpdate } from './aiConversationTaskProgress';

const collect = async (stream: AsyncIterable<AiRuntimeStreamEvent>): Promise<AiRuntimeStreamEvent[]> => {
  const events: AiRuntimeStreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
};

const request: AiRuntimeRequest = {
  conversationId: 'task-progress',
  messages: [{ id: 'user', role: 'user', content: 'Create and configure an entity' }],
};

describe('AI agent task progress', () => {
  it('emits one mutable task lifecycle linked to every tool call in a provider step', async () => {
    let providerTurn = 0;
    const execute = vi.fn(() =>
      (async function* () {
        ++providerTurn;
        if (providerTurn === 1) {
          yield {
            type: 'tool-call' as const,
            call: { id: 'create', name: 'edit.apply', arguments: { action: 'create' } },
          };
          yield {
            type: 'tool-call' as const,
            call: { id: 'rename', name: 'edit.apply', arguments: { action: 'rename' } },
          };
          yield { type: 'done' as const, finishReason: 'tool_calls' as const };
          return;
        }
        yield { type: 'done' as const, finishReason: 'stop' as const };
      })(),
    );
    const invokeTool = vi.fn(async (call: AiToolCall) => ({
      name: call.name,
      operation: call.name,
      content: '{}',
      truncated: false,
      originalBytes: 2,
    }));

    const events = await collect(runAiAgentToolLoop(request, execute, invokeTool));
    const tasks = events.filter(
      (event): event is Extract<AiRuntimeStreamEvent, { type: 'task-update' }> => event.type === 'task-update',
    );

    expect(tasks).toHaveLength(2);
    expect(tasks[0]!.task).toEqual({
      id: 'agent-step-0',
      title: 'Run 2 editor operations',
      state: 'in_progress',
      agentStep: 0,
      toolCallIds: ['create', 'rename'],
    });
    expect(tasks[1]!.task).toMatchObject({
      id: 'agent-step-0',
      state: 'completed',
      toolCallIds: ['create', 'rename'],
    });
  });

  it('marks the task failed while retaining the linked tool calls when one operation fails', async () => {
    let providerTurn = 0;
    const execute = () =>
      (async function* () {
        ++providerTurn;
        if (providerTurn === 1) {
          yield {
            type: 'tool-call' as const,
            call: { id: 'broken', name: 'editor.applyBatch', arguments: {} },
          };
          yield { type: 'done' as const, finishReason: 'tool_calls' as const };
          return;
        }
        yield { type: 'done' as const, finishReason: 'stop' as const };
      })();

    const events = await collect(
      runAiAgentToolLoop(request, execute, async () => {
        throw new Error('batch rejected');
      }),
    );
    const failed = events.find(
      (event): event is Extract<AiRuntimeStreamEvent, { type: 'task-update' }> =>
        event.type === 'task-update' && event.task.state === 'failed',
    );

    expect(failed?.task).toMatchObject({
      id: 'agent-step-0',
      toolCallIds: ['broken'],
      detail: 'Failed while running editor.applyBatch',
    });
  });

  it('keeps model-authored plans grouped, semantic, and local to the runtime', async () => {
    let providerTurn = 0;
    const execute = vi.fn(() =>
      (async function* (): AsyncGenerator<AiRuntimeStreamEvent> {
        ++providerTurn;
        if (providerTurn === 1) {
          yield {
            type: 'tool-call',
            call: {
              id: 'plan-1',
              name: 'agent.updatePlan',
              arguments: {
                planId: 'capsule-plan',
                title: 'Create green capsule',
                steps: [
                  { id: 'inspect', title: 'Inspect scene', state: 'completed' },
                  {
                    id: 'build',
                    title: 'Build capsule',
                    state: 'in_progress',
                    children: [
                      { id: 'create', title: 'Create capsule', state: 'in_progress' },
                      { id: 'verify', title: 'Verify result', state: 'planned' },
                    ],
                  },
                ],
              },
            },
          };
          yield {
            type: 'tool-call',
            call: { id: 'mutate', name: 'editor.applyBatch', arguments: { operations: [] } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        if (providerTurn === 2) {
          yield {
            type: 'tool-call',
            call: {
              id: 'plan-2',
              name: 'agent.updatePlan',
              arguments: {
                planId: 'capsule-plan',
                title: 'Create green capsule',
                steps: [
                  { id: 'inspect', title: 'Inspect scene', state: 'completed' },
                  {
                    id: 'build',
                    title: 'Build capsule',
                    state: 'completed',
                    children: [
                      { id: 'create', title: 'Create capsule', state: 'completed' },
                      { id: 'verify', title: 'Verify result', state: 'completed' },
                    ],
                  },
                ],
              },
            },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        yield { type: 'done', finishReason: 'stop' };
      })(),
    );
    const invokeTool = vi.fn(async (call: AiToolCall) => ({
      name: call.name,
      operation: call.name,
      content: '{}',
      truncated: false,
      originalBytes: 2,
    }));

    const events = await collect(runAiAgentToolLoop(request, execute, invokeTool));
    const tasks = events.filter(
      (event): event is Extract<AiRuntimeStreamEvent, { type: 'task-update' }> => event.type === 'task-update',
    );

    expect(invokeTool).toHaveBeenCalledTimes(1);
    expect(invokeTool).toHaveBeenCalledWith(expect.objectContaining({ id: 'mutate', name: 'editor.applyBatch' }));
    expect(events.some((event) => event.type === 'tool-call' && event.call.name === 'agent.updatePlan')).toBe(false);
    expect(tasks[0]?.task).toMatchObject({
      id: 'capsule-plan',
      title: 'Create green capsule',
      state: 'in_progress',
      children: [
        { id: 'inspect', title: 'Inspect scene', state: 'completed' },
        {
          id: 'build',
          title: 'Build capsule',
          state: 'in_progress',
          children: [
            { id: 'create', title: 'Create capsule', state: 'in_progress' },
            { id: 'verify', title: 'Verify result', state: 'planned' },
          ],
        },
      ],
    });
    const linked = tasks.find((event) =>
      event.task.children?.some((step) =>
        step.children?.some((child) => child.id === 'create' && child.toolCallIds?.includes('mutate')),
      ),
    );
    expect(linked).toBeDefined();
    expect(tasks.at(-1)?.task).toMatchObject({
      id: 'capsule-plan',
      state: 'completed',
      children: [
        { id: 'inspect', state: 'completed' },
        { id: 'build', state: 'completed' },
      ],
    });
  });

  it('keeps retries and commit cleanup inside one semantic plan', async () => {
    let providerTurn = 0;
    let mutationAttempt = 0;
    const execute = () =>
      (async function* (): AsyncGenerator<AiRuntimeStreamEvent> {
        ++providerTurn;
        if (providerTurn === 1) {
          yield {
            type: 'tool-call',
            call: {
              id: 'plan',
              name: 'agent.updatePlan',
              arguments: {
                planId: 'trees-plan',
                title: 'Build two trees',
                steps: [
                  { id: 'inspect', title: 'Inspect primitive options', state: 'completed' },
                  { id: 'build', title: 'Build two trees', state: 'in_progress' },
                ],
              },
            },
          };
          yield {
            type: 'tool-call',
            call: { id: 'mutate-1', name: 'editor.applyBatch', arguments: { operations: [] } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        if (providerTurn === 2) {
          yield {
            type: 'tool-call',
            call: { id: 'mutate-2', name: 'editor.applyBatch', arguments: { operations: [] } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        if (providerTurn === 3) {
          yield {
            type: 'tool-call',
            call: { id: 'commit', name: 'edit.commit', arguments: { editSessionId: 'edit-1' } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        yield { type: 'done', finishReason: 'stop' };
      })();

    const invokeTool = vi.fn(async (call: AiToolCall) => {
      if (call.name === 'editor.applyBatch') {
        ++mutationAttempt;
        if (mutationAttempt === 1) throw new Error('transient mutation failure');
      }
      return {
        name: call.name,
        operation: call.name,
        content: '{}',
        truncated: false,
        originalBytes: 2,
      };
    });

    const events = await collect(runAiAgentToolLoop(request, execute, invokeTool));
    const tasks = events.filter(
      (event): event is Extract<AiRuntimeStreamEvent, { type: 'task-update' }> => event.type === 'task-update',
    );

    expect(tasks.length).toBeGreaterThan(0);
    expect(tasks.every((event) => event.task.id === 'trees-plan')).toBe(true);
    expect(tasks.some((event) => event.task.id.startsWith('agent-step-'))).toBe(false);
    expect(tasks.at(-1)?.task).toMatchObject({
      id: 'trees-plan',
      state: 'completed',
      children: [
        { id: 'inspect', state: 'completed' },
        { id: 'build', state: 'completed', toolCallIds: ['mutate-1', 'mutate-2'] },
      ],
    });
  });

  it('updates a persisted task reference in place instead of appending progress spam', () => {
    const started = recordConversationTaskUpdate(
      undefined,
      { id: 'task-1', title: 'Build scene', state: 'in_progress', toolCallIds: ['call-1'] },
      '2026-10-05T05:00:00Z',
    );
    const completed = recordConversationTaskUpdate(
      started,
      { id: 'task-1', title: 'Build scene', state: 'completed', toolCallIds: ['call-1'] },
      '2026-10-05T05:00:02Z',
    );

    expect(completed).toHaveLength(1);
    expect(completed[0]).toMatchObject({
      id: 'task-1',
      state: 'completed',
      startedAt: '2026-10-05T05:00:00Z',
      completedAt: '2026-10-05T05:00:02Z',
    });
  });
});

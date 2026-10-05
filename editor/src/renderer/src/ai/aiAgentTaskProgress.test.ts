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

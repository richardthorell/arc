import { describe, expect, it, vi } from 'vitest';

import type { AiRuntimeRequest, AiRuntimeStreamEvent, AiToolCall } from '../../../common/aiRuntimeTypes';
import { runAiAgentToolLoop } from './aiAgentToolLoop';
import { recordConversationTaskUpdate } from './aiConversationTaskProgress';

const collect = async (stream: AsyncIterable<AiRuntimeStreamEvent>): Promise<AiRuntimeStreamEvent[]> => {
  const events: AiRuntimeStreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
};

const flattenState = (task: Extract<AiRuntimeStreamEvent, { type: 'task-update' }>['task']): string[] => [
  task.state,
  ...(task.children?.flatMap(flattenState) ?? []),
];

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

  it('keeps retryable revision conflicts in progress instead of flashing failed', async () => {
    let providerTurn = 0;
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
                planId: 'scene-plan',
                title: 'Edit scene',
                steps: [{ id: 'apply', title: 'Apply scene changes', state: 'in_progress' }],
              },
            },
          };
          yield {
            type: 'tool-call',
            call: {
              id: 'mutate-1',
              name: 'editor.applyBatch',
              arguments: { expectedSceneRevision: 1, operations: [] },
            },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        yield { type: 'done', finishReason: 'stop' };
      })();

    const events = await collect(
      runAiAgentToolLoop(request, execute, async () => {
        throw new Error('Edit session expects scene revision 2');
      }),
    );
    const tasks = events.filter(
      (event): event is Extract<AiRuntimeStreamEvent, { type: 'task-update' }> => event.type === 'task-update',
    );

    expect(tasks.some((event) => flattenState(event.task).includes('failed'))).toBe(false);
    expect(tasks.at(-1)?.task).toMatchObject({
      id: 'scene-plan',
      state: 'in_progress',
      children: [{ id: 'apply', state: 'in_progress', detail: 'Retrying after editor.applyBatch' }],
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

  it('advances plan steps when the next semantic tool turn begins and preserves omitted completed history', async () => {
    let providerTurn = 0;
    const execute = () =>
      (async function* (): AsyncGenerator<AiRuntimeStreamEvent> {
        ++providerTurn;
        if (providerTurn === 1) {
          yield {
            type: 'tool-call',
            call: {
              id: 'plan-1',
              name: 'agent.updatePlan',
              arguments: {
                planId: 'playground-plan',
                title: 'Build playground',
                steps: [
                  { id: 'layout', title: 'Lay out the playground', state: 'in_progress' },
                  { id: 'decorate', title: 'Add colorful props', state: 'planned' },
                  { id: 'tag', title: 'Tag the result', state: 'planned' },
                ],
              },
            },
          };
          yield {
            type: 'tool-call',
            call: { id: 'layout-tool', name: 'editor.applyBatch', arguments: { operations: [] } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        if (providerTurn === 2) {
          yield {
            type: 'tool-call',
            call: { id: 'decorate-tool', name: 'editor.applyBatch', arguments: { operations: [] } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        if (providerTurn === 3) {
          yield {
            type: 'tool-call',
            call: {
              id: 'plan-2',
              name: 'agent.updatePlan',
              arguments: {
                planId: 'playground-plan',
                title: 'Build playground',
                steps: [{ id: 'tag', title: 'Tag the result', state: 'in_progress' }],
              },
            },
          };
          yield {
            type: 'tool-call',
            call: { id: 'tag-tool', name: 'editor.applyBatch', arguments: { operations: [] } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        yield { type: 'done', finishReason: 'stop' };
      })();

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

    const firstTurnStates = tasks
      .filter((event) => event.task.children?.some((step) => step.id === 'layout' && step.toolCallIds?.includes('layout-tool')))
      .map((event) => event.task.children?.find((step) => step.id === 'layout')?.state);
    expect(firstTurnStates).toContain('in_progress');
    expect(firstTurnStates).not.toContain('completed');

    const afterLayout = tasks.find((event) =>
      event.task.children?.some((step) => step.id === 'layout' && step.state === 'completed') &&
      event.task.children?.some((step) => step.id === 'decorate' && step.state === 'in_progress'),
    );
    expect(afterLayout).toBeDefined();

    const afterDecorate = tasks.find((event) =>
      event.task.children?.some((step) => step.id === 'decorate' && step.state === 'completed') &&
      event.task.children?.some((step) => step.id === 'tag' && step.state === 'in_progress'),
    );
    expect(afterDecorate).toBeDefined();

    const partialPlanUpdate = tasks.find(
      (event) =>
        event.task.children?.some((step) => step.id === 'tag' && step.state === 'in_progress') &&
        event.task.children?.some((step) => step.id === 'layout') &&
        event.task.children?.some((step) => step.id === 'decorate'),
    );
    expect(partialPlanUpdate?.task.children).toHaveLength(3);

    expect(tasks.at(-1)?.task).toMatchObject({
      id: 'playground-plan',
      state: 'completed',
      children: [
        { id: 'layout', state: 'completed' },
        { id: 'decorate', state: 'completed' },
        { id: 'tag', state: 'completed' },
      ],
    });
  });

  it('keeps a failed middle step active while recovering and preserves the final step', async () => {
    let providerTurn = 0;
    let decorateAttempt = 0;
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
                planId: 'playground-plan',
                title: 'Build playground',
                steps: [
                  { id: 'layout', title: 'Lay out playground', state: 'in_progress' },
                  { id: 'decorate', title: 'Add colorful props', state: 'planned' },
                  { id: 'tag', title: 'Tag result', state: 'planned' },
                ],
              },
            },
          };
          yield {
            type: 'tool-call',
            call: { id: 'layout-tool', name: 'editor.applyBatch', arguments: { operations: [] } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        if (providerTurn === 2 || providerTurn === 3) {
          yield {
            type: 'tool-call',
            call: {
              id: `decorate-${providerTurn}`,
              name: 'editor.applyBatch',
              arguments: { operations: [] },
            },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        if (providerTurn === 4) {
          yield {
            type: 'tool-call',
            call: { id: 'tag-tool', name: 'editor.applyBatch', arguments: { operations: [] } },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        yield {
          type: 'tool-call',
          call: {
            id: 'final-plan',
            name: 'agent.updatePlan',
            arguments: {
              planId: 'playground-plan',
              title: 'Build playground',
              steps: [
                { id: 'layout', title: 'Lay out playground', state: 'completed' },
                { id: 'decorate', title: 'Add colorful props', state: 'completed' },
                { id: 'tag', title: 'Tag result', state: 'completed' },
              ],
            },
          },
        };
        yield { type: 'done', finishReason: 'tool_calls' };
      })();

    const invokeTool = vi.fn(async (call: AiToolCall) => {
      if (call.id.startsWith('decorate-')) {
        ++decorateAttempt;
        if (decorateAttempt === 1) throw new Error('temporary editor mutation failure');
      }
      return {
        name: call.name,
        operation: call.name,
        content: '{}',
        truncated: false,
        originalBytes: 2,
      };
    });

    const events = await collect(runAiAgentToolLoop(request, execute, invokeTool, { maximumSteps: 6 }));
    const tasks = events.filter(
      (event): event is Extract<AiRuntimeStreamEvent, { type: 'task-update' }> => event.type === 'task-update',
    );

    expect(tasks.some((event) => flattenState(event.task).includes('failed'))).toBe(false);

    const retrying = tasks.find((event) =>
      event.task.children?.some(
        (step) => step.id === 'decorate' && step.state === 'in_progress' && step.detail?.startsWith('Retrying after '),
      ),
    );
    expect(retrying?.task.children).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'layout', state: 'completed' }),
        expect.objectContaining({ id: 'decorate', state: 'in_progress' }),
        expect.objectContaining({ id: 'tag', state: 'planned' }),
      ]),
    );

    const recovered = tasks.find((event) =>
      event.task.children?.some((step) => step.id === 'decorate' && step.state === 'completed') &&
      event.task.children?.some((step) => step.id === 'tag' && step.state === 'in_progress'),
    );
    expect(recovered).toBeDefined();
    expect(tasks.at(-1)?.task.children).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ id: 'layout', state: 'completed' }),
        expect.objectContaining({ id: 'decorate', state: 'completed' }),
        expect.objectContaining({ id: 'tag', state: 'completed' }),
      ]),
    );
  });

  it('does not advance a task for edit control calls before the semantic mutation begins', async () => {
    let providerTurn = 0;
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
                planId: 'robot-plan',
                title: 'Build robot',
                steps: [
                  { id: 'design', title: 'Design robot', state: 'completed' },
                  { id: 'build', title: 'Build robot', state: 'in_progress' },
                  { id: 'verify', title: 'Verify robot', state: 'planned' },
                ],
              },
            },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        const calls: Record<number, AiToolCall> = {
          2: { id: 'request', name: 'edit.request', arguments: {} },
          3: { id: 'begin', name: 'edit.begin', arguments: {} },
          4: { id: 'apply', name: 'editor.applyBatch', arguments: { operations: [] } },
          5: { id: 'inspect', name: 'scene.getEntity', arguments: {} },
        };
        const call = calls[providerTurn];
        if (call) {
          yield { type: 'tool-call', call };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        yield { type: 'done', finishReason: 'stop' };
      })();

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

    const beforeApply = tasks.filter((event) =>
      event.task.children?.some((step) => step.id === 'build' && step.state === 'in_progress'),
    );
    expect(beforeApply.some((event) =>
      event.task.children?.some((step) => step.id === 'build' && step.toolCallIds?.includes('request')),
    )).toBe(false);
    expect(beforeApply.some((event) =>
      event.task.children?.some((step) => step.id === 'build' && step.toolCallIds?.includes('begin')),
    )).toBe(false);

    const applyLinked = tasks.find((event) =>
      event.task.children?.some(
        (step) => step.id === 'build' && step.state === 'in_progress' && step.toolCallIds?.includes('apply'),
      ),
    );
    expect(applyLinked).toBeDefined();

    const verifyActive = tasks.find((event) =>
      event.task.children?.some((step) => step.id === 'build' && step.state === 'completed') &&
      event.task.children?.some((step) => step.id === 'verify' && step.state === 'in_progress'),
    );
    expect(verifyActive).toBeDefined();
  });

  it('keeps the final task active across additional semantic verification turns', async () => {
    let providerTurn = 0;
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
                planId: 'verify-plan',
                title: 'Verify result',
                steps: [{ id: 'verify', title: 'Verify result', state: 'in_progress' }],
              },
            },
          };
          yield {
            type: 'tool-call',
            call: { id: 'inspect-1', name: 'scene.getEntity', arguments: {} },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        if (providerTurn === 2) {
          yield {
            type: 'tool-call',
            call: { id: 'inspect-2', name: 'viewport.debug', arguments: {} },
          };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        yield {
          type: 'tool-call',
          call: {
            id: 'final-plan',
            name: 'agent.updatePlan',
            arguments: {
              planId: 'verify-plan',
              title: 'Verify result',
              steps: [{ id: 'verify', title: 'Verify result', state: 'completed' }],
            },
          },
        };
        yield { type: 'done', finishReason: 'tool_calls' };
      })();

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

    const secondInspect = tasks.find((event) =>
      event.task.children?.some(
        (step) => step.id === 'verify' && step.state === 'in_progress' && step.toolCallIds?.includes('inspect-2'),
      ),
    );
    expect(secondInspect).toBeDefined();
    expect(tasks.filter((event) => event.task.children?.some((step) => step.id === 'verify' && step.state === 'completed')))
      .toHaveLength(1);
  });

  it('does not spend the semantic tool budget on edit control-only turns', async () => {
    let providerTurn = 0;
    const execute = () =>
      (async function* (): AsyncGenerator<AiRuntimeStreamEvent> {
        ++providerTurn;
        const calls: AiToolCall[] = [
          { id: 'request', name: 'edit.request', arguments: {} },
          { id: 'begin', name: 'edit.begin', arguments: {} },
          { id: 'commit', name: 'edit.commit', arguments: {} },
          { id: 'observe-1', name: 'scene.overview', arguments: {} },
          { id: 'observe-2', name: 'viewport.observe', arguments: {} },
        ];
        const call = calls[providerTurn - 1];
        if (call) {
          yield { type: 'tool-call', call };
          yield { type: 'done', finishReason: 'tool_calls' };
          return;
        }
        yield { type: 'done', finishReason: 'stop' };
      })();

    const invokeTool = vi.fn(async (call: AiToolCall) => ({
      name: call.name,
      operation: call.name,
      content: '{}',
      truncated: false,
      originalBytes: 2,
    }));

    const events = await collect(runAiAgentToolLoop(request, execute, invokeTool, { maximumSteps: 2 }));

    expect(events.some((event) => event.type === 'error' && event.message.includes('maximum of 2 tool steps'))).toBe(false);
    expect(invokeTool).toHaveBeenCalledTimes(5);
  });

  it('only recovers a failed persisted task when a new tool call proves a retry', () => {
    const failed = recordConversationTaskUpdate(
      undefined,
      { id: 'build', title: 'Build two trees', state: 'failed', toolCallIds: ['mutate-1'] },
      '2026-10-05T05:00:00Z',
    );
    const modelOnly = recordConversationTaskUpdate(
      failed,
      { id: 'build', title: 'Build two trees', state: 'completed', toolCallIds: ['mutate-1'] },
      '2026-10-05T05:00:01Z',
    );
    expect(modelOnly[0]?.state).toBe('failed');

    const retrying = recordConversationTaskUpdate(
      modelOnly,
      { id: 'build', title: 'Build two trees', state: 'in_progress', toolCallIds: ['mutate-1', 'mutate-2'] },
      '2026-10-05T05:00:02Z',
    );
    expect(retrying[0]?.state).toBe('in_progress');

    const completed = recordConversationTaskUpdate(
      retrying,
      { id: 'build', title: 'Build two trees', state: 'completed', toolCallIds: ['mutate-1', 'mutate-2'] },
      '2026-10-05T05:00:03Z',
    );
    expect(completed[0]?.state).toBe('completed');
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

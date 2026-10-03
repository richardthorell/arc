import { afterEach, describe, expect, it, vi } from 'vitest';

import type { AiRuntimeRequest, AiRuntimeStreamEvent } from '../../../common/aiRuntimeTypes';
import { runAiAgentToolLoop } from './aiAgentToolLoop';

const request = (signal?: AbortSignal): AiRuntimeRequest => ({
  conversationId: 'conversation-1',
  messages: [{ id: 'user-1', role: 'user', content: 'Inspect the floor' }],
  tools: [
    {
      name: 'scene.findEntities',
      description: 'Find scene entities.',
      inputSchema: { type: 'object' },
    },
  ],
  signal,
});

const collect = async (stream: AsyncIterable<AiRuntimeStreamEvent>): Promise<AiRuntimeStreamEvent[]> => {
  const events: AiRuntimeStreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
};

afterEach(() => {
  vi.useRealTimers();
});

describe('AI agent tool loop', () => {
  it('executes requested tools and continues the provider until it returns a final answer', async () => {
    const execute = vi.fn((runtimeRequest: AiRuntimeRequest) =>
      (async function* () {
        const resultMessage = runtimeRequest.messages.find((message) => message.role === 'tool');
        if (!resultMessage) {
          yield {
            type: 'tool-call' as const,
            call: { id: 'call-1', name: 'scene.findEntities', arguments: { search: 'Floor' } },
          };
          yield { type: 'done' as const, finishReason: 'tool_calls' as const };
          return;
        }

        expect(resultMessage.toolResult).toMatchObject({
          toolCallId: 'call-1',
          name: 'scene.findEntities',
          isError: undefined,
        });
        yield { type: 'delta' as const, text: 'I found the floor.' };
        yield { type: 'done' as const, finishReason: 'stop' as const };
      })(),
    );
    const invokeTool = vi.fn(async () => ({
      name: 'scene.findEntities',
      operation: 'scene.findEntities',
      content: '{"entities":[{"guid":"floor-guid","name":"Floor"}]}',
      truncated: false,
      originalBytes: 52,
    }));

    const events = await collect(runAiAgentToolLoop(request(), execute, invokeTool));

    expect(invokeTool).toHaveBeenCalledWith({
      id: 'call-1',
      name: 'scene.findEntities',
      arguments: { search: 'Floor' },
    });
    expect(events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({ type: 'tool-call' }),
        expect.objectContaining({ type: 'tool-result' }),
        { type: 'delta', text: 'I found the floor.' },
        { type: 'done', finishReason: 'stop' },
      ]),
    );
    expect(events.filter((event) => event.type === 'done')).toEqual([{ type: 'done', finishReason: 'stop' }]);
    expect(execute).toHaveBeenCalledTimes(2);
  });

  it('feeds normalized tool failures back to the model so it can recover', async () => {
    const execute = vi.fn((runtimeRequest: AiRuntimeRequest) =>
      (async function* () {
        const resultMessage = runtimeRequest.messages.find((message) => message.role === 'tool');
        if (!resultMessage) {
          yield {
            type: 'tool-call' as const,
            call: { id: 'call-1', name: 'scene.findEntities', arguments: { search: 'Floor' } },
          };
          yield { type: 'done' as const, finishReason: 'tool_calls' as const };
          return;
        }

        expect(resultMessage.toolResult).toMatchObject({ isError: true });
        expect(String(resultMessage.content)).toContain('ARC tool error: harness unavailable');
        yield { type: 'delta' as const, text: 'I could not inspect the floor.' };
        yield { type: 'done' as const, finishReason: 'stop' as const };
      })(),
    );

    const events = await collect(
      runAiAgentToolLoop(request(), execute, async () => {
        throw new Error('harness unavailable');
      }),
    );

    expect(events).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          type: 'tool-result',
          result: expect.objectContaining({ isError: true }),
        }),
        { type: 'delta', text: 'I could not inspect the floor.' },
      ]),
    );
  });

  it('stops further tool execution when the request is cancelled', async () => {
    const controller = new AbortController();
    const invokeTool = vi.fn();
    const execute = () =>
      (async function* () {
        yield {
          type: 'tool-call' as const,
          call: { id: 'call-1', name: 'scene.findEntities', arguments: { search: 'Floor' } },
        };
        yield { type: 'done' as const, finishReason: 'tool_calls' as const };
      })();
    const iterator = runAiAgentToolLoop(request(controller.signal), execute, invokeTool)[Symbol.asyncIterator]();

    expect((await iterator.next()).value).toMatchObject({ type: 'tool-call' });
    controller.abort();
    expect((await iterator.next()).done).toBe(true);
    expect(invokeTool).not.toHaveBeenCalled();
  });

  it('terminates runaway tool loops at the configured maximum depth', async () => {
    let call = 0;
    const execute = () =>
      (async function* () {
        call += 1;
        yield {
          type: 'tool-call' as const,
          call: { id: `call-${call}`, name: 'scene.findEntities', arguments: { search: 'Floor' } },
        };
        yield { type: 'done' as const, finishReason: 'tool_calls' as const };
      })();
    const invokeTool = vi.fn(async () => ({
      name: 'scene.findEntities',
      operation: 'scene.findEntities',
      content: '{}',
      truncated: false,
      originalBytes: 2,
    }));

    const events = await collect(runAiAgentToolLoop(request(), execute, invokeTool, { maximumSteps: 2 }));

    expect(events.at(-1)).toMatchObject({
      type: 'error',
      code: 'tool',
      message: 'AI agent reached the maximum of 2 tool steps',
    });
    expect(invokeTool).toHaveBeenCalledTimes(1);
  });

  it('aborts a provider step that exceeds the configured timeout', async () => {
    vi.useFakeTimers();
    const execute = (runtimeRequest: AiRuntimeRequest) =>
      (async function* () {
        await new Promise<void>((resolve) =>
          runtimeRequest.signal?.addEventListener('abort', () => resolve(), { once: true }),
        );
      })();
    const iterator = runAiAgentToolLoop(request(), execute, vi.fn(), { stepTimeoutMs: 25 })[Symbol.asyncIterator]();

    const pending = iterator.next();
    await vi.advanceTimersByTimeAsync(25);
    const event = await pending;

    expect(event.value).toMatchObject({
      type: 'error',
      code: 'provider',
      message: 'AI agent step 1 exceeded the 25 ms timeout',
    });
  });
});

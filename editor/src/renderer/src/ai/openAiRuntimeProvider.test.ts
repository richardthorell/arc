import { describe, expect, it, vi } from 'vitest';

import { AI_RUNTIME_SCHEMA_VERSION, type AiRuntimeStreamEvent } from '../../../common/aiRuntimeTypes';
import { streamOpenAiRuntime, type ArcAiRuntimeBridge } from './openAiRuntimeProvider';

const collect = async (stream: AsyncIterable<AiRuntimeStreamEvent>) => {
  const events: AiRuntimeStreamEvent[] = [];
  for await (const event of stream) events.push(event);
  return events;
};

describe('streamOpenAiRuntime', () => {
  it('serializes provider-neutral requests and forwards canonical stream events', async () => {
    let listener: Parameters<ArcAiRuntimeBridge['onEvent']>[0] | null = null;
    const bridge: ArcAiRuntimeBridge = {
      start: vi.fn(async (request) => {
        queueMicrotask(() => {
          listener?.({ requestId: request.requestId, event: { type: 'delta', text: 'Hello' } });
          listener?.({ requestId: request.requestId, event: { type: 'usage', usage: { totalTokens: 7 } } });
          listener?.({ requestId: request.requestId, event: { type: 'done', finishReason: 'stop' } });
        });
      }),
      cancel: vi.fn(async () => true),
      onEvent: (callback) => {
        listener = callback;
        return () => {
          listener = null;
        };
      },
    };
    const controller = new AbortController();

    await expect(
      collect(
        streamOpenAiRuntime(
          'gpt-5.6-sol',
          {
            conversationId: 'conversation-1',
            messages: [{ id: 'message-1', role: 'user', content: 'Hello' }],
            signal: controller.signal,
          },
          bridge,
        ),
      ),
    ).resolves.toEqual([
      { type: 'delta', text: 'Hello' },
      { type: 'usage', usage: { totalTokens: 7 } },
      { type: 'done', finishReason: 'stop' },
    ]);

    expect(bridge.start).toHaveBeenCalledWith(
      expect.objectContaining({
        providerId: 'openai',
        modelId: 'gpt-5.6-sol',
        request: {
          schemaVersion: AI_RUNTIME_SCHEMA_VERSION,
          conversationId: 'conversation-1',
          messages: [{ id: 'message-1', role: 'user', content: 'Hello' }],
        },
      }),
    );
    expect(bridge.cancel).not.toHaveBeenCalled();
  });

  it('cancels the main-process request when the renderer aborts', async () => {
    let listener: Parameters<ArcAiRuntimeBridge['onEvent']>[0] | null = null;
    const bridge: ArcAiRuntimeBridge = {
      start: vi.fn(async () => undefined),
      cancel: vi.fn(async () => true),
      onEvent: (callback) => {
        listener = callback;
        return () => {
          listener = null;
        };
      },
    };
    const controller = new AbortController();
    const iterator = streamOpenAiRuntime(
      'gpt-5.6-sol',
      { conversationId: 'conversation-1', messages: [], signal: controller.signal },
      bridge,
    )[Symbol.asyncIterator]();

    const pending = iterator.next();
    await Promise.resolve();
    controller.abort();

    await expect(pending).resolves.toEqual({ done: true, value: undefined });
    expect(bridge.cancel).toHaveBeenCalledTimes(1);
    expect(listener).toBeNull();
  });
});

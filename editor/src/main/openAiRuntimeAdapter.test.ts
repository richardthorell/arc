import { describe, expect, it, vi } from 'vitest';

import type { AiRuntimeRequest, AiRuntimeStreamEvent } from '../common/aiRuntimeTypes';
import { OpenAiRuntimeAdapter } from './openAiRuntimeAdapter';

const request = (signal?: AbortSignal): AiRuntimeRequest => ({
  conversationId: 'conversation-1',
  messages: [{ id: 'message-1', role: 'user', content: 'Hello ARC' }],
  signal,
});

const sseResponse = (...events: unknown[]) =>
  new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(''), {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });

const collect = async (adapter: OpenAiRuntimeAdapter, runtimeRequest = request()) => {
  const events: AiRuntimeStreamEvent[] = [];
  for await (const event of adapter.stream(runtimeRequest, { modelId: 'gpt-5.6-sol', reasoningEffort: 'medium' }))
    events.push(event);
  return events;
};

describe('OpenAiRuntimeAdapter', () => {
  it('normalizes streamed text, usage, and completion', async () => {
    const transport = vi.fn(async () =>
      sseResponse(
        { type: 'response.output_text.delta', delta: 'Hello ' },
        { type: 'response.output_text.delta', delta: 'world' },
        {
          type: 'response.completed',
          response: {
            usage: {
              input_tokens: 8,
              output_tokens: 3,
              total_tokens: 11,
              input_tokens_details: { cached_tokens: 2 },
              output_tokens_details: { reasoning_tokens: 1 },
            },
          },
        },
      ),
    );
    const adapter = new OpenAiRuntimeAdapter(transport);

    await expect(collect(adapter)).resolves.toEqual([
      { type: 'delta', text: 'Hello ' },
      { type: 'delta', text: 'world' },
      {
        type: 'usage',
        usage: { inputTokens: 8, outputTokens: 3, cachedInputTokens: 2, reasoningTokens: 1, totalTokens: 11 },
      },
      { type: 'done', finishReason: 'stop' },
    ]);
    expect(transport).toHaveBeenCalledWith(
      expect.objectContaining({
        model: 'gpt-5.6-sol',
        stream: true,
        input: [{ role: 'user', content: 'Hello ARC' }],
        reasoning: { effort: 'medium' },
      }),
      undefined,
    );
  });

  it('normalizes function-call streaming into ARC tool events', async () => {
    const adapter = new OpenAiRuntimeAdapter(async () =>
      sseResponse(
        {
          type: 'response.output_item.added',
          item: { type: 'function_call', call_id: 'call-1', name: 'scene.inspect' },
        },
        { type: 'response.function_call_arguments.delta', call_id: 'call-1', delta: '{"guid":' },
        {
          type: 'response.output_item.done',
          item: {
            type: 'function_call',
            call_id: 'call-1',
            name: 'scene.inspect',
            arguments: '{"guid":"entity-1"}',
          },
        },
        { type: 'response.completed', response: { usage: { input_tokens: 4, output_tokens: 2, total_tokens: 6 } } },
      ),
    );

    expect(await collect(adapter)).toEqual([
      { type: 'tool-call-start', callId: 'call-1', name: 'scene.inspect' },
      { type: 'tool-call-arguments-delta', callId: 'call-1', delta: '{"guid":' },
      {
        type: 'tool-call',
        call: { id: 'call-1', name: 'scene.inspect', arguments: { guid: 'entity-1' } },
      },
      { type: 'usage', usage: { inputTokens: 4, outputTokens: 2, totalTokens: 6 } },
      { type: 'done', finishReason: 'tool_calls' },
    ]);
  });

  it('normalizes provider status failures', async () => {
    const adapter = new OpenAiRuntimeAdapter(async () => new Response('', { status: 429 }));

    expect(await collect(adapter)).toEqual([
      {
        type: 'error',
        code: 'rate_limit',
        message: 'OpenAI request failed (HTTP 429)',
        retryable: true,
      },
    ]);
  });

  it('propagates cancellation to the transport and prevents later stream events', async () => {
    const controller = new AbortController();
    const transport = vi.fn(async (_body: Record<string, unknown>, signal?: AbortSignal) => {
      expect(signal).toBe(controller.signal);
      controller.abort();
      throw new DOMException('aborted', 'AbortError');
    });
    const adapter = new OpenAiRuntimeAdapter(transport);

    expect(await collect(adapter, request(controller.signal))).toEqual([
      { type: 'error', code: 'cancelled', message: 'OpenAI response was cancelled', retryable: false },
    ]);
  });
});

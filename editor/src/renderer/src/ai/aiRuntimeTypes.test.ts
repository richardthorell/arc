import { describe, expect, it } from 'vitest';

import {
  AI_RUNTIME_SCHEMA_VERSION,
  isTerminalAiRuntimeEvent,
  serializeAiRuntimeRequest,
  textContent,
  textFromRuntimeMessage,
  type AiRuntimeRequest,
  type AiRuntimeStreamEvent,
  type AiToolCall,
  type AiToolResult,
} from '../../../common/aiRuntimeTypes';
import { createAiMessage, createAiModelRequest } from './aiChat';

describe('AI runtime contracts', () => {
  it('adapts chat messages into provider-neutral runtime messages', () => {
    const message = createAiMessage('user', 'Inspect the selected cabin');
    const request = createAiModelRequest('conversation-1', [message]);

    expect(request).toMatchObject({
      conversationId: 'conversation-1',
      messages: [
        {
          id: message.id,
          role: 'user',
          content: [{ type: 'text', text: 'Inspect the selected cabin' }],
          createdAt: message.createdAt,
        },
      ],
    });
  });

  it('keeps cancellation local while serializing a versioned request payload', () => {
    const controller = new AbortController();
    const request: AiRuntimeRequest = {
      conversationId: 'conversation-1',
      messages: [{ id: 'user-1', role: 'user', content: 'Hello' }],
      signal: controller.signal,
    };

    controller.abort();
    expect(request.signal?.aborted).toBe(true);

    const serialized = serializeAiRuntimeRequest(request);
    expect(serialized.schemaVersion).toBe(AI_RUNTIME_SCHEMA_VERSION);
    expect(serialized).not.toHaveProperty('signal');
    expect(serialized.messages[0]).toMatchObject({ id: 'user-1', role: 'user', content: 'Hello' });
  });

  it('normalizes text, tool calls, tool results, usage, errors, and completion into one event union', () => {
    const call: AiToolCall = {
      id: 'call-1',
      name: 'scene.getEntity',
      arguments: { guid: 'entity-guid' },
    };
    const result: AiToolResult = {
      toolCallId: call.id,
      name: call.name,
      content: [textContent('Cabin')],
    };
    const events: AiRuntimeStreamEvent[] = [
      { type: 'delta', text: 'Inspecting…' },
      { type: 'tool-call-start', callId: call.id, name: call.name },
      { type: 'tool-call-arguments-delta', callId: call.id, delta: '{"guid":' },
      { type: 'tool-call', call },
      { type: 'tool-result', result },
      { type: 'usage', usage: { inputTokens: 12, outputTokens: 4, totalTokens: 16 } },
      { type: 'error', code: 'rate_limit', message: 'Try again later', retryable: true, retryAfterMs: 1000 },
      { type: 'done', finishReason: 'stop' },
    ];

    expect(events.map((event) => event.type)).toEqual([
      'delta',
      'tool-call-start',
      'tool-call-arguments-delta',
      'tool-call',
      'tool-result',
      'usage',
      'error',
      'done',
    ]);
    expect(isTerminalAiRuntimeEvent(events[0])).toBe(false);
    expect(isTerminalAiRuntimeEvent(events[6])).toBe(true);
    expect(isTerminalAiRuntimeEvent(events[7])).toBe(true);
  });

  it('extracts text from both shorthand and block content', () => {
    expect(textFromRuntimeMessage({ id: 'a', role: 'assistant', content: 'Hello' })).toBe('Hello');
    expect(
      textFromRuntimeMessage({
        id: 'b',
        role: 'assistant',
        content: [textContent('Hello '), { type: 'image', mimeType: 'image/png', uri: 'arc://capture/1' }, textContent('world')],
      }),
    ).toBe('Hello world');
  });
});

import { describe, expect, it, vi } from 'vitest';

import type { AiRuntimeRequest, AiRuntimeStreamEvent } from '../common/aiRuntimeTypes';
import { OpenAiRuntimeAdapter } from './openAiRuntimeAdapter';

const tool = {
  name: 'scene.getEntity',
  description: 'Inspect one scene entity.',
  inputSchema: {
    type: 'object' as const,
    properties: { guid: { type: 'string' } },
    required: ['guid'],
    additionalProperties: false,
  },
};

const sseResponse = (...events: unknown[]) =>
  new Response(events.map((event) => `data: ${JSON.stringify(event)}\n\n`).join(''), {
    status: 200,
    headers: { 'content-type': 'text/event-stream' },
  });

const collect = async (adapter: OpenAiRuntimeAdapter, request: AiRuntimeRequest) => {
  const events: AiRuntimeStreamEvent[] = [];
  for await (const event of adapter.stream(request, { modelId: 'gpt-5.6-sol' })) events.push(event);
  return events;
};

describe('OpenAiRuntimeAdapter tool projection', () => {
  it('uses provider-safe function names on the wire and restores stable ARC operation names in events', async () => {
    const transport = vi.fn(async () =>
      sseResponse(
        {
          type: 'response.output_item.added',
          item: { type: 'function_call', call_id: 'call-1', name: 'arc_scene_get_entity' },
        },
        {
          type: 'response.output_item.done',
          item: {
            type: 'function_call',
            call_id: 'call-1',
            name: 'arc_scene_get_entity',
            arguments: '{"guid":"entity-1"}',
          },
        },
        { type: 'response.completed', response: {} },
      ),
    );
    const adapter = new OpenAiRuntimeAdapter(transport);
    const request: AiRuntimeRequest = {
      conversationId: 'conversation-1',
      messages: [{ id: 'message-1', role: 'user', content: 'Inspect the selected entity' }],
      tools: [tool],
    };

    expect(await collect(adapter, request)).toEqual([
      { type: 'tool-call-start', callId: 'call-1', name: 'scene.getEntity' },
      {
        type: 'tool-call',
        call: { id: 'call-1', name: 'scene.getEntity', arguments: { guid: 'entity-1' } },
      },
      { type: 'done', finishReason: 'tool_calls' },
    ]);
    expect(transport).toHaveBeenCalledWith(
      expect.objectContaining({
        tools: [
          {
            type: 'function',
            name: 'arc_scene_get_entity',
            description: tool.description,
            parameters: tool.inputSchema,
          },
        ],
      }),
      undefined,
    );
  });

  it('projects stable ARC names in assistant tool-call history before sending it back to OpenAI', async () => {
    const transport = vi.fn(async () => sseResponse({ type: 'response.completed', response: {} }));
    const adapter = new OpenAiRuntimeAdapter(transport);
    const request: AiRuntimeRequest = {
      conversationId: 'conversation-1',
      messages: [
        {
          id: 'assistant-1',
          role: 'assistant',
          content: '',
          toolCalls: [{ id: 'call-1', name: 'scene.getEntity', arguments: { guid: 'entity-1' } }],
        },
      ],
      tools: [tool],
    };

    await collect(adapter, request);
    expect(transport).toHaveBeenCalledWith(
      expect.objectContaining({
        input: [
          {
            type: 'function_call',
            call_id: 'call-1',
            name: 'arc_scene_get_entity',
            arguments: '{"guid":"entity-1"}',
          },
        ],
      }),
      undefined,
    );
  });
});

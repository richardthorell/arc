import { describe, expect, it } from 'vitest';

import type { AiConversationMessage } from '../../../common/aiConversationTypes';
import {
  AI_CONVERSATION_TOOL_RESULT_MAX_CHARACTERS,
  finishPendingConversationTools,
  recordConversationToolCall,
  recordConversationToolResult,
  runtimeMessagesForConversationMessage,
} from './aiConversationToolTranscript';

describe('AI conversation tool transcript', () => {
  it('records calls and results with stable operation metadata', () => {
    const startedAt = '2026-10-04T04:00:00.000Z';
    const completedAt = '2026-10-04T04:00:01.000Z';
    const pending = recordConversationToolCall(
      undefined,
      { id: 'call-1', name: 'scene.getEntity', arguments: { guid: 'floor-guid' } },
      2,
      startedAt,
    );
    const complete = recordConversationToolResult(
      pending,
      {
        toolCallId: 'call-1',
        name: 'scene.getEntity',
        operation: 'scene.getEntity',
        content: '{"name":"Floor"}',
        truncated: false,
        originalBytes: 16,
      },
      2,
      completedAt,
    );

    expect(complete).toEqual([
      expect.objectContaining({
        toolCallId: 'call-1',
        name: 'scene.getEntity',
        operation: 'scene.getEntity',
        state: 'complete',
        step: 2,
        arguments: { guid: 'floor-guid' },
        resultContent: '{"name":"Floor"}',
        resultTruncated: false,
        originalBytes: 16,
        startedAt,
        completedAt,
      }),
    ]);
  });

  it('bounds persisted tool results without losing original result metadata', () => {
    const oversized = 'x'.repeat(AI_CONVERSATION_TOOL_RESULT_MAX_CHARACTERS + 500);
    const complete = recordConversationToolResult(
      recordConversationToolCall(
        undefined,
        { id: 'call-1', name: 'viewport.observe', arguments: {} },
        0,
        '2026-10-04T04:00:00.000Z',
      ),
      {
        toolCallId: 'call-1',
        name: 'viewport.observe',
        operation: 'viewport.observe',
        content: oversized,
        truncated: false,
        originalBytes: oversized.length,
      },
      0,
      '2026-10-04T04:00:01.000Z',
    );

    expect(complete[0]?.resultContent?.length).toBeLessThanOrEqual(AI_CONVERSATION_TOOL_RESULT_MAX_CHARACTERS);
    expect(complete[0]).toMatchObject({ resultTruncated: true, originalBytes: oversized.length });
  });

  it('reconstructs ordered assistant tool calls and tool results before the final assistant message', () => {
    const message: AiConversationMessage = {
      id: 'assistant-1',
      role: 'assistant',
      content: 'The floor is selected.',
      createdAt: '2026-10-04T04:00:02.000Z',
      state: 'complete',
      toolReferences: [
        {
          toolCallId: 'call-1',
          name: 'scene.findEntities',
          operation: 'scene.findEntities',
          state: 'complete',
          step: 0,
          arguments: { search: 'Floor' },
          resultContent: '{"entities":[{"guid":"floor-guid"}]}',
          startedAt: '2026-10-04T04:00:00.000Z',
          completedAt: '2026-10-04T04:00:00.500Z',
        },
        {
          toolCallId: 'call-2',
          name: 'viewport.pick',
          operation: 'viewport.pick',
          state: 'complete',
          step: 1,
          arguments: { x: 100, y: 120 },
          resultContent: '{"selection":{"guid":"floor-guid"}}',
          startedAt: '2026-10-04T04:00:01.000Z',
          completedAt: '2026-10-04T04:00:01.500Z',
        },
      ],
    };

    const runtime = runtimeMessagesForConversationMessage(message);

    expect(runtime.map((entry) => entry.role)).toEqual(['assistant', 'tool', 'assistant', 'tool', 'assistant']);
    expect(runtime[0]?.toolCalls?.[0]).toMatchObject({
      id: 'call-1',
      name: 'scene.findEntities',
      arguments: { search: 'Floor' },
    });
    expect(runtime[1]?.toolResult).toMatchObject({
      toolCallId: 'call-1',
      operation: 'scene.findEntities',
    });
    expect(runtime[2]?.toolCalls?.[0]).toMatchObject({ id: 'call-2', name: 'viewport.pick' });
    expect(runtime.at(-1)?.content).toEqual([{ type: 'text', text: 'The floor is selected.' }]);
  });

  it('marks pending operations as cancelled without rewriting completed operations', () => {
    const references = finishPendingConversationTools(
      [
        { toolCallId: 'call-1', name: 'scene.findEntities', state: 'complete' },
        { toolCallId: 'call-2', name: 'viewport.observe', state: 'pending' },
      ],
      'cancelled',
      'Cancelled by user',
      '2026-10-04T04:00:02.000Z',
    );

    expect(references).toEqual([
      { toolCallId: 'call-1', name: 'scene.findEntities', state: 'complete' },
      expect.objectContaining({
        toolCallId: 'call-2',
        state: 'cancelled',
        summary: 'Cancelled by user',
        completedAt: '2026-10-04T04:00:02.000Z',
      }),
    ]);
  });
});

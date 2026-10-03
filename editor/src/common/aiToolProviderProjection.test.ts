import { describe, expect, it } from 'vitest';

import type { AiToolDefinition } from './aiRuntimeTypes';
import {
  projectAiToolsForAnthropic,
  projectAiToolsForOpenAi,
  providerToolName,
  stableToolNameFromProvider,
} from './aiToolProviderProjection';

const tools: AiToolDefinition[] = [
  {
    name: 'scene.getEntity',
    description: 'Inspect one scene entity.',
    inputSchema: {
      type: 'object',
      properties: { guid: { type: 'string' } },
      required: ['guid'],
      additionalProperties: false,
    },
  },
];

describe('AI tool provider projection', () => {
  it('keeps ARC operation identity stable while producing provider-safe function names', () => {
    expect(providerToolName('scene.getEntity')).toBe('arc_scene_get_entity');
    expect(projectAiToolsForOpenAi(tools)).toEqual([
      {
        type: 'function',
        name: 'arc_scene_get_entity',
        description: 'Inspect one scene entity.',
        parameters: tools[0]!.inputSchema,
      },
    ]);
    expect(projectAiToolsForAnthropic(tools)).toEqual([
      {
        name: 'arc_scene_get_entity',
        description: 'Inspect one scene entity.',
        input_schema: tools[0]!.inputSchema,
      },
    ]);
  });

  it('maps provider function names back to the stable ARC operation name', () => {
    expect(stableToolNameFromProvider('arc_scene_get_entity', tools)).toBe('scene.getEntity');
    expect(stableToolNameFromProvider('unregistered_tool', tools)).toBe('unregistered_tool');
  });
});

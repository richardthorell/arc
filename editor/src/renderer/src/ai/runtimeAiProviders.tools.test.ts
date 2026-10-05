import { describe, expect, it, vi } from 'vitest';

import type { AiInstructionSourceSnapshot } from '../../../common/aiInstructionTypes';
import type { AiToolDefinition } from '../../../common/aiRuntimeTypes';
import type { EditorSettingsSnapshot } from '../../../common/editorWorkflowTypes';
import { runtimeAiProvidersFromSettings } from './runtimeAiProviders';

const settings: EditorSettingsSnapshot = {
  revision: 1,
  values: { 'ai.anthropic.model': 'claude-test' },
  sources: {},
  restartRequired: [],
  schema: [
    {
      key: 'ai.anthropic.model',
      section: 'Anthropic',
      label: 'Model',
      description: 'Anthropic model',
      type: 'enum',
      defaultValue: 'claude-test',
      options: ['claude-test'],
      scopes: ['user'],
    },
  ],
  aiProviders: {
    secureStorageAvailable: true,
    providers: [
      {
        id: 'anthropic',
        label: 'Anthropic',
        connected: true,
        connectionStatus: 'connected',
      },
    ],
  },
};

const sources: AiInstructionSourceSnapshot = {
  revision: 1,
  projectGuid: 'project-1',
  skills: [],
  diagnostics: [],
};

const tools: AiToolDefinition[] = [
  {
    name: 'scene.getEntity',
    description: 'Inspect an entity.',
    inputSchema: { type: 'object', properties: { guid: { type: 'string' } }, required: ['guid'] },
  },
  {
    name: 'edit.begin',
    description: 'Begin an approved edit.',
    inputSchema: { type: 'object', properties: {} },
  },
];

describe('runtime AI provider harness tools', () => {
  it('loads registry tools before resolving skills and capabilities', async () => {
    const onInstructionResolution = vi.fn();
    const provider = runtimeAiProvidersFromSettings(settings, {
      instructionSources: async () => sources,
      agentTools: async () => tools,
      onInstructionResolution,
    })[0]!;

    const iterator = provider
      .stream({ conversationId: 'conversation-1', messages: [{ id: 'user-1', role: 'user', content: 'Inspect it' }] })
      [Symbol.asyncIterator]();
    await iterator.next();

    expect(onInstructionResolution).toHaveBeenCalledWith(
      expect.objectContaining({
        availableTools: ['agent.updatePlan', 'edit.begin', 'scene.getEntity'],
        availableCapabilities: expect.arrayContaining(['scene.read', 'scene.mutate']),
      }),
    );
  });

  it('preserves existing request tools while letting the harness registry own matching operation definitions', async () => {
    const onInstructionResolution = vi.fn();
    const provider = runtimeAiProvidersFromSettings(settings, {
      instructionSources: async () => sources,
      agentTools: async () => tools,
      onInstructionResolution,
    })[0]!;

    const iterator = provider
      .stream({
        conversationId: 'conversation-1',
        messages: [{ id: 'user-1', role: 'user', content: 'Inspect it' }],
        tools: [
          {
            name: 'custom.lookup',
            description: 'Custom request tool.',
            inputSchema: { type: 'object' },
          },
          {
            name: 'scene.getEntity',
            description: 'Divergent duplicate that must not hide the registry operation.',
            inputSchema: { type: 'object' },
          },
        ],
      })
      [Symbol.asyncIterator]();
    await iterator.next();

    expect(onInstructionResolution).toHaveBeenCalledWith(
      expect.objectContaining({ availableTools: ['agent.updatePlan', 'custom.lookup', 'edit.begin', 'scene.getEntity'] }),
    );
  });
});

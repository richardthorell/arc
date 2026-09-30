import { describe, expect, it } from 'vitest';

import type { EditorSettingsSnapshot } from '../../../common/editorWorkflowTypes';
import { runtimeAiProvidersFromSettings } from './runtimeAiProviders';

const snapshot = (connectionStatus: 'disconnected' | 'cold' | 'connected' | 'invalid'): EditorSettingsSnapshot => ({
  revision: 1,
  values: { 'ai.openai.model': 'gpt-5.6-sol' },
  sources: {},
  restartRequired: [],
  schema: [
    {
      key: 'ai.openai.model',
      section: 'OpenAI',
      label: 'Model',
      description: 'OpenAI model',
      type: 'enum',
      defaultValue: 'gpt-5.6-sol',
      options: ['gpt-5.6-sol', 'gpt-5.6-luna'],
      optionLabels: { 'gpt-5.6-sol': 'GPT-5.6 Sol', 'gpt-5.6-luna': 'GPT-5.6 Luna' },
      scopes: ['user'],
    },
  ],
  aiProviders: {
    secureStorageAvailable: true,
    providers: [
      {
        id: 'openai',
        label: 'OpenAI',
        connected: connectionStatus !== 'disconnected',
        connectionStatus,
      },
    ],
  },
});

describe('runtimeAiProvidersFromSettings', () => {
  it('exposes configured models for a connected account with the selected default first', () => {
    const providers = runtimeAiProvidersFromSettings(snapshot('connected'));

    expect(providers.map(({ id, label }) => ({ id, label }))).toEqual([
      { id: 'openai:gpt-5.6-sol', label: 'GPT-5.6 Sol' },
      { id: 'openai:gpt-5.6-luna', label: 'GPT-5.6 Luna' },
    ]);
  });

  it('accepts cold stored credentials but excludes disconnected or invalid accounts', () => {
    expect(runtimeAiProvidersFromSettings(snapshot('cold'))).toHaveLength(2);
    expect(runtimeAiProvidersFromSettings(snapshot('disconnected'))).toEqual([]);
    expect(runtimeAiProvidersFromSettings(snapshot('invalid'))).toEqual([]);
  });
});

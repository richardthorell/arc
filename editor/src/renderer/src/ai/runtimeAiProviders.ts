import type { AiProviderId } from '../../../common/aiProviderTypes';
import type { EditorSettingDescriptor, EditorSettingsSnapshot } from '../../../common/editorWorkflowTypes';
import type { AiModelProvider } from './aiChat';

const modelSettingKeys: Record<AiProviderId, string> = {
  openai: 'ai.openai.model',
  anthropic: 'ai.anthropic.model',
};

const modelOptions = (snapshot: EditorSettingsSnapshot, providerId: AiProviderId): string[] => {
  const key = modelSettingKeys[providerId];
  const descriptor = snapshot.schema.find((candidate) => candidate.key === key);
  const selected = snapshot.values[key];
  const selectedModel = typeof selected === 'string' && selected ? selected : String(descriptor?.defaultValue ?? '');
  const options = (descriptor?.options ?? []).filter(Boolean);
  return [selectedModel, ...options.filter((option) => option !== selectedModel)].filter(Boolean);
};

const modelLabel = (descriptor: EditorSettingDescriptor | undefined, modelId: string) =>
  descriptor?.optionLabels?.[modelId] ?? modelId;

export const runtimeAiProvidersFromSettings = (
  snapshot: EditorSettingsSnapshot | null | undefined,
): AiModelProvider[] => {
  if (!snapshot?.aiProviders) return [];

  return snapshot.aiProviders.providers.flatMap((account) => {
    if (!account.connected || account.connectionStatus === 'invalid') return [];
    const key = modelSettingKeys[account.id];
    const descriptor = snapshot.schema.find((candidate) => candidate.key === key);
    return modelOptions(snapshot, account.id).map((modelId) => ({
      id: `${account.id}:${modelId}`,
      label: modelLabel(descriptor, modelId),
      configured: true,
      async *stream() {
        yield {
          type: 'error' as const,
          message: `${account.label} is connected, but provider execution is not wired to AI Chat yet.`,
        };
      },
    }));
  });
};

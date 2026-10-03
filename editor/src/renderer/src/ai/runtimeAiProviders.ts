import type {
  AiInstructionResolutionDiagnostics,
  AiInstructionSourceSnapshot,
} from '../../../common/aiInstructionTypes';
import type { AiProviderId } from '../../../common/aiProviderTypes';
import type { AiModelCapabilities, AiRuntimeRequest, AiRuntimeStreamEvent } from '../../../common/aiRuntimeTypes';
import { assertAiRuntimeRequestSafeForProvider } from '../../../common/aiSecurityPolicy';
import type { EditorSettingDescriptor, EditorSettingsSnapshot } from '../../../common/editorWorkflowTypes';
import type { AiModelProvider } from './aiChat';
import { resolveAiRuntimeInstructions } from './aiInstructionResolver';
import { streamOpenAiRuntime } from './openAiRuntimeProvider';

const modelSettingKeys: Record<AiProviderId, string> = {
  openai: 'ai.openai.model',
  anthropic: 'ai.anthropic.model',
};

const openAiCapabilities: AiModelCapabilities = {
  streaming: true,
  tools: true,
  inputModalities: ['text', 'image'],
};

const placeholderCapabilities: AiModelCapabilities = {
  streaming: true,
  tools: true,
  inputModalities: ['text'],
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

const unavailableInstructionSources = (message: string): AiInstructionSourceSnapshot => ({
  revision: 0,
  projectGuid: null,
  skills: [],
  diagnostics: [{ source: 'builtin-skills', message }],
});

const defaultInstructionSources = async (): Promise<AiInstructionSourceSnapshot> => {
  const bridge = typeof window === 'undefined' ? undefined : window.arcAiRuntime;
  if (!bridge?.instructionSources)
    return unavailableInstructionSources('AI instruction source bridge is unavailable; using ARC base instructions only');
  try {
    return await bridge.instructionSources();
  } catch (error) {
    return unavailableInstructionSources(
      `AI instruction sources could not be loaded: ${error instanceof Error ? error.message : String(error)}`,
    );
  }
};

export type RuntimeAiProviderOptions = {
  instructionSources?: () => Promise<AiInstructionSourceSnapshot>;
  onInstructionResolution?: (diagnostics: AiInstructionResolutionDiagnostics) => void;
};

const withResolvedInstructions = (
  request: AiRuntimeRequest,
  execute: (prepared: AiRuntimeRequest) => AsyncIterable<AiRuntimeStreamEvent>,
  options: RuntimeAiProviderOptions,
): AsyncIterable<AiRuntimeStreamEvent> =>
  (async function* () {
    const sources = await (options.instructionSources ?? defaultInstructionSources)();
    if (request.signal?.aborted) return;
    const resolution = resolveAiRuntimeInstructions(request, sources);
    options.onInstructionResolution?.(resolution.diagnostics);
    yield* execute(resolution.request);
  })();

export const runtimeAiProvidersFromSettings = (
  snapshot: EditorSettingsSnapshot | null | undefined,
  options: RuntimeAiProviderOptions = {},
): AiModelProvider[] => {
  if (!snapshot?.aiProviders) return [];

  return snapshot.aiProviders.providers.flatMap((account) => {
    if (!account.connected || account.connectionStatus === 'invalid') return [];
    const key = modelSettingKeys[account.id];
    const descriptor = snapshot.schema.find((candidate) => candidate.key === key);
    return modelOptions(snapshot, account.id).map((modelId) => ({
      id: `${account.id}:${modelId}`,
      providerId: account.id,
      modelId,
      label: modelLabel(descriptor, modelId),
      capabilities: account.id === 'openai' ? openAiCapabilities : placeholderCapabilities,
      configured: true,
      stream(request) {
        if (account.id === 'openai')
          return withResolvedInstructions(request, (prepared) => streamOpenAiRuntime(modelId, prepared), options);
        return withResolvedInstructions(
          request,
          async function* (prepared) {
            assertAiRuntimeRequestSafeForProvider(prepared);
            yield {
              type: 'error' as const,
              code: 'provider' as const,
              retryable: false,
              message: `${account.label} is connected, but provider execution is not wired to AI Chat yet.`,
            };
          },
          options,
        );
      },
    }));
  });
};

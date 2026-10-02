import type {
  AiConversationMessage,
  AiConversationMessageState,
  AiConversationRole,
  AiStoredConversation,
} from '../../../common/aiConversationTypes';
import {
  textContent,
  type AiModelCapabilities,
  type AiModelDescriptor,
  type AiRuntimeMessage,
  type AiRuntimeRequest,
  type AiRuntimeStreamEvent,
} from '../../../common/aiRuntimeTypes';

export type AiChatRole = AiConversationRole;
export type AiChatMessageState = AiConversationMessageState;
export type AiChatMessage = AiConversationMessage;
export type AiConversation = AiStoredConversation;

export type AiModelRequest = AiRuntimeRequest;
export type AiModelStreamEvent = AiRuntimeStreamEvent;

export interface AiModelProvider extends AiModelDescriptor {
  readonly configured: boolean;
  stream(request: AiModelRequest): AsyncIterable<AiModelStreamEvent>;
}

export const textOnlyAiModelCapabilities = {
  streaming: true,
  tools: false,
  inputModalities: ['text'],
} as const satisfies AiModelCapabilities;

// Kept only so older renderer data can be identified and migrated by
// aiConversationStore. New production persistence is project-scoped.
export const aiConversationStorageKey = 'arc.ai.conversations.v1';

const makeId = () =>
  typeof crypto !== 'undefined' && 'randomUUID' in crypto
    ? crypto.randomUUID()
    : `ai-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`;

const now = () => new Date().toISOString();

export const createAiConversation = (): AiConversation => {
  const timestamp = now();
  return {
    id: makeId(),
    title: 'New Chat',
    createdAt: timestamp,
    updatedAt: timestamp,
    messages: [],
  };
};

export const createAiMessage = (
  role: AiChatRole,
  content: string,
  state: AiChatMessageState = 'complete',
): AiChatMessage => ({
  id: makeId(),
  role,
  content,
  createdAt: now(),
  state,
});

export const toAiRuntimeMessages = (messages: readonly AiChatMessage[]): AiRuntimeMessage[] =>
  messages.map((message) => ({
    id: message.id,
    role: message.role,
    content: [textContent(message.content)],
    createdAt: message.createdAt,
  }));

export const createAiModelRequest = (
  conversationId: string,
  messages: readonly AiChatMessage[],
  signal?: AbortSignal,
): AiModelRequest => ({
  conversationId,
  messages: toAiRuntimeMessages(messages),
  signal,
});

export const conversationTitleFromPrompt = (prompt: string): string => {
  const normalized = prompt.replace(/\s+/g, ' ').trim();
  if (!normalized) return 'New Chat';
  return normalized.length > 42 ? `${normalized.slice(0, 39).trimEnd()}…` : normalized;
};

export const unavailableAiModelProvider: AiModelProvider = {
  id: 'unconfigured',
  providerId: 'none',
  modelId: 'unconfigured',
  label: 'No provider',
  capabilities: textOnlyAiModelCapabilities,
  configured: false,
  async *stream() {
    const text =
      'No AI model provider is configured yet. Stage 1 adds the conversation and streaming foundation; model configuration and editor-aware tools are layered on next.';
    for (const chunk of text.match(/.{1,24}(?:\s|$)/g) ?? [text]) {
      yield { type: 'delta' as const, text: chunk };
      await Promise.resolve();
    }
    yield { type: 'done' as const, finishReason: 'stop' as const };
  },
};

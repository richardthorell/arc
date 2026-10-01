import {
  textContent,
  type AiModelCapabilities,
  type AiModelDescriptor,
  type AiRuntimeMessage,
  type AiRuntimeRequest,
  type AiRuntimeStreamEvent,
} from '../../../common/aiRuntimeTypes';

export type AiChatRole = 'user' | 'assistant' | 'system';

export type AiChatMessageState = 'complete' | 'streaming' | 'error';

export type AiChatMessage = {
  id: string;
  role: AiChatRole;
  content: string;
  createdAt: string;
  state: AiChatMessageState;
  modelId?: string;
  modelLabel?: string;
};

export type AiConversation = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: AiChatMessage[];
  modelId?: string;
  modelLabel?: string;
};

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

const isMessage = (value: unknown): value is AiChatMessage => {
  if (!value || typeof value !== 'object') return false;
  const message = value as Partial<AiChatMessage>;
  return (
    typeof message.id === 'string' &&
    (message.role === 'user' || message.role === 'assistant' || message.role === 'system') &&
    typeof message.content === 'string' &&
    typeof message.createdAt === 'string' &&
    (message.state === 'complete' || message.state === 'streaming' || message.state === 'error') &&
    (message.modelId === undefined || typeof message.modelId === 'string') &&
    (message.modelLabel === undefined || typeof message.modelLabel === 'string')
  );
};

const isConversation = (value: unknown): value is AiConversation => {
  if (!value || typeof value !== 'object') return false;
  const conversation = value as Partial<AiConversation>;
  return (
    typeof conversation.id === 'string' &&
    typeof conversation.title === 'string' &&
    typeof conversation.createdAt === 'string' &&
    typeof conversation.updatedAt === 'string' &&
    (conversation.modelId === undefined || typeof conversation.modelId === 'string') &&
    (conversation.modelLabel === undefined || typeof conversation.modelLabel === 'string') &&
    Array.isArray(conversation.messages) &&
    conversation.messages.every(isMessage)
  );
};

export const loadAiConversations = (storage: Pick<Storage, 'getItem'> = localStorage): AiConversation[] => {
  try {
    const raw = storage.getItem(aiConversationStorageKey);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as unknown;
    if (!Array.isArray(parsed)) return [];
    return parsed
      .filter(isConversation)
      .filter((conversation) => conversation.messages.length > 0)
      .map((conversation) => ({
        ...conversation,
        messages: conversation.messages.map((message) =>
          message.state === 'streaming' ? { ...message, state: 'error' as const } : message,
        ),
      }));
  } catch {
    return [];
  }
};

export const saveAiConversations = (
  conversations: readonly AiConversation[],
  storage: Pick<Storage, 'setItem'> = localStorage,
): void => {
  storage.setItem(
    aiConversationStorageKey,
    JSON.stringify(conversations.filter((conversation) => conversation.messages.length > 0)),
  );
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

import type { AiImageContentPart, AiJsonObject } from './aiRuntimeTypes';

export const AI_CONVERSATION_STORE_VERSION = 1 as const;
export type AiConversationStoreVersion = typeof AI_CONVERSATION_STORE_VERSION;

export type AiConversationRole = 'user' | 'assistant' | 'system';
export type AiConversationMessageState = 'complete' | 'streaming' | 'error';

export type AiConversationContextReference = {
  id: string;
  kind: string;
  label?: string;
  stableId?: string;
  metadata?: AiJsonObject;
  attachment?: AiImageContentPart;
};

export type AiConversationToolReference = {
  toolCallId: string;
  name: string;
  state: 'pending' | 'complete' | 'error' | 'cancelled';
  summary?: string;
};

export type AiConversationMessage = {
  id: string;
  role: AiConversationRole;
  content: string;
  createdAt: string;
  state: AiConversationMessageState;
  modelId?: string;
  modelLabel?: string;
  contextReferences?: AiConversationContextReference[];
  toolReferences?: AiConversationToolReference[];
};

export type AiConversationSummary = {
  text: string;
  throughMessageId?: string;
  updatedAt: string;
};

export type AiStoredConversation = {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
  messages: AiConversationMessage[];
  modelId?: string;
  modelLabel?: string;
  summary?: AiConversationSummary;
  pinnedContext?: AiConversationContextReference[];
};

export type AiConversationUiState = {
  activeConversationId?: string;
  selectedModelId?: string;
};

export type AiConversationStoreSnapshot = {
  schemaVersion: AiConversationStoreVersion;
  projectGuid: string;
  updatedAt: string;
  conversations: AiStoredConversation[];
  uiState: AiConversationUiState;
};
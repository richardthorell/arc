export const AI_RUNTIME_SCHEMA_VERSION = 1 as const;

export type AiRuntimeSchemaVersion = typeof AI_RUNTIME_SCHEMA_VERSION;

export type AiJsonPrimitive = string | number | boolean | null;
export type AiJsonObject = { readonly [key: string]: AiJsonValue };
export type AiJsonValue = AiJsonPrimitive | AiJsonObject | readonly AiJsonValue[];

export type AiRuntimeRole = 'system' | 'user' | 'assistant' | 'tool';

export type AiTextContentPart = {
  type: 'text';
  text: string;
};

export type AiImageContentPart = {
  type: 'image';
  mimeType: string;
  uri: string;
  alt?: string;
};

export type AiRuntimeContentPart = AiTextContentPart | AiImageContentPart;
export type AiRuntimeMessageContent = string | readonly AiRuntimeContentPart[];

export type AiToolDefinition = {
  name: string;
  operationId?: string;
  description: string;
  inputSchema: AiJsonObject;
};

export type AiToolCall = {
  id: string;
  name: string;
  arguments: AiJsonObject;
};

export type AiToolResult = {
  toolCallId: string;
  name: string;
  content: AiRuntimeMessageContent;
  isError?: boolean;
};

export type AiRuntimeMessage = {
  id: string;
  role: AiRuntimeRole;
  content: AiRuntimeMessageContent;
  createdAt?: string;
  toolCalls?: readonly AiToolCall[];
  toolResult?: AiToolResult;
};

export type AiModelCapabilities = {
  streaming: boolean;
  tools: boolean;
  inputModalities: readonly ('text' | 'image')[];
  maxContextTokens?: number;
  maxOutputTokens?: number;
};

export type AiModelDescriptor = {
  id: string;
  label: string;
  providerId?: string;
  modelId?: string;
  capabilities?: AiModelCapabilities;
};

export type AiRuntimeRequest = {
  conversationId: string;
  messages: readonly AiRuntimeMessage[];
  tools?: readonly AiToolDefinition[];
  metadata?: AiJsonObject;
  signal?: AbortSignal;
};

export type AiSerializedRuntimeRequest = {
  schemaVersion: AiRuntimeSchemaVersion;
  conversationId: string;
  messages: readonly AiRuntimeMessage[];
  tools?: readonly AiToolDefinition[];
  metadata?: AiJsonObject;
};

export type AiTokenUsage = {
  inputTokens?: number;
  outputTokens?: number;
  cachedInputTokens?: number;
  reasoningTokens?: number;
  totalTokens?: number;
};

export type AiRuntimeErrorCode =
  | 'authentication'
  | 'rate_limit'
  | 'invalid_request'
  | 'model_unavailable'
  | 'context_length'
  | 'cancelled'
  | 'transport'
  | 'provider'
  | 'tool'
  | 'unknown';

export type AiRuntimeFinishReason = 'stop' | 'tool_calls' | 'length' | 'cancelled' | 'error' | 'unknown';

export type AiRuntimeStreamEvent =
  | { type: 'delta'; text: string }
  | { type: 'tool-call-start'; callId: string; name: string }
  | { type: 'tool-call-arguments-delta'; callId: string; delta: string }
  | { type: 'tool-call'; call: AiToolCall }
  | { type: 'tool-result'; result: AiToolResult }
  | { type: 'usage'; usage: AiTokenUsage }
  | {
      type: 'error';
      message: string;
      code?: AiRuntimeErrorCode;
      retryable?: boolean;
      retryAfterMs?: number;
    }
  | { type: 'done'; finishReason?: AiRuntimeFinishReason };

export const textContent = (text: string): AiTextContentPart => ({ type: 'text', text });

export const textFromRuntimeMessage = (message: AiRuntimeMessage): string => {
  if (typeof message.content === 'string') return message.content;
  return message.content
    .filter((part): part is AiTextContentPart => part.type === 'text')
    .map((part) => part.text)
    .join('');
};

export const serializeAiRuntimeRequest = (request: AiRuntimeRequest): AiSerializedRuntimeRequest => ({
  schemaVersion: AI_RUNTIME_SCHEMA_VERSION,
  conversationId: request.conversationId,
  messages: request.messages,
  ...(request.tools ? { tools: request.tools } : {}),
  ...(request.metadata ? { metadata: request.metadata } : {}),
});

export const isTerminalAiRuntimeEvent = (event: AiRuntimeStreamEvent): boolean =>
  event.type === 'done' || event.type === 'error';

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

export type AiToolDefinition = {
  name: string;
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
  content: readonly AiRuntimeContentPart[];
  isError?: boolean;
};

export type AiRuntimeMessage = {
  id: string;
  role: AiRuntimeRole;
  content: readonly AiRuntimeContentPart[];
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
  providerId: string;
  modelId: string;
  label: string;
  capabilities: AiModelCapabilities;
};

export type AiRuntimeRequestPayload = {
  schemaVersion: AiRuntimeSchemaVersion;
  conversationId: string;
  messages: readonly AiRuntimeMessage[];
  tools?: readonly AiToolDefinition[];
  metadata?: AiJsonObject;
};

export type AiRuntimeRequest = AiRuntimeRequestPayload & {
  signal?: AbortSignal;
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

export type AiRuntimeError = {
  code: AiRuntimeErrorCode;
  message: string;
  retryable: boolean;
  retryAfterMs?: number;
};

export type AiRuntimeFinishReason = 'stop' | 'tool_calls' | 'length' | 'cancelled' | 'error' | 'unknown';

export type AiRuntimeStreamEvent =
  | { type: 'text.delta'; text: string }
  | { type: 'tool.call.start'; callId: string; name: string }
  | { type: 'tool.call.arguments.delta'; callId: string; delta: string }
  | { type: 'tool.call.ready'; call: AiToolCall }
  | { type: 'tool.result'; result: AiToolResult }
  | { type: 'usage'; usage: AiTokenUsage }
  | { type: 'error'; error: AiRuntimeError }
  | { type: 'done'; finishReason: AiRuntimeFinishReason };

export const textContent = (text: string): AiTextContentPart => ({ type: 'text', text });

export const textFromRuntimeMessage = (message: AiRuntimeMessage): string =>
  message.content
    .filter((part): part is AiTextContentPart => part.type === 'text')
    .map((part) => part.text)
    .join('');

export const isTerminalAiRuntimeEvent = (event: AiRuntimeStreamEvent): boolean =>
  event.type === 'done' || event.type === 'error';

import {
  type AiJsonObject,
  type AiRuntimeErrorCode,
  type AiRuntimeFinishReason,
  type AiRuntimeMessage,
  type AiRuntimeRequest,
  type AiRuntimeStreamEvent,
  type AiTokenUsage,
  textFromRuntimeMessage,
} from '../common/aiRuntimeTypes';
import { assertAiRuntimeRequestSafeForProvider, redactAiDiagnosticText } from '../common/aiSecurityPolicy';

export type OpenAiRuntimeSettings = {
  modelId: string;
  reasoningEffort?: string;
  storeResponses?: boolean;
};

export type OpenAiResponseTransport = (body: Record<string, unknown>, signal?: AbortSignal) => Promise<Response>;

type OpenAiStreamEvent = Record<string, unknown> & { type?: string };

const stringField = (value: unknown, key: string): string => {
  if (!value || typeof value !== 'object') return '';
  const field = (value as Record<string, unknown>)[key];
  return typeof field === 'string' ? field : '';
};

const jsonObject = (value: unknown): AiJsonObject =>
  value && typeof value === 'object' && !Array.isArray(value) ? (value as AiJsonObject) : {};

const messageContent = (message: AiRuntimeMessage): unknown => {
  if (typeof message.content === 'string') return message.content;
  const parts = message.content.map((part) =>
    part.type === 'text' ? { type: 'input_text', text: part.text } : { type: 'input_image', image_url: part.uri },
  );
  return parts.length === 1 && parts[0].type === 'input_text' ? parts[0].text : parts;
};

const openAiInput = (messages: readonly AiRuntimeMessage[]): unknown[] => {
  const input: unknown[] = [];
  for (const message of messages) {
    if (message.role === 'tool' && message.toolResult) {
      input.push({
        type: 'function_call_output',
        call_id: message.toolResult.toolCallId,
        output: textFromRuntimeMessage(message),
      });
      continue;
    }
    const text = textFromRuntimeMessage(message);
    if (text || message.content !== '')
      input.push({ role: message.role === 'tool' ? 'user' : message.role, content: messageContent(message) });
    for (const call of message.toolCalls ?? []) {
      input.push({
        type: 'function_call',
        call_id: call.id,
        name: call.name,
        arguments: JSON.stringify(call.arguments),
      });
    }
  }
  return input;
};

const usageFromResponse = (response: unknown): AiTokenUsage | null => {
  if (!response || typeof response !== 'object') return null;
  const usage = (response as { usage?: unknown }).usage;
  if (!usage || typeof usage !== 'object') return null;
  const source = usage as Record<string, unknown>;
  const inputDetails = source.input_tokens_details as Record<string, unknown> | undefined;
  const outputDetails = source.output_tokens_details as Record<string, unknown> | undefined;
  const number = (value: unknown) => (typeof value === 'number' && Number.isFinite(value) ? value : undefined);
  return {
    ...(number(source.input_tokens) !== undefined ? { inputTokens: number(source.input_tokens) } : {}),
    ...(number(source.output_tokens) !== undefined ? { outputTokens: number(source.output_tokens) } : {}),
    ...(number(inputDetails?.cached_tokens) !== undefined ? { cachedInputTokens: number(inputDetails?.cached_tokens) } : {}),
    ...(number(outputDetails?.reasoning_tokens) !== undefined
      ? { reasoningTokens: number(outputDetails?.reasoning_tokens) }
      : {}),
    ...(number(source.total_tokens) !== undefined ? { totalTokens: number(source.total_tokens) } : {}),
  };
};

async function* readServerSentEvents(response: Response): AsyncGenerator<OpenAiStreamEvent> {
  if (!response.body) throw new Error('OpenAI streaming response had no body');
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer = `${buffer}${decoder.decode(value, { stream: true })}`.replaceAll('\r\n', '\n');
      let boundary = buffer.indexOf('\n\n');
      while (boundary >= 0) {
        const block = buffer.slice(0, boundary);
        buffer = buffer.slice(boundary + 2);
        const data = block
          .split('\n')
          .filter((line) => line.startsWith('data:'))
          .map((line) => line.slice(5).trimStart())
          .join('\n');
        if (data && data !== '[DONE]') yield JSON.parse(data) as OpenAiStreamEvent;
        boundary = buffer.indexOf('\n\n');
      }
    }
  } finally {
    reader.releaseLock();
  }
}

const eventErrorCode = (code: string, type: string): AiRuntimeErrorCode => {
  const combined = `${code} ${type}`.toLocaleLowerCase();
  if (combined.includes('auth') || combined.includes('key')) return 'authentication';
  if (combined.includes('rate')) return 'rate_limit';
  if (combined.includes('context') || combined.includes('token_limit')) return 'context_length';
  if (combined.includes('model')) return 'model_unavailable';
  if (combined.includes('invalid')) return 'invalid_request';
  return 'provider';
};

const thrownErrorEvent = (error: unknown, signal?: AbortSignal): AiRuntimeStreamEvent => {
  if (signal?.aborted || (error instanceof DOMException && error.name === 'AbortError'))
    return { type: 'error', code: 'cancelled', message: 'OpenAI response was cancelled', retryable: false };
  return {
    type: 'error',
    code: error instanceof TypeError ? 'transport' : 'provider',
    message: redactAiDiagnosticText(error instanceof Error ? error.message : String(error)),
    retryable: error instanceof TypeError,
  };
};

const incompleteReason = (response: unknown): AiRuntimeFinishReason => {
  if (!response || typeof response !== 'object') return 'unknown';
  const details = (response as { incomplete_details?: unknown }).incomplete_details;
  const reason = stringField(details, 'reason').toLocaleLowerCase();
  return reason.includes('max_output') || reason.includes('length') ? 'length' : 'unknown';
};

export class OpenAiRuntimeAdapter {
  constructor(private readonly transport: OpenAiResponseTransport) {}

  async *stream(request: AiRuntimeRequest, settings: OpenAiRuntimeSettings): AsyncGenerator<AiRuntimeStreamEvent> {
    assertAiRuntimeRequestSafeForProvider(request);
    if (request.signal?.aborted) {
      yield { type: 'error', code: 'cancelled', message: 'OpenAI response was cancelled', retryable: false };
      return;
    }

    const body: Record<string, unknown> = {
      model: settings.modelId,
      stream: true,
      input: openAiInput(request.messages),
      store: settings.storeResponses ?? false,
    };
    if (request.tools?.length)
      body.tools = request.tools.map((tool) => ({
        type: 'function',
        name: tool.name,
        description: tool.description,
        parameters: tool.inputSchema,
      }));
    if (settings.reasoningEffort && settings.reasoningEffort !== 'none')
      body.reasoning = { effort: settings.reasoningEffort };

    let response: Response;
    try {
      response = await this.transport(body, request.signal);
    } catch (error) {
      yield thrownErrorEvent(error, request.signal);
      return;
    }
    if (!response.ok) {
      yield {
        type: 'error',
        code:
          response.status === 401 || response.status === 403
            ? 'authentication'
            : response.status === 429
              ? 'rate_limit'
              : response.status === 404
                ? 'model_unavailable'
                : response.status === 400 || response.status === 422
                  ? 'invalid_request'
                  : 'provider',
        message: `OpenAI request failed (HTTP ${String(response.status)})`,
        retryable: response.status === 408 || response.status === 409 || response.status === 429 || response.status >= 500,
      };
      return;
    }

    let sawToolCall = false;
    try {
      for await (const event of readServerSentEvents(response)) {
        if (request.signal?.aborted) return;
        const type = stringField(event, 'type');
        if (type === 'response.output_text.delta') {
          const delta = stringField(event, 'delta');
          if (delta) yield { type: 'delta', text: delta };
          continue;
        }
        if (type === 'response.output_item.added') {
          const item = event.item;
          if (stringField(item, 'type') === 'function_call') {
            const callId = stringField(item, 'call_id');
            const name = stringField(item, 'name');
            if (callId && name) {
              sawToolCall = true;
              yield { type: 'tool-call-start', callId, name };
            }
          }
          continue;
        }
        if (type === 'response.function_call_arguments.delta') {
          const callId = stringField(event, 'call_id');
          const delta = stringField(event, 'delta');
          if (callId && delta) yield { type: 'tool-call-arguments-delta', callId, delta };
          continue;
        }
        if (type === 'response.output_item.done') {
          const item = event.item;
          if (stringField(item, 'type') === 'function_call') {
            const callId = stringField(item, 'call_id');
            const name = stringField(item, 'name');
            try {
              const arguments_ = JSON.parse(stringField(item, 'arguments') || '{}') as unknown;
              yield { type: 'tool-call', call: { id: callId, name, arguments: jsonObject(arguments_) } };
            } catch {
              yield {
                type: 'error',
                code: 'tool',
                message: `OpenAI returned invalid arguments for tool '${name || 'unknown'}'`,
                retryable: false,
              };
              return;
            }
          }
          continue;
        }
        if (type === 'response.completed' || type === 'response.incomplete') {
          const responseObject = event.response;
          const usage = usageFromResponse(responseObject);
          if (usage) yield { type: 'usage', usage };
          yield {
            type: 'done',
            finishReason:
              type === 'response.incomplete' ? incompleteReason(responseObject) : sawToolCall ? 'tool_calls' : 'stop',
          };
          return;
        }
        if (type === 'response.failed' || type === 'error') {
          const error =
            type === 'response.failed' && event.response && typeof event.response === 'object'
              ? (event.response as { error?: unknown }).error
              : event.error;
          yield {
            type: 'error',
            code: eventErrorCode(stringField(error, 'code'), stringField(error, 'type')),
            message: redactAiDiagnosticText(stringField(error, 'message') || 'OpenAI response failed'),
            retryable: false,
          };
          return;
        }
      }
      if (!request.signal?.aborted) yield { type: 'done', finishReason: sawToolCall ? 'tool_calls' : 'unknown' };
    } catch (error) {
      yield thrownErrorEvent(error, request.signal);
    }
  }
}

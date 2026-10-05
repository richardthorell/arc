import type { BuiltInAgentToolExecutionResult } from '../../../common/builtInAgentTypes';
import { redactAiDiagnosticText } from '../../../common/aiSecurityPolicy';
import type {
  AiRuntimeMessage,
  AiRuntimeRequest,
  AiRuntimeStreamEvent,
  AiToolCall,
  AiToolResult,
} from '../../../common/aiRuntimeTypes';

export const AI_AGENT_MAX_STEPS = 10;
export const AI_AGENT_STEP_TIMEOUT_MS = 60_000;

export type AiAgentModelExecutor = (request: AiRuntimeRequest) => AsyncIterable<AiRuntimeStreamEvent>;
export type AiAgentToolInvoker = (call: AiToolCall, signal?: AbortSignal) => Promise<BuiltInAgentToolExecutionResult>;

export type AiAgentToolLoopOptions = Readonly<{
  maximumSteps?: number;
  stepTimeoutMs?: number;
}>;

const revisionSensitiveTools = new Set([
  'edit.begin',
  'edit.apply',
  'editor.applyBatch',
  'edit.commit',
  'history.undo',
  'history.redo',
]);

const runtimeMessageId = (prefix: string, step: number, index = 0): string =>
  `${prefix}-${step.toString(36)}-${index.toString(36)}-${Date.now().toString(36)}`;

const isRevisionConflict = (call: AiToolCall, message: string): boolean =>
  revisionSensitiveTools.has(call.name) && /scene revision/iu.test(message);

const normalizedToolFailure = (call: AiToolCall, error: unknown): AiToolResult => {
  const message = redactAiDiagnosticText(error instanceof Error ? error.message : String(error));
  if (isRevisionConflict(call, message)) {
    return {
      toolCallId: call.id,
      name: call.name,
      operation: call.name,
      content:
        `ARC tool revision conflict: ${message}. ` +
        'Refresh authoritative scene state with scene.overview. If an edit session is still active, use the revision reported by that session or cancel and begin a new transaction before retrying the mutation.',
      isError: true,
      errorCode: 'revision_conflict',
      retryable: true,
    };
  }
  return {
    toolCallId: call.id,
    name: call.name,
    operation: call.name,
    content: `ARC tool error: ${message}`,
    isError: true,
    errorCode: 'tool_error',
  };
};

const successfulToolResult = (call: AiToolCall, result: BuiltInAgentToolExecutionResult): AiToolResult => ({
  toolCallId: call.id,
  name: call.name,
  operation: result.operation,
  content: result.content,
  truncated: result.truncated,
  originalBytes: result.originalBytes,
});

const positiveInteger = (value: number | undefined, fallback: number, name: string): number => {
  const resolved = value ?? fallback;
  if (!Number.isSafeInteger(resolved) || resolved < 1) throw new Error(`${name} must be a positive integer`);
  return resolved;
};

const executeTool = async (
  call: AiToolCall,
  invokeTool: AiAgentToolInvoker,
  signal?: AbortSignal,
): Promise<AiToolResult> => {
  try {
    const result = signal ? await invokeTool(call, signal) : await invokeTool(call);
    return successfulToolResult(call, result);
  } catch (error) {
    return normalizedToolFailure(call, error);
  }
};

export async function* runAiAgentToolLoop(
  request: AiRuntimeRequest,
  execute: AiAgentModelExecutor,
  invokeTool: AiAgentToolInvoker,
  options: AiAgentToolLoopOptions = {},
): AsyncGenerator<AiRuntimeStreamEvent> {
  const maximumSteps = positiveInteger(options.maximumSteps, AI_AGENT_MAX_STEPS, 'maximumSteps');
  const stepTimeoutMs = positiveInteger(options.stepTimeoutMs, AI_AGENT_STEP_TIMEOUT_MS, 'stepTimeoutMs');
  const messages: AiRuntimeMessage[] = [...request.messages];
  let toolSteps = 0;

  // maximumSteps bounds provider turns that execute tools. One additional provider turn is
  // reserved for the terminal response after the final allowed tool result is available.
  for (let providerStep = 0; providerStep <= maximumSteps; ++providerStep) {
    if (request.signal?.aborted) return;

    const controller = new AbortController();
    let timedOut = false;
    const onAbort = () => controller.abort();
    request.signal?.addEventListener('abort', onAbort, { once: true });
    const timeout = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, stepTimeoutMs);

    const calls: AiToolCall[] = [];
    let stepText = '';
    let finishReason: Extract<AiRuntimeStreamEvent, { type: 'done' }>['finishReason'] = 'unknown';
    let sawDone = false;
    let terminalError = false;

    try {
      for await (const event of execute({ ...request, messages, signal: controller.signal })) {
        if (request.signal?.aborted) return;
        if (timedOut) break;

        if (event.type === 'delta') {
          stepText = `${stepText}${event.text}`;
          yield event;
          continue;
        }
        if (event.type === 'tool-call-start') {
          yield { ...event, agentStep: providerStep };
          continue;
        }
        if (event.type === 'tool-call-arguments-delta') {
          yield { ...event, agentStep: providerStep };
          continue;
        }
        if (event.type === 'tool-call') {
          calls.push(event.call);
          yield { type: 'tool-call', call: event.call, agentStep: providerStep };
          continue;
        }
        if (event.type === 'done') {
          sawDone = true;
          finishReason = event.finishReason ?? 'unknown';
          continue;
        }
        if (event.type === 'error') {
          terminalError = true;
          yield event;
          return;
        }
        yield event;
      }
    } finally {
      clearTimeout(timeout);
      request.signal?.removeEventListener('abort', onAbort);
    }

    if (request.signal?.aborted) return;
    if (timedOut) {
      yield {
        type: 'error',
        code: 'provider',
        message: `AI agent step ${providerStep + 1} exceeded the ${stepTimeoutMs} ms timeout`,
        retryable: true,
      };
      return;
    }
    if (terminalError) return;

    if (!sawDone || finishReason !== 'tool_calls') {
      yield { type: 'done', finishReason: sawDone ? finishReason : 'unknown' };
      return;
    }

    if (!calls.length) {
      yield {
        type: 'error',
        code: 'tool',
        message: 'AI provider requested tool execution but returned no complete tool calls',
        retryable: false,
      };
      return;
    }

    if (toolSteps >= maximumSteps) {
      yield {
        type: 'error',
        code: 'tool',
        message: `AI agent reached the maximum of ${maximumSteps} tool steps`,
        retryable: false,
      };
      return;
    }

    messages.push({
      id: runtimeMessageId('agent-assistant', providerStep),
      role: 'assistant',
      content: stepText,
      toolCalls: calls,
    });

    for (let index = 0; index < calls.length; ++index) {
      if (request.signal?.aborted) return;
      const call = calls[index]!;
      const result = await executeTool(call, invokeTool, request.signal);
      if (request.signal?.aborted) return;
      yield { type: 'tool-result', result, agentStep: providerStep };
      messages.push({
        id: runtimeMessageId('agent-tool', providerStep, index),
        role: 'tool',
        content: result.content,
        toolResult: result,
      });
    }
    ++toolSteps;
  }
}
import type {
  AiConversationMessage,
  AiConversationToolReference,
} from '../../../common/aiConversationTypes';
import {
  textContent,
  type AiRuntimeMessage,
  type AiToolCall,
  type AiToolResult,
} from '../../../common/aiRuntimeTypes';

export const AI_CONVERSATION_TOOL_RESULT_MAX_CHARACTERS = 16 * 1024;

const normalizeStep = (step: number | undefined): number =>
  Number.isSafeInteger(step) && Number(step) >= 0 ? Number(step) : 0;

const resultText = (result: AiToolResult): string => {
  if (typeof result.content === 'string') return result.content;
  return result.content
    .filter((part) => part.type === 'text')
    .map((part) => part.text)
    .join('');
};

const boundedResult = (text: string): { content: string; truncated: boolean } => {
  if (text.length <= AI_CONVERSATION_TOOL_RESULT_MAX_CHARACTERS) return { content: text, truncated: false };
  const retained = AI_CONVERSATION_TOOL_RESULT_MAX_CHARACTERS - 96;
  return {
    content:
      `${text.slice(0, retained).trimEnd()}\n` +
      `[ARC persisted tool-result preview truncated from ${text.length.toLocaleString()} characters]`,
    truncated: true,
  };
};

const compactErrorSummary = (text: string): string => {
  const normalized = text.replace(/\s+/gu, ' ').trim();
  return normalized.length > 240 ? `${normalized.slice(0, 237).trimEnd()}…` : normalized;
};

export const recordConversationToolCall = (
  references: readonly AiConversationToolReference[] | undefined,
  call: AiToolCall,
  agentStep: number | undefined,
  timestamp: string,
): AiConversationToolReference[] => {
  const current = references ?? [];
  const index = current.findIndex((reference) => reference.toolCallId === call.id);
  const next: AiConversationToolReference = {
    ...(index >= 0 ? current[index] : {}),
    toolCallId: call.id,
    name: call.name,
    state: 'pending',
    step: normalizeStep(agentStep),
    arguments: call.arguments,
    operation: call.name,
    startedAt: index >= 0 ? current[index]?.startedAt ?? timestamp : timestamp,
  };
  if (index < 0) return [...current, next];
  return current.map((reference, candidate) => (candidate === index ? next : reference));
};

export const recordConversationToolResult = (
  references: readonly AiConversationToolReference[] | undefined,
  result: AiToolResult,
  agentStep: number | undefined,
  timestamp: string,
): AiConversationToolReference[] => {
  const current = references ?? [];
  const index = current.findIndex((reference) => reference.toolCallId === result.toolCallId);
  const text = resultText(result);
  const bounded = boundedResult(text);
  const existing = index >= 0 ? current[index] : undefined;
  const truncated = Boolean(result.truncated || bounded.truncated);
  const next: AiConversationToolReference = {
    ...(existing ?? {}),
    toolCallId: result.toolCallId,
    name: result.name,
    state: result.isError ? 'error' : 'complete',
    step: existing?.step ?? normalizeStep(agentStep),
    operation: result.operation ?? existing?.operation ?? result.name,
    resultContent: bounded.content,
    resultTruncated: truncated,
    ...(result.originalBytes !== undefined ? { originalBytes: result.originalBytes } : {}),
    ...(result.errorCode ? { errorCode: result.errorCode } : {}),
    ...(result.retryable !== undefined ? { retryable: result.retryable } : {}),
    summary: result.isError
      ? compactErrorSummary(text)
      : truncated && result.originalBytes
        ? `Completed · result truncated from ${result.originalBytes.toLocaleString()} bytes`
        : 'Completed',
    startedAt: existing?.startedAt ?? timestamp,
    completedAt: timestamp,
  };
  if (index < 0) return [...current, next];
  return current.map((reference, candidate) => (candidate === index ? next : reference));
};

export const finishPendingConversationTools = (
  references: readonly AiConversationToolReference[] | undefined,
  state: 'error' | 'cancelled',
  summary: string,
  timestamp: string,
): AiConversationToolReference[] | undefined => {
  if (!references?.length) return references ? [...references] : undefined;
  return references.map((reference) =>
    reference.state === 'pending'
      ? {
          ...reference,
          state,
          summary,
          completedAt: timestamp,
        }
      : reference,
  );
};

const replayableReferences = (message: AiConversationMessage): AiConversationToolReference[] =>
  (message.toolReferences ?? []).filter(
    (reference) =>
      (reference.state === 'complete' || reference.state === 'error') &&
      reference.arguments !== undefined &&
      reference.resultContent !== undefined,
  );

export const runtimeMessagesForConversationMessage = (
  message: AiConversationMessage,
  baseMessage?: AiRuntimeMessage,
): AiRuntimeMessage[] => {
  const base: AiRuntimeMessage =
    baseMessage ?? {
      id: message.id,
      role: message.role,
      content: [textContent(message.content)],
      createdAt: message.createdAt,
    };
  if (message.role !== 'assistant') return [base];

  const replayable = replayableReferences(message);
  if (!replayable.length) return [base];

  const byStep = new Map<number, AiConversationToolReference[]>();
  for (const reference of replayable) {
    const step = normalizeStep(reference.step);
    const entries = byStep.get(step) ?? [];
    entries.push(reference);
    byStep.set(step, entries);
  }

  const history: AiRuntimeMessage[] = [];
  for (const [step, references] of [...byStep.entries()].sort(([left], [right]) => left - right)) {
    history.push({
      id: `${message.id}:agent-step:${step}`,
      role: 'assistant',
      content: '',
      createdAt: references[0]?.startedAt ?? message.createdAt,
      toolCalls: references.map((reference) => ({
        id: reference.toolCallId,
        name: reference.name,
        arguments: reference.arguments ?? {},
      })),
    });
    for (const reference of references) {
      const result: AiToolResult = {
        toolCallId: reference.toolCallId,
        name: reference.name,
        content: reference.resultContent ?? '',
        ...(reference.state === 'error' ? { isError: true } : {}),
        ...(reference.errorCode ? { errorCode: reference.errorCode } : {}),
        ...(reference.retryable !== undefined ? { retryable: reference.retryable } : {}),
        ...(reference.operation ? { operation: reference.operation } : {}),
        ...(reference.resultTruncated !== undefined ? { truncated: reference.resultTruncated } : {}),
        ...(reference.originalBytes !== undefined ? { originalBytes: reference.originalBytes } : {}),
      };
      history.push({
        id: `${message.id}:agent-result:${reference.toolCallId}`,
        role: 'tool',
        content: result.content,
        createdAt: reference.completedAt ?? message.createdAt,
        toolResult: result,
      });
    }
  }

  return [...history, base];
};

export const runtimeMessagesForConversation = (messages: readonly AiConversationMessage[]): AiRuntimeMessage[] =>
  messages.flatMap((message) => runtimeMessagesForConversationMessage(message));

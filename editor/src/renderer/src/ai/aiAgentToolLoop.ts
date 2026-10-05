import type { BuiltInAgentToolExecutionResult } from '../../../common/builtInAgentTypes';
import { redactAiDiagnosticText } from '../../../common/aiSecurityPolicy';
import type {
  AiRuntimeMessage,
  AiRuntimeRequest,
  AiRuntimeStreamEvent,
  AiTaskProgress,
  AiToolCall,
  AiToolResult,
} from '../../../common/aiRuntimeTypes';
import {
  AI_AGENT_PLAN_TOOL_NAME,
  agentPlanToolResult,
  parseAgentPlan,
  taskUpdatesForAgentPlan,
} from './aiAgentPlan';

export const AI_AGENT_MAX_STEPS = 10;
export const AI_AGENT_STEP_TIMEOUT_MS = 60_000;
const additionalPlanTurns = 4;

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

const taskForCalls = (
  providerStep: number,
  calls: readonly AiToolCall[],
  state: AiTaskProgress['state'],
  detail?: string,
): AiTaskProgress => ({
  id: `agent-step-${providerStep}`,
  title: calls.length === 1 ? calls[0]!.name : `Run ${calls.length} editor operations`,
  state,
  agentStep: providerStep,
  toolCallIds: calls.map((call) => call.id),
  ...(detail ? { detail } : {}),
});

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

const activePlanTask = (tasks: ReadonlyMap<string, AiTaskProgress>): AiTaskProgress | undefined => {
  const inProgress = [...tasks.values()].filter(
    (task) => task.planId && task.parentId && task.state === 'in_progress',
  );
  if (!inProgress.length) return undefined;
  const parentIds = new Set(inProgress.map((task) => task.parentId));
  return (
    inProgress.find((task) => !parentIds.has(task.id) && ![...tasks.values()].some((candidate) => candidate.parentId === task.id && candidate.state === 'in_progress')) ??
    inProgress.at(-1)
  );
};

const withLinkedCalls = (task: AiTaskProgress, calls: readonly AiToolCall[], providerStep: number): AiTaskProgress => ({
  ...task,
  agentStep: providerStep,
  toolCallIds: [...new Set([...(task.toolCallIds ?? []), ...calls.map((call) => call.id)])],
});

export async function* runAiAgentToolLoop(
  request: AiRuntimeRequest,
  execute: AiAgentModelExecutor,
  invokeTool: AiAgentToolInvoker,
  options: AiAgentToolLoopOptions = {},
): AsyncGenerator<AiRuntimeStreamEvent> {
  const maximumSteps = positiveInteger(options.maximumSteps, AI_AGENT_MAX_STEPS, 'maximumSteps');
  const stepTimeoutMs = positiveInteger(options.stepTimeoutMs, AI_AGENT_STEP_TIMEOUT_MS, 'stepTimeoutMs');
  const messages: AiRuntimeMessage[] = [...request.messages];
  const planTasks = new Map<string, AiTaskProgress>();
  let toolSteps = 0;

  // Plan-only provider turns do not consume the editor-tool budget. A small extra
  // allowance lets the model publish/revise a plan without stealing mutation turns.
  for (let providerStep = 0; providerStep <= maximumSteps + additionalPlanTurns; ++providerStep) {
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
        if (event.type === 'tool-call-start' || event.type === 'tool-call-arguments-delta') {
          if (event.name !== AI_AGENT_PLAN_TOOL_NAME) yield { ...event, agentStep: providerStep };
          continue;
        }
        if (event.type === 'tool-call') {
          calls.push(event.call);
          if (event.call.name !== AI_AGENT_PLAN_TOOL_NAME)
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

    const planCalls = calls.filter((call) => call.name === AI_AGENT_PLAN_TOOL_NAME);
    const executionCalls = calls.filter((call) => call.name !== AI_AGENT_PLAN_TOOL_NAME);
    if (executionCalls.length && toolSteps >= maximumSteps) {
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

    for (let index = 0; index < planCalls.length; ++index) {
      const call = planCalls[index]!;
      let result: AiToolResult;
      try {
        const updates = taskUpdatesForAgentPlan(parseAgentPlan(call.arguments), providerStep);
        for (const task of updates) {
          const previous = planTasks.get(task.id);
          const merged: AiTaskProgress = {
            ...previous,
            ...task,
            ...(previous?.toolCallIds && !task.toolCallIds ? { toolCallIds: previous.toolCallIds } : {}),
          };
          planTasks.set(task.id, merged);
          yield { type: 'task-update', task: merged };
        }
        result = agentPlanToolResult(call, updates.length - 1);
      } catch (error) {
        result = normalizedToolFailure(call, error);
      }
      messages.push({
        id: runtimeMessageId('agent-plan', providerStep, index),
        role: 'tool',
        content: result.content,
        toolResult: result,
      });
    }

    if (!executionCalls.length) continue;

    const plannedTask = activePlanTask(planTasks);
    let genericTask: AiTaskProgress | undefined;
    if (plannedTask) {
      const linked = withLinkedCalls(plannedTask, executionCalls, providerStep);
      planTasks.set(linked.id, linked);
      yield { type: 'task-update', task: linked };
    } else {
      genericTask = taskForCalls(providerStep, executionCalls, 'in_progress');
      yield { type: 'task-update', task: genericTask };
    }

    let failedResult: AiToolResult | undefined;
    for (let index = 0; index < executionCalls.length; ++index) {
      if (request.signal?.aborted) return;
      const call = executionCalls[index]!;
      const result = await executeTool(call, invokeTool, request.signal);
      if (request.signal?.aborted) return;
      if (result.isError && !failedResult) failedResult = result;
      yield { type: 'tool-result', result, agentStep: providerStep };
      messages.push({
        id: runtimeMessageId('agent-tool', providerStep, index),
        role: 'tool',
        content: result.content,
        toolResult: result,
      });
    }

    if (plannedTask && failedResult) {
      const failed: AiTaskProgress = {
        ...planTasks.get(plannedTask.id)!,
        state: 'failed',
        detail: `Failed while running ${failedResult.name}`,
      };
      planTasks.set(failed.id, failed);
      yield { type: 'task-update', task: failed };
      const root = failed.planId ? planTasks.get(failed.planId) : undefined;
      if (root) {
        const failedRoot: AiTaskProgress = { ...root, state: 'failed' };
        planTasks.set(failedRoot.id, failedRoot);
        yield { type: 'task-update', task: failedRoot };
      }
    } else if (genericTask) {
      yield {
        type: 'task-update',
        task: taskForCalls(
          providerStep,
          executionCalls,
          failedResult ? 'failed' : 'completed',
          failedResult ? `Failed while running ${failedResult.name}` : undefined,
        ),
      };
    }
    ++toolSteps;
  }
}

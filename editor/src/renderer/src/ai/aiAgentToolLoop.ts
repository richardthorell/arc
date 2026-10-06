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
  flattenTaskTree,
  mapTaskTree,
  parseAgentPlan,
  taskTreeForAgentPlan,
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

const preserveRuntimeTaskData = (previous: AiTaskProgress | undefined, next: AiTaskProgress): AiTaskProgress => {
  if (!previous) return next;

  const merge = (old: AiTaskProgress | undefined, task: AiTaskProgress): AiTaskProgress => {
    if (!old) {
      return {
        ...task,
        ...(task.children?.length ? { children: task.children.map((child) => merge(undefined, child)) } : {}),
      };
    }

    const nextChildren = new Map((task.children ?? []).map((child) => [child.id, child]));
    const previousChildren = old.children ?? [];
    const mergedChildren = [
      ...previousChildren.map((child) => merge(child, nextChildren.get(child.id) ?? child)),
      ...(task.children ?? [])
        .filter((child) => !previousChildren.some((previousChild) => previousChild.id === child.id))
        .map((child) => merge(undefined, child)),
    ];

    const preserveTerminalState =
      (old.state === 'completed' || old.state === 'cancelled') &&
      (task.state === 'planned' || task.state === 'in_progress');

    return {
      ...task,
      state: preserveTerminalState ? old.state : task.state,
      ...(old.toolCallIds?.length ? { toolCallIds: old.toolCallIds } : {}),
      ...(mergedChildren.length ? { children: mergedChildren } : {}),
    };
  };

  return merge(previous, next);
};

const activePlanTask = (
  roots: ReadonlyMap<string, AiTaskProgress>,
): { root: AiTaskProgress; task: AiTaskProgress } | undefined => {
  const candidates = [...roots.values()].reverse();
  for (const root of candidates) {
    const flat = flattenTaskTree(root);
    const parentIds = new Set(flat.flatMap((task) => task.children?.map((child) => child.id) ?? []));
    const active = [...flat]
      .reverse()
      .find((task) => task.id !== root.id && task.state === 'in_progress' && !task.children?.length);
    if (active) return { root, task: active };
    const fallback = [...flat]
      .reverse()
      .find((task) => task.id !== root.id && task.state === 'in_progress' && !parentIds.has(task.id));
    if (fallback) return { root, task: fallback };
  }
  return undefined;
};

const withLinkedCalls = (task: AiTaskProgress, calls: readonly AiToolCall[], providerStep: number): AiTaskProgress => ({
  ...task,
  agentStep: providerStep,
  toolCallIds: [...new Set([...(task.toolCallIds ?? []), ...calls.map((call) => call.id)])],
});

const agentControlTools = new Set(['edit.begin', 'edit.request', 'edit.commit', 'edit.cancel']);

const failedPlanTask = (
  roots: ReadonlyMap<string, AiTaskProgress>,
): { root: AiTaskProgress; task: AiTaskProgress } | undefined => {
  const candidates = [...roots.values()].reverse();
  for (const root of candidates) {
    const failed = [...flattenTaskTree(root)]
      .reverse()
      .find((task) => task.id !== root.id && task.state === 'failed' && !task.children?.length);
    if (failed) return { root, task: failed };
  }
  return undefined;
};

const deriveContainerState = (children: readonly AiTaskProgress[]): AiTaskProgress['state'] => {
  if (children.some((child) => child.state === 'failed')) return 'failed';
  if (children.some((child) => child.state === 'in_progress')) return 'in_progress';
  if (children.some((child) => child.state === 'planned')) {
    return children.some((child) => child.state === 'completed') ? 'in_progress' : 'planned';
  }
  if (children.length && children.every((child) => child.state === 'cancelled')) return 'cancelled';
  return 'completed';
};

const deriveTaskTreeStates = (task: AiTaskProgress): AiTaskProgress => {
  if (!task.children?.length) return task;
  const children = task.children.map(deriveTaskTreeStates);
  return { ...task, children, state: deriveContainerState(children) };
};

const withoutTaskDetail = (task: AiTaskProgress): AiTaskProgress => {
  const copy = { ...task };
  delete copy.detail;
  return copy;
};

const advanceCompletedPlanTask = (
  root: AiTaskProgress,
  completedTaskId: string,
): AiTaskProgress => {
  let nextRoot = mapTaskTree(root, completedTaskId, (task) => ({
    ...withoutTaskDetail(task),
    state: 'completed',
  }));

  const flat = flattenTaskTree(nextRoot);
  const nextPlanned = flat.find(
    (task) => task.id !== nextRoot.id && task.state === 'planned' && !task.children?.length,
  );
  if (nextPlanned) {
    nextRoot = mapTaskTree(nextRoot, nextPlanned.id, (task) => ({
      ...withoutTaskDetail(task),
      state: 'in_progress',
    }));
  }

  return deriveTaskTreeStates(nextRoot);
};

const hasNextPlannedLeaf = (root: AiTaskProgress, currentTaskId: string): boolean => {
  const flat = flattenTaskTree(root);
  const currentIndex = flat.findIndex((task) => task.id === currentTaskId);
  return flat.slice(currentIndex + 1).some((task) => task.state === 'planned' && !task.children?.length);
};

const shouldAdvanceForNewSemanticTurn = (root: AiTaskProgress, task: AiTaskProgress): boolean =>
  Boolean(task.toolCallIds?.length) &&
  !task.detail?.startsWith('Retrying after ') &&
  hasNextPlannedLeaf(root, task.id);

export async function* runAiAgentToolLoop(
  request: AiRuntimeRequest,
  execute: AiAgentModelExecutor,
  invokeTool: AiAgentToolInvoker,
  options: AiAgentToolLoopOptions = {},
): AsyncGenerator<AiRuntimeStreamEvent> {
  const maximumSteps = positiveInteger(options.maximumSteps, AI_AGENT_MAX_STEPS, 'maximumSteps');
  const stepTimeoutMs = positiveInteger(options.stepTimeoutMs, AI_AGENT_STEP_TIMEOUT_MS, 'stepTimeoutMs');
  const messages: AiRuntimeMessage[] = [...request.messages];
  const planRoots = new Map<string, AiTaskProgress>();
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
    const localPlanCallIds = new Set<string>();
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
          if (event.name === AI_AGENT_PLAN_TOOL_NAME) localPlanCallIds.add(event.callId);
          else yield { ...event, agentStep: providerStep };
          continue;
        }
        if (event.type === 'tool-call-arguments-delta') {
          if (!localPlanCallIds.has(event.callId)) yield { ...event, agentStep: providerStep };
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
      const unresolved = activePlanTask(planRoots);
      if (unresolved?.task.detail?.startsWith('Retrying after ')) {
        let failedRoot = mapTaskTree(unresolved.root, unresolved.task.id, (task) => ({
          ...task,
          state: 'failed',
          detail: task.detail?.replace(/^Retrying after /u, 'Failed while running '),
        }));
        failedRoot = deriveTaskTreeStates(failedRoot);
        planRoots.set(failedRoot.id, failedRoot);
        yield { type: 'task-update', task: failedRoot };
      }
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
    const semanticExecutionCalls = executionCalls.filter((call) => !agentControlTools.has(call.name));
    if (semanticExecutionCalls.length && toolSteps >= maximumSteps) {
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
        const parsed = parseAgentPlan(call.arguments);
        const proposedRoot = taskTreeForAgentPlan(parsed, providerStep);
        const root = preserveRuntimeTaskData(planRoots.get(proposedRoot.id), proposedRoot);
        planRoots.set(root.id, root);
        yield { type: 'task-update', task: root };
        result = agentPlanToolResult(call, flattenTaskTree(root).length - 1);
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

    let planned = activePlanTask(planRoots);
    let recoveringPlanTask = false;
    const hasSemanticExecution = semanticExecutionCalls.length > 0;

    if (!planned && planRoots.size && hasSemanticExecution) {
      const failed = failedPlanTask(planRoots);
      if (failed) {
        let recoveringRoot = mapTaskTree(failed.root, failed.task.id, (task) => ({
          ...withoutTaskDetail(withLinkedCalls(task, semanticExecutionCalls, providerStep)),
          state: 'in_progress',
        }));
        recoveringRoot = deriveTaskTreeStates(recoveringRoot);
        planRoots.set(recoveringRoot.id, recoveringRoot);
        planned = {
          root: recoveringRoot,
          task: flattenTaskTree(recoveringRoot).find((task) => task.id === failed.task.id)!,
        };
        recoveringPlanTask = true;
        yield { type: 'task-update', task: recoveringRoot };
      }
    }

    let genericTask: AiTaskProgress | undefined;
    const retryingPlanTask = Boolean(planned?.task.detail?.startsWith('Retrying after '));
    if (planned && !recoveringPlanTask) {
      if (hasSemanticExecution && shouldAdvanceForNewSemanticTurn(planned.root, planned.task)) {
        const advancedRoot = advanceCompletedPlanTask(planRoots.get(planned.root.id)!, planned.task.id);
        planRoots.set(advancedRoot.id, advancedRoot);
        yield { type: 'task-update', task: advancedRoot };

        const advanced = activePlanTask(planRoots);
        if (advanced) planned = advanced;
      }

      const linkedRoot = mapTaskTree(planned.root, planned.task.id, (task) =>
        withLinkedCalls(
          task.detail?.startsWith('Retrying after ') ? withoutTaskDetail(task) : task,
          semanticExecutionCalls,
          providerStep,
        ),
      );
      planRoots.set(linkedRoot.id, linkedRoot);
      planned = {
        root: linkedRoot,
        task: flattenTaskTree(linkedRoot).find((task) => task.id === planned!.task.id)!,
      };
      yield { type: 'task-update', task: linkedRoot };
    } else if (!planned && !planRoots.size && hasSemanticExecution) {
      genericTask = taskForCalls(providerStep, semanticExecutionCalls, 'in_progress');
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

    if (planned && failedResult) {
      let retryingRoot = mapTaskTree(planRoots.get(planned.root.id)!, planned.task.id, (task) => ({
        ...task,
        state: 'in_progress',
        detail: `Retrying after ${failedResult.name}`,
      }));
      retryingRoot = deriveTaskTreeStates(retryingRoot);
      planRoots.set(retryingRoot.id, retryingRoot);
      yield { type: 'task-update', task: retryingRoot };
    } else if (planned && (recoveringPlanTask || retryingPlanTask)) {
      const recoveredRoot = advanceCompletedPlanTask(planRoots.get(planned.root.id)!, planned.task.id);
      planRoots.set(recoveredRoot.id, recoveredRoot);
      yield { type: 'task-update', task: recoveredRoot };
    } else if (genericTask) {
      yield {
        type: 'task-update',
        task: taskForCalls(
          providerStep,
          executionCalls,
          failedResult ? (failedResult.retryable ? 'in_progress' : 'failed') : 'completed',
          failedResult
            ? failedResult.retryable
              ? `Retrying after ${failedResult.name}`
              : `Failed while running ${failedResult.name}`
            : undefined,
        ),
      };
    }
    if (hasSemanticExecution) ++toolSteps;
  }
}

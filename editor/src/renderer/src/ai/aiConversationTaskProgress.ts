import type { AiConversationTaskReference } from '../../../common/aiConversationTypes';
import type { AiTaskProgress, AiTaskProgressState } from '../../../common/aiRuntimeTypes';

const terminalTaskStates = new Set<AiTaskProgressState>(['completed', 'failed', 'cancelled']);

const resolvedTaskState = (
  task: AiTaskProgress,
  existing: AiConversationTaskReference | undefined,
): AiTaskProgressState => {
  if (existing && terminalTaskStates.has(existing.state)) return existing.state;
  return task.state;
};

const toConversationTaskReference = (
  task: AiTaskProgress,
  timestamp: string,
  existing?: AiConversationTaskReference,
  blockedByFailure = false,
): AiConversationTaskReference => {
  const existingChildren = new Map((existing?.children ?? []).map((child) => [child.id, child]));
  const state = blockedByFailure && existing?.state === 'planned' ? 'cancelled' : resolvedTaskState(task, existing);
  const startedAt = existing?.startedAt ?? (state === 'planned' ? undefined : timestamp);

  let childBlocked = false;
  const children = task.children?.map((child) => {
    const existingChild = existingChildren.get(child.id);
    const next = toConversationTaskReference(child, timestamp, existingChild, childBlocked);
    if (next.state === 'failed') childBlocked = true;
    return next;
  });

  return {
    id: task.id,
    title: task.title,
    state,
    ...(task.agentStep !== undefined ? { step: task.agentStep } : {}),
    ...(task.toolCallIds
      ? { toolCallIds: [...task.toolCallIds] }
      : existing?.toolCallIds
        ? { toolCallIds: [...existing.toolCallIds] }
        : {}),
    ...(state === existing?.state && existing?.detail
      ? { detail: existing.detail }
      : task.detail
        ? { detail: task.detail }
        : {}),
    ...(task.planId ? { planId: task.planId } : {}),
    ...(task.parentId ? { parentId: task.parentId } : {}),
    ...(task.order !== undefined ? { order: task.order } : {}),
    ...(children?.length ? { children } : {}),
    ...(startedAt ? { startedAt } : {}),
    ...(terminalTaskStates.has(state) ? { completedAt: existing?.completedAt ?? timestamp } : {}),
  };
};

export const recordConversationTaskUpdate = (
  references: readonly AiConversationTaskReference[] | undefined,
  task: AiTaskProgress,
  timestamp: string,
): AiConversationTaskReference[] => {
  const current = references ?? [];
  const index = current.findIndex((reference) => reference.id === task.id);
  const existing = index >= 0 ? current[index] : undefined;
  const next = toConversationTaskReference(task, timestamp, existing);
  if (index < 0) return [...current, next];
  return current.map((reference, candidate) => (candidate === index ? next : { ...reference }));
};

const finishTask = (
  reference: AiConversationTaskReference,
  state: Extract<AiTaskProgressState, 'failed' | 'cancelled'>,
  detail: string,
  timestamp: string,
): AiConversationTaskReference => ({
  ...reference,
  ...(reference.state === 'planned' || reference.state === 'in_progress'
    ? { state, detail, completedAt: timestamp }
    : {}),
  ...(reference.children?.length
    ? { children: reference.children.map((child) => finishTask(child, state, detail, timestamp)) }
    : {}),
});

export const finishPendingConversationTasks = (
  references: readonly AiConversationTaskReference[] | undefined,
  state: Extract<AiTaskProgressState, 'failed' | 'cancelled'>,
  detail: string,
  timestamp: string,
): AiConversationTaskReference[] | undefined =>
  references?.map((reference) => finishTask(reference, state, detail, timestamp));

import type { AiConversationTaskReference } from '../../../common/aiConversationTypes';
import type { AiTaskProgress, AiTaskProgressState } from '../../../common/aiRuntimeTypes';

const terminalTaskStates = new Set<AiTaskProgressState>(['completed', 'failed', 'cancelled']);

const toConversationTaskReference = (
  task: AiTaskProgress,
  timestamp: string,
  existing?: AiConversationTaskReference,
): AiConversationTaskReference => {
  const existingChildren = new Map((existing?.children ?? []).map((child) => [child.id, child]));
  const startedAt = existing?.startedAt ?? (task.state === 'planned' ? undefined : timestamp);
  return {
    id: task.id,
    title: task.title,
    state: task.state,
    ...(task.agentStep !== undefined ? { step: task.agentStep } : {}),
    ...(task.toolCallIds ? { toolCallIds: [...task.toolCallIds] } : existing?.toolCallIds ? { toolCallIds: [...existing.toolCallIds] } : {}),
    ...(task.detail ? { detail: task.detail } : {}),
    ...(task.planId ? { planId: task.planId } : {}),
    ...(task.parentId ? { parentId: task.parentId } : {}),
    ...(task.order !== undefined ? { order: task.order } : {}),
    ...(task.children?.length
      ? {
          children: task.children.map((child) =>
            toConversationTaskReference(child, timestamp, existingChildren.get(child.id)),
          ),
        }
      : {}),
    ...(startedAt ? { startedAt } : {}),
    ...(terminalTaskStates.has(task.state) ? { completedAt: timestamp } : {}),
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
  return current.map((reference, candidate) => (candidate === index ? { ...reference, ...next } : { ...reference }));
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

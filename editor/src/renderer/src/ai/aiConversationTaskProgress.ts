import type { AiConversationTaskReference } from '../../../common/aiConversationTypes';
import type { AiTaskProgress, AiTaskProgressState } from '../../../common/aiRuntimeTypes';

const terminalTaskStates = new Set<AiTaskProgressState>(['completed', 'failed', 'cancelled']);

export const recordConversationTaskUpdate = (
  references: readonly AiConversationTaskReference[] | undefined,
  task: AiTaskProgress,
  timestamp: string,
): AiConversationTaskReference[] => {
  const next: AiConversationTaskReference = {
    id: task.id,
    title: task.title,
    state: task.state,
    ...(task.agentStep !== undefined ? { step: task.agentStep } : {}),
    ...(task.toolCallIds ? { toolCallIds: [...task.toolCallIds] } : {}),
    ...(task.detail ? { detail: task.detail } : {}),
    startedAt:
      references?.find((reference) => reference.id === task.id)?.startedAt ??
      (task.state === 'planned' ? undefined : timestamp),
    ...(terminalTaskStates.has(task.state) ? { completedAt: timestamp } : {}),
  };
  const current = references ?? [];
  const index = current.findIndex((reference) => reference.id === task.id);
  if (index < 0) return [...current, next];
  return current.map((reference, candidate) => (candidate === index ? { ...reference, ...next } : { ...reference }));
};

export const finishPendingConversationTasks = (
  references: readonly AiConversationTaskReference[] | undefined,
  state: Extract<AiTaskProgressState, 'failed' | 'cancelled'>,
  detail: string,
  timestamp: string,
): AiConversationTaskReference[] | undefined =>
  references?.map((reference) =>
    reference.state === 'planned' || reference.state === 'in_progress'
      ? { ...reference, state, detail, completedAt: timestamp }
      : { ...reference },
  );

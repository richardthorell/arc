import { Check } from 'lucide-react';
import type { AiConversationTaskReference, AiConversationToolReference } from '../../../common/aiConversationTypes';
import './aiChatLiveProgress.css';

type ProgressRowState = 'complete' | 'working' | 'waiting';

type ProgressRow = {
  id: string;
  title: string;
  state: ProgressRowState;
};

type AiChatLiveProgressProps = {
  tasks?: readonly AiConversationTaskReference[];
  tools?: readonly AiConversationToolReference[];
  showIdle?: boolean;
};

const collectRunnableTasks = (references: readonly AiConversationTaskReference[] | undefined): ProgressRow[] => {
  const rows: ProgressRow[] = [];

  const visit = (reference: AiConversationTaskReference) => {
    if (reference.children?.length) {
      for (const child of reference.children) visit(child);
      return;
    }
    if (reference.state === 'completed') rows.push({ id: reference.id, title: reference.title, state: 'complete' });
    if (reference.state === 'in_progress') rows.push({ id: reference.id, title: reference.title, state: 'working' });
    if (reference.state === 'planned') rows.push({ id: reference.id, title: reference.title, state: 'waiting' });
  };

  for (const reference of references ?? []) visit(reference);
  return rows;
};

const friendlyToolLabel = (reference: AiConversationToolReference): string => {
  if (reference.summary?.trim()) return reference.summary.trim();
  const operation = reference.operation ?? reference.name;
  if (operation === 'editor.applyBatch' || operation.startsWith('edit.')) return 'Applying editor changes';
  if (operation.startsWith('scene.')) return 'Inspecting the scene';
  if (operation.startsWith('viewport.')) return 'Inspecting the viewport';
  if (operation.startsWith('assets.')) return 'Inspecting project assets';
  if (operation.startsWith('diagnostics.')) return 'Checking diagnostics';
  return 'Working on the request';
};

const collectPendingTools = (references: readonly AiConversationToolReference[] | undefined): ProgressRow[] =>
  (references ?? [])
    .filter((reference) => reference.state === 'pending')
    .map((reference) => ({
      id: reference.toolCallId,
      title: friendlyToolLabel(reference),
      state: 'working' as const,
    }));

export function AiChatLiveProgress({ tasks, tools, showIdle = false }: AiChatLiveProgressProps) {
  const taskRows = collectRunnableTasks(tasks);
  const rows = taskRows.length ? taskRows : collectPendingTools(tools);

  if (!rows.length && !showIdle) return null;
  const visibleRows = rows.length ? rows : [{ id: 'idle', title: 'Thinking…', state: 'working' as const }];

  return (
    <div className="ai-chat-live-progress" aria-live="polite" aria-label="AI progress" role="status">
      {visibleRows.map((row) => (
        <div className={`ai-chat-live-progress-row is-${row.state}`} data-progress-state={row.state} key={row.id}>
          <span className="ai-chat-live-progress-indicator" aria-hidden="true">
            {row.state === 'complete' ? <Check size={11} strokeWidth={2.4} /> : null}
          </span>
          <span className="ai-chat-live-progress-label" key={`${row.id}:${row.state}:${row.title}`}>
            {row.title}
          </span>
          {row.state === 'waiting' ? <span className="ai-chat-live-progress-state">Waiting</span> : null}
        </div>
      ))}
    </div>
  );
}

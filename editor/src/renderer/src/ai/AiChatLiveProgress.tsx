import { Check, Copy, ListChecks, X } from 'lucide-react';
import { useEffect, useRef, useState } from 'react';
import type { AiConversationTaskReference, AiConversationToolReference } from '../../../common/aiConversationTypes';
import './aiChatLiveProgress.css';

type ProgressRowState = 'complete' | 'failed' | 'working' | 'waiting';

type ProgressRow = {
  id: string;
  title: string;
  state: ProgressRowState;
};

type AiChatLiveProgressProps = {
  tasks?: readonly AiConversationTaskReference[];
  tools?: readonly AiConversationToolReference[];
  diagnostics?: readonly string[];
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
    if (reference.state === 'failed') rows.push({ id: reference.id, title: reference.title, state: 'failed' });
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

export function AiChatLiveProgress({ tasks, tools, diagnostics, showIdle = false }: AiChatLiveProgressProps) {
  const taskRows = collectRunnableTasks(tasks);
  const rows = taskRows.length ? taskRows : collectPendingTools(tools);
  const visibleRows = rows.length ? rows : [{ id: 'idle', title: 'Thinking…', state: 'working' as const }];
  const hasFailure = rows.some((row) => row.state === 'failed');
  const hasActiveWork = rows.some((row) => row.state === 'working' || row.state === 'waiting');
  const shellRef = useRef<HTMLDivElement | null>(null);
  const [messageStreaming, setMessageStreaming] = useState(hasActiveWork);
  const [expanded, setExpanded] = useState(hasFailure || hasActiveWork);
  const [diagnosticsCopied, setDiagnosticsCopied] = useState(false);
  const wasTerminalRef = useRef(false);

  useEffect(() => {
    const turn = shellRef.current?.closest('.ai-chat-assistant-turn');
    const card = turn?.querySelector<HTMLElement>('.ai-chat-agent-card');
    if (!card) {
      setMessageStreaming(hasActiveWork);
      return;
    }

    const update = () => setMessageStreaming(card.dataset.state === 'streaming');
    update();
    const observer = new MutationObserver(update);
    observer.observe(card, { attributes: true, attributeFilter: ['data-state'] });
    return () => observer.disconnect();
  }, [hasActiveWork]);

  const terminal = !messageStreaming;

  useEffect(() => {
    if (!terminal) {
      setExpanded(true);
      wasTerminalRef.current = false;
      return;
    }
    if (!wasTerminalRef.current) setExpanded(hasFailure);
    wasTerminalRef.current = true;
  }, [hasFailure, terminal]);

  if (!rows.length && !showIdle) return null;

  const toggleLabel = expanded ? 'Hide tasks' : 'Show tasks';
  const copyDiagnostics = async () => {
    if (!diagnostics?.length || !navigator.clipboard?.writeText) return;
    await navigator.clipboard.writeText(['ARC AI task diagnostics', ...diagnostics].join('\n'));
    setDiagnosticsCopied(true);
    window.setTimeout(() => setDiagnosticsCopied(false), 1200);
  };

  return (
    <div
      className={`ai-chat-task-progress-shell${terminal ? ' is-terminal' : ''}${expanded ? ' is-expanded' : ''}`}
      ref={shellRef}
    >
      {terminal ? (
        <button
          aria-label={toggleLabel}
          className="ui-agent-card-action-button ai-chat-task-list-toggle"
          title={toggleLabel}
          type="button"
          onClick={() => setExpanded((current) => !current)}
        >
          <ListChecks aria-hidden="true" size={15} />
        </button>
      ) : null}
      {expanded || !terminal ? (
        <div className="ai-chat-live-progress" aria-live="polite" aria-label="AI progress" role="status">
          {visibleRows.map((row) => (
            <div className={`ai-chat-live-progress-row is-${row.state}`} data-progress-state={row.state} key={row.id}>
              <span className="ai-chat-live-progress-indicator" aria-hidden="true">
                {row.state === 'failed' ? <X size={11} strokeWidth={2.4} /> : null}
              </span>
              <span className="ai-chat-live-progress-label" key={`${row.id}:${row.title}`}>
                {row.title}
              </span>
              {row.state === 'waiting' ? <span className="ai-chat-live-progress-state">Waiting</span> : null}
            </div>
          ))}
        </div>
      ) : null}
      {diagnostics?.length ? (
        <button
          aria-label={diagnosticsCopied ? 'Diagnostics copied' : 'Copy task diagnostics'}
          className="ai-chat-task-diagnostics-button"
          title={diagnosticsCopied ? 'Diagnostics copied' : 'Copy task diagnostics'}
          type="button"
          onClick={() => void copyDiagnostics()}
        >
          {diagnosticsCopied ? <Check aria-hidden="true" size={12} /> : <Copy aria-hidden="true" size={12} />}
          <span>{diagnosticsCopied ? 'Copied' : 'Copy diagnostics'}</span>
        </button>
      ) : null}
    </div>
  );
}

import { useState } from 'react';

import type { AiConversationToolReference } from '../../../common/aiConversationTypes';
import { UiAgentAssetCard, UiAgentToolCard, UiAgentViewportCard, type UiAgentActivityState } from '../ui';

const detailCharacterLimit = 1800;

const boundedText = (value: string): string =>
  value.length <= detailCharacterLimit ? value : `${value.slice(0, detailCharacterLimit)}\n…`;

const formattedResult = (value: string): string => {
  try {
    return JSON.stringify(JSON.parse(value) as unknown, null, 2);
  } catch {
    return value;
  }
};

const activityStateFor = (reference: AiConversationToolReference): UiAgentActivityState => {
  if (reference.state === 'pending') return 'running';
  if (reference.state === 'error') return 'error';
  if (reference.state === 'cancelled') return 'cancelled';
  return 'complete';
};

const activitySummaryFor = (reference: AiConversationToolReference): string => {
  if (reference.summary?.trim()) return reference.summary.trim();
  if (reference.state === 'pending') return 'ARC is running this editor operation.';
  if (reference.state === 'error') return 'The editor operation did not complete.';
  if (reference.state === 'cancelled') return 'The editor operation was cancelled.';
  return 'The editor operation completed.';
};

const durationLabel = (reference: AiConversationToolReference): string | undefined => {
  if (!reference.startedAt || !reference.completedAt) return undefined;
  const started = Date.parse(reference.startedAt);
  const completed = Date.parse(reference.completedAt);
  if (!Number.isFinite(started) || !Number.isFinite(completed) || completed < started) return undefined;
  const durationMs = completed - started;
  if (durationMs < 1000) return `${durationMs} ms`;
  if (durationMs < 10_000) return `${(durationMs / 1000).toFixed(1)} s`;
  return `${Math.round(durationMs / 1000)} s`;
};

function ToolDetails({ reference }: { reference: AiConversationToolReference }) {
  const [copied, setCopied] = useState(false);
  const argumentsText =
    reference.arguments && Object.keys(reference.arguments).length ? JSON.stringify(reference.arguments, null, 2) : null;
  const resultText = reference.resultContent ? formattedResult(reference.resultContent) : null;

  const copyResult = async () => {
    if (!resultText || !navigator.clipboard?.writeText) return;
    await navigator.clipboard.writeText(resultText);
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1200);
  };

  if (!argumentsText && !resultText && !reference.resultTruncated && !reference.errorCode) return undefined;

  return (
    <div className="ai-chat-tool-details">
      {argumentsText ? (
        <section className="ai-chat-tool-detail-section">
          <div className="ai-chat-tool-detail-label">Arguments</div>
          <pre>{boundedText(argumentsText)}</pre>
        </section>
      ) : null}
      {resultText ? (
        <section className="ai-chat-tool-detail-section">
          <div className="ai-chat-tool-detail-label">Result</div>
          <pre>{boundedText(resultText)}</pre>
          <button className="ai-chat-tool-copy-json" type="button" onClick={() => void copyResult()}>
            {copied ? 'Copied' : 'Copy JSON'}
          </button>
        </section>
      ) : null}
      {reference.resultTruncated ? (
        <div className="ai-chat-tool-detail-note">
          Result was truncated{reference.originalBytes ? ` from ${reference.originalBytes.toLocaleString()} bytes` : ''}.
        </div>
      ) : null}
      {reference.errorCode ? (
        <section className="ai-chat-tool-detail-section">
          <div className="ai-chat-tool-detail-label">Error code</div>
          <pre>{`${reference.errorCode}${reference.retryable ? ' (retryable)' : ''}`}</pre>
        </section>
      ) : null}
    </div>
  );
}

export function AiChatToolActivityCard({ reference }: { reference: AiConversationToolReference }) {
  const state = activityStateFor(reference);
  const details = <ToolDetails reference={reference} />;
  const duration = durationLabel(reference);
  const metadata = [reference.step !== undefined ? `Step ${reference.step + 1}` : undefined, duration]
    .filter(Boolean)
    .join(' · ');
  const shared = {
    title: reference.name,
    subtitle: reference.operation && reference.operation !== reference.name ? reference.operation : undefined,
    summary: activitySummaryFor(reference),
    metadata: metadata || undefined,
    details,
    state,
    defaultExpanded: state === 'error',
  } as const;

  if (reference.name.startsWith('assets.')) return <UiAgentAssetCard {...shared} />;
  if (reference.name.startsWith('viewport.')) return <UiAgentViewportCard {...shared} />;
  return <UiAgentToolCard {...shared} />;
}

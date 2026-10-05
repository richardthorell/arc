import { Check, X } from 'lucide-react';

import { UiButton } from '../ui';
import type { AiAgentApprovalRequest } from './aiAgentApproval';
import './aiChatApprovalStatus.css';

type AiChatApprovalPromptProps = {
  request: AiAgentApprovalRequest;
  busy?: boolean;
  canApprove: boolean;
  canDeny: boolean;
  onApprove: () => void;
  onDeny: () => void;
};

export function AiChatApprovalPrompt({
  request,
  busy = false,
  canApprove,
  canDeny,
  onApprove,
  onDeny,
}: AiChatApprovalPromptProps) {
  return (
    <div className="ai-chat-approval-prompt" aria-label="AI editor action approval" role="alertdialog">
      <div className="ai-chat-approval-prompt-main">
        <span className="ai-chat-approval-spinner" aria-hidden="true" />
        <div className="ai-chat-approval-copy">
          <strong>{busy ? 'Applying approval…' : 'Awaiting approval'}</strong>
          <span>{request.label}</span>
        </div>
      </div>
      <div className="ai-chat-approval-actions">
        <UiButton disabled={busy || !canDeny} type="button" variant="ghost" onClick={onDeny}>
          <X size={12} /> Deny
        </UiButton>
        <UiButton disabled={busy || !canApprove} type="button" variant="primary" onClick={onApprove}>
          <Check size={12} /> Allow
        </UiButton>
      </div>
    </div>
  );
}

export function AiChatApprovalDeclined({ label }: { label: string }) {
  return (
    <div className="ai-chat-approval-declined" role="status">
      <strong>Declined</strong>
      <span>{label}</span>
    </div>
  );
}

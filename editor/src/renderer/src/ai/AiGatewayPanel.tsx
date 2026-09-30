import { Check, ShieldCheck, X } from 'lucide-react';
import type { ArcAiGatewayStatus } from '../../../preload/preload';
import { UiButton } from '../ui';
import { AiChatPanel } from './AiChatPanel';
import type { AiModelProvider } from './aiChat';
import './aiGateway.css';

// Compatibility wrapper for existing workbench call sites. The drawer now exposes chat only;
// gateway status and administration stay outside the panel.
export function AiGatewayPanel({
  provider,
}: {
  status: ArcAiGatewayStatus | null;
  onApprove: (requestId: string) => void;
  onDeny: (requestId: string) => void;
  onRevoke: (clientId: string) => void;
  onCancelEdit: (sessionId: string, clientId: string) => void;
  onUndoLastEdit: () => void;
  provider?: AiModelProvider;
}) {
  return <AiChatPanel provider={provider} />;
}

export function AiGatewayApprovalPrompt({
  status,
  onApprove,
  onDeny,
  onOpenGateway,
}: {
  status: ArcAiGatewayStatus | null;
  onApprove: (requestId: string) => void;
  onDeny: (requestId: string) => void;
  onOpenGateway: () => void;
}) {
  const request = status?.pendingEditRequests[0];
  if (!request) return null;
  return (
    <aside className="ai-gateway-approval-prompt" role="alertdialog" aria-label="AI editor action approval">
      <span>
        <ShieldCheck size={18} />
      </span>
      <div>
        <strong>{request.clientName} requests editor action access</strong>
        <small>{request.label} · applies only on commit · expires after 15 minutes of inactivity</small>
      </div>
      <UiButton onClick={() => onApprove(request.id)} variant="primary">
        <Check size={13} /> Allow
      </UiButton>
      <UiButton onClick={() => onDeny(request.id)} variant="ghost">
        <X size={13} /> Deny
      </UiButton>
      <UiButton onClick={onOpenGateway} variant="ghost">
        Open chat
      </UiButton>
    </aside>
  );
}

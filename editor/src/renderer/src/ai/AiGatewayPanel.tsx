import { Check, ShieldCheck, X } from 'lucide-react';
import { useEffect, useState } from 'react';
import type { EditorSettingsSnapshot } from '../../../common/editorWorkflowTypes';
import type { ArcAiGatewayStatus } from '../../../preload/preload';
import { UiButton } from '../ui';
import { AiChatPanel } from './AiChatPanel';
import type { AiModelProvider } from './aiChat';
import { runtimeAiProvidersFromSettings } from './runtimeAiProviders';
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
  const [runtimeProviders, setRuntimeProviders] = useState<AiModelProvider[]>([]);
  const [projectGuid, setProjectGuid] = useState<string | null>(null);

  useEffect(() => {
    let disposed = false;
    let refreshTimer: ReturnType<typeof setTimeout> | null = null;

    const refreshProject = async () => {
      try {
        const snapshot = await window.arc.projects.snapshot();
        if (!disposed) setProjectGuid(snapshot?.activeProject?.descriptor.guid ?? null);
      } catch {
        if (!disposed) setProjectGuid(null);
      }
    };
    const scheduleRefresh = () => {
      if (refreshTimer) clearTimeout(refreshTimer);
      refreshTimer = setTimeout(() => void refreshProject(), 0);
    };

    void refreshProject();
    const unsubscribe =
      window.arc.host?.onEvent((event) => {
        if (event.type === 'project.opened' || event.type === 'project.closed') scheduleRefresh();
      }) ?? (() => undefined);
    return () => {
      disposed = true;
      if (refreshTimer) clearTimeout(refreshTimer);
      unsubscribe();
    };
  }, []);

  useEffect(() => {
    if (provider) return;
    let disposed = false;

    const applySnapshot = (snapshot: EditorSettingsSnapshot | null | undefined) => {
      if (!disposed) setRuntimeProviders(runtimeAiProvidersFromSettings(snapshot));
    };
    const refresh = async () => {
      try {
        applySnapshot(await window.arc.settings.snapshot());
      } catch {
        applySnapshot(null);
      }
    };
    const onSettingsChanged = (event: Event) => {
      const snapshot = (event as CustomEvent<EditorSettingsSnapshot | null>).detail;
      if (snapshot) applySnapshot(snapshot);
      else void refresh();
    };
    const onSettingsClosed = () => void refresh();

    void refresh();
    window.addEventListener('arc-editor-settings-changed', onSettingsChanged);
    window.addEventListener('arc-editor-settings-closed', onSettingsClosed);
    return () => {
      disposed = true;
      window.removeEventListener('arc-editor-settings-changed', onSettingsChanged);
      window.removeEventListener('arc-editor-settings-closed', onSettingsClosed);
    };
  }, [provider]);

  return provider ? (
    <AiChatPanel key={projectGuid ?? 'no-project'} projectGuid={projectGuid ?? undefined} provider={provider} />
  ) : (
    <AiChatPanel
      key={projectGuid ?? 'no-project'}
      projectGuid={projectGuid ?? undefined}
      providers={runtimeProviders}
    />
  );
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

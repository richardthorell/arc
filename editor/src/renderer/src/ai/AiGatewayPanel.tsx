import { Bot, Check, Copy, RotateCcw, ShieldCheck, ShieldX, Unplug, X } from 'lucide-react';
import { useState } from 'react';
import type { ArcAiGatewayStatus } from '../../../preload/preload';
import { UiButton, UiIconButton } from '../ui';
import './aiGateway.css';

export function AiGatewayPanel({
  status,
  onApprove,
  onDeny,
  onRevoke,
  onCancelEdit,
  onUndoLastEdit,
}: {
  status: ArcAiGatewayStatus | null;
  onApprove: (requestId: string) => void;
  onDeny: (requestId: string) => void;
  onRevoke: (clientId: string) => void;
  onCancelEdit: (sessionId: string, clientId: string) => void;
  onUndoLastEdit: () => void;
}) {
  const [copied, setCopied] = useState('');
  const copy = async (label: string, text: string) => {
    await navigator.clipboard.writeText(text);
    setCopied(label);
    window.setTimeout(() => setCopied(''), 1200);
  };
  if (!status) {
    return (
      <div className="ai-gateway-empty">
        <Bot size={24} />
        <strong>AI Gateway starting…</strong>
      </div>
    );
  }
  return (
    <section className="ai-gateway-panel" aria-label="AI Gateway">
      <header className="ai-gateway-header">
        <span className={status.enabled ? 'online' : 'offline'}>
          <Bot size={17} />
        </span>
        <div>
          <strong>AI Scene Gateway</strong>
          <small>{status.enabled ? 'Localhost · authenticated' : 'Unavailable'}</small>
        </div>
        <span className="ai-gateway-protocol">v{status.protocolVersion}</span>
      </header>

      <div className="ai-gateway-connections">
        <div>
          <span>Endpoint</span>
          <code>{status.endpoint || 'Not listening'}</code>
          <UiIconButton label="Copy endpoint" onClick={() => void copy('endpoint', status.endpoint)}>
            {copied === 'endpoint' ? <Check size={13} /> : <Copy size={13} />}
          </UiIconButton>
        </div>
        <div>
          <span>Discovery</span>
          <code title={status.discoveryFile}>{status.discoveryFile || 'Unavailable'}</code>
          <UiIconButton label="Copy discovery path" onClick={() => void copy('discovery', status.discoveryFile)}>
            {copied === 'discovery' ? <Check size={13} /> : <Copy size={13} />}
          </UiIconButton>
        </div>
        <div>
          <span>MCP HTTP</span>
          <code>{status.endpoint}/mcp</code>
          <UiIconButton label="Copy MCP endpoint" onClick={() => void copy('mcp', `${status.endpoint}/mcp`)}>
            {copied === 'mcp' ? <Check size={13} /> : <Copy size={13} />}
          </UiIconButton>
        </div>
        <div>
          <span>OpenAPI</span>
          <code>{status.endpoint}/openapi.json</code>
          <UiIconButton
            label="Copy OpenAPI endpoint"
            onClick={() => void copy('openapi', `${status.endpoint}/openapi.json`)}
          >
            {copied === 'openapi' ? <Check size={13} /> : <Copy size={13} />}
          </UiIconButton>
        </div>
        <div>
          <span>MCP stdio</span>
          <code>arc-mcp --discovery &quot;{status.discoveryFile}&quot;</code>
          <UiIconButton
            label="Copy MCP stdio command"
            onClick={() => void copy('stdio', `arc-mcp --discovery "${status.discoveryFile}"`)}
          >
            {copied === 'stdio' ? <Check size={13} /> : <Copy size={13} />}
          </UiIconButton>
        </div>
        <div>
          <span>Revisions</span>
          <code>
            scene {status.sceneRevision} / world {status.worldEpoch} / frame {status.frameRevision}
          </code>
        </div>
      </div>

      {status.pendingEditRequests.length > 0 && (
        <div className="ai-gateway-section pending">
          <h4>
            <ShieldCheck size={14} /> Edit access requests
          </h4>
          {status.pendingEditRequests.map((request) => (
            <article key={request.id}>
              <div>
                <strong>{request.clientName}</strong>
                <small>{request.label}</small>
              </div>
              <UiButton onClick={() => onApprove(request.id)} variant="primary">
                <Check size={13} /> Allow 15 min
              </UiButton>
              <UiButton onClick={() => onDeny(request.id)} variant="ghost">
                <X size={13} /> Deny
              </UiButton>
            </article>
          ))}
        </div>
      )}

      <div className="ai-gateway-section">
        <h4>
          <ShieldCheck size={14} /> Connected clients <span>{status.clients.length}</span>
        </h4>
        {status.clients.length === 0 ? (
          <p className="ai-gateway-muted">No model clients are connected.</p>
        ) : (
          status.clients.map((client) => (
            <article key={client.id}>
              <div>
                <strong>{client.name}</strong>
                <small>
                  {client.id} · {new Date(client.lastSeenAt).toLocaleTimeString()}
                </small>
              </div>
              <UiIconButton label={`Revoke ${client.name}`} onClick={() => onRevoke(client.id)}>
                <Unplug size={14} />
              </UiIconButton>
            </article>
          ))
        )}
      </div>

      {status.activeEditSession && (
        <div className="ai-gateway-section active-edit">
          <h4>
            <ShieldCheck size={14} /> Active in-memory edit
          </h4>
          <article>
            <div>
              <strong>{status.activeEditSession.label}</strong>
              <small>Revision {status.activeEditSession.expectedSceneRevision} · never saved automatically</small>
            </div>
            <UiButton
              onClick={() => onCancelEdit(status.activeEditSession!.id, status.activeEditSession!.clientId)}
              variant="danger"
            >
              <ShieldX size={13} /> Cancel
            </UiButton>
          </article>
        </div>
      )}

      {status.lastCommittedEdit && (
        <div className="ai-gateway-section">
          <h4>
            <RotateCcw size={14} /> Last committed AI edit
          </h4>
          <article>
            <div>
              <strong>{status.lastCommittedEdit.label}</strong>
              <small>Revision {status.lastCommittedEdit.sceneRevision} / undo is rejected if the scene changed</small>
            </div>
            <UiButton onClick={onUndoLastEdit} variant="ghost">
              <RotateCcw size={13} /> Undo
            </UiButton>
          </article>
        </div>
      )}

      {status.viewportLease && (
        <div className="ai-gateway-section">
          <h4>
            <ShieldCheck size={14} /> Viewport control
          </h4>
          <p className="ai-gateway-muted">
            {status.viewportLease.clientId} until {new Date(status.viewportLease.expiresAt).toLocaleTimeString()}
          </p>
        </div>
      )}

      <div className="ai-gateway-section audit">
        <h4>Audit timeline</h4>
        <div className="ai-gateway-audit-list">
          {status.audit.length === 0 && <p className="ai-gateway-muted">Gateway activity will appear here.</p>}
          {[...status.audit].reverse().map((entry) => (
            <div className={entry.succeeded ? '' : 'failed'} key={entry.sequence}>
              <time>{new Date(entry.timestamp).toLocaleTimeString()}</time>
              <code>{entry.operation}</code>
              <span>{entry.clientId}</span>
              {entry.detail && <small>{entry.detail}</small>}
            </div>
          ))}
        </div>
      </div>
    </section>
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
    <aside className="ai-gateway-approval-prompt" role="alertdialog" aria-label="AI scene edit approval">
      <span>
        <ShieldCheck size={18} />
      </span>
      <div>
        <strong>{request.clientName} requests scene edit access</strong>
        <small>{request.label} · in-memory only · expires after 15 minutes of inactivity</small>
      </div>
      <UiButton onClick={() => onApprove(request.id)} variant="primary">
        <Check size={13} /> Allow
      </UiButton>
      <UiButton onClick={() => onDeny(request.id)} variant="ghost">
        <X size={13} /> Deny
      </UiButton>
      <UiButton onClick={onOpenGateway} variant="ghost">
        Details
      </UiButton>
    </aside>
  );
}

import {
  Bot,
  Check,
  Copy,
  History,
  MessageSquarePlus,
  RotateCcw,
  Send,
  Settings2,
  ShieldCheck,
  ShieldX,
  Sparkles,
  Unplug,
  User,
  X,
} from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ArcAiGatewayStatus } from '../../../preload/preload';
import { UiButton, UiIconButton } from '../ui';
import {
  conversationTitleFromPrompt,
  createAiConversation,
  createAiMessage,
  loadAiConversations,
  saveAiConversations,
  unavailableAiModelProvider,
  type AiConversation,
  type AiModelProvider,
} from './aiChat';
import './aiGateway.css';

const replaceConversation = (
  conversations: readonly AiConversation[],
  next: AiConversation,
): AiConversation[] => {
  const existing = conversations.findIndex((conversation) => conversation.id === next.id);
  if (existing === -1) return [next, ...conversations];
  return conversations.map((conversation, index) => (index === existing ? next : conversation));
};

function AiGatewayDiagnostics({
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
    <div className="ai-gateway-diagnostics" aria-label="AI Gateway diagnostics">
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
    </div>
  );
}

export function AiGatewayPanel({
  status,
  onApprove,
  onDeny,
  onRevoke,
  onCancelEdit,
  onUndoLastEdit,
  provider = unavailableAiModelProvider,
}: {
  status: ArcAiGatewayStatus | null;
  onApprove: (requestId: string) => void;
  onDeny: (requestId: string) => void;
  onRevoke: (clientId: string) => void;
  onCancelEdit: (sessionId: string, clientId: string) => void;
  onUndoLastEdit: () => void;
  provider?: AiModelProvider;
}) {
  const initial = useMemo(() => loadAiConversations(), []);
  const [conversations, setConversations] = useState<AiConversation[]>(() =>
    initial.length > 0 ? initial : [createAiConversation()],
  );
  const [activeConversationId, setActiveConversationId] = useState(() =>
    initial[0]?.id ?? conversations[0]?.id ?? '',
  );
  const [prompt, setPrompt] = useState('');
  const [view, setView] = useState<'chat' | 'diagnostics'>('chat');
  const streamGeneration = useRef(0);
  const messageEnd = useRef<HTMLDivElement | null>(null);

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ?? conversations[0];
  const busy = activeConversation?.messages.some((message) => message.state === 'streaming') ?? false;

  useEffect(() => {
    saveAiConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    messageEnd.current?.scrollIntoView({ block: 'end' });
  }, [activeConversation?.messages]);

  const newChat = () => {
    ++streamGeneration.current;
    const next = createAiConversation();
    setConversations((current) => [next, ...current]);
    setActiveConversationId(next.id);
    setPrompt('');
    setView('chat');
  };

  const send = async () => {
    const text = prompt.trim();
    if (!text || !activeConversation || busy) return;

    const generation = ++streamGeneration.current;
    const userMessage = createAiMessage('user', text);
    const assistantMessage = createAiMessage('assistant', '', 'streaming');
    const started: AiConversation = {
      ...activeConversation,
      title:
        activeConversation.messages.length === 0 ? conversationTitleFromPrompt(text) : activeConversation.title,
      updatedAt: new Date().toISOString(),
      messages: [...activeConversation.messages, userMessage, assistantMessage],
    };
    setPrompt('');
    setConversations((current) => replaceConversation(current, started));

    let responseText = '';
    try {
      for await (const event of provider.stream({
        conversationId: started.id,
        messages: [...started.messages.slice(0, -1)],
      })) {
        if (generation !== streamGeneration.current) return;
        if (event.type === 'delta') responseText += event.text;
        if (event.type === 'error') throw new Error(event.message);
        const state = event.type === 'done' ? 'complete' : 'streaming';
        setConversations((current) => {
          const conversation = current.find((candidate) => candidate.id === started.id);
          if (!conversation) return current;
          return replaceConversation(current, {
            ...conversation,
            updatedAt: new Date().toISOString(),
            messages: conversation.messages.map((message) =>
              message.id === assistantMessage.id ? { ...message, content: responseText, state } : message,
            ),
          });
        });
      }
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      setConversations((current) => {
        const conversation = current.find((candidate) => candidate.id === started.id);
        if (!conversation) return current;
        return replaceConversation(current, {
          ...conversation,
          updatedAt: new Date().toISOString(),
          messages: conversation.messages.map((entry) =>
            entry.id === assistantMessage.id
              ? { ...entry, content: responseText || message, state: 'error' as const }
              : entry,
          ),
        });
      });
    }
  };

  return (
    <section className="ai-assistant-panel" aria-label="ARC Assistant">
      <header className="ai-assistant-header">
        <span className={status?.enabled ? 'online' : 'offline'}>
          <Sparkles size={16} />
        </span>
        <div className="ai-assistant-heading">
          <strong>ARC Assistant</strong>
          <small>{view === 'chat' ? `${provider.label}${provider.configured ? '' : ' · setup required'}` : 'Gateway diagnostics'}</small>
        </div>
        <UiIconButton label="New AI chat" onClick={newChat}>
          <MessageSquarePlus size={15} />
        </UiIconButton>
        <UiIconButton
          active={view === 'diagnostics'}
          label={view === 'diagnostics' ? 'Back to chat' : 'Gateway diagnostics'}
          onClick={() => setView((current) => (current === 'chat' ? 'diagnostics' : 'chat'))}
        >
          {view === 'diagnostics' ? <History size={15} /> : <Settings2 size={15} />}
        </UiIconButton>
      </header>

      {view === 'diagnostics' ? (
        <AiGatewayDiagnostics
          status={status}
          onApprove={onApprove}
          onDeny={onDeny}
          onRevoke={onRevoke}
          onCancelEdit={onCancelEdit}
          onUndoLastEdit={onUndoLastEdit}
        />
      ) : (
        <>
          <div className="ai-assistant-conversation-bar">
            <select
              aria-label="AI conversation"
              value={activeConversation?.id ?? ''}
              onChange={(event) => setActiveConversationId(event.target.value)}
            >
              {conversations.map((conversation) => (
                <option key={conversation.id} value={conversation.id}>
                  {conversation.title}
                </option>
              ))}
            </select>
            <span>{activeConversation?.messages.length ?? 0} messages</span>
          </div>

          <div className="ai-assistant-messages" aria-live="polite">
            {(!activeConversation || activeConversation.messages.length === 0) && (
              <div className="ai-assistant-empty">
                <span>
                  <Bot size={22} />
                </span>
                <strong>Ask ARC about your project</strong>
                <p>
                  Stage 1 establishes conversations, streaming responses, history, and the provider boundary. Scene and
                  viewport context arrive in the next stage.
                </p>
              </div>
            )}
            {activeConversation?.messages.map((message) => (
              <article className={`ai-message ${message.role} ${message.state}`} key={message.id}>
                <span className="ai-message-avatar" aria-hidden="true">
                  {message.role === 'user' ? <User size={14} /> : <Bot size={14} />}
                </span>
                <div>
                  <strong>{message.role === 'user' ? 'You' : 'ARC'}</strong>
                  <p>{message.content || (message.state === 'streaming' ? 'Thinking…' : '')}</p>
                  {message.state === 'error' && <small>Response failed</small>}
                </div>
              </article>
            ))}
            <div ref={messageEnd} />
          </div>

          <form
            className="ai-assistant-composer"
            onSubmit={(event) => {
              event.preventDefault();
              void send();
            }}
          >
            <textarea
              aria-label="Ask ARC"
              placeholder="Ask ARC..."
              value={prompt}
              rows={2}
              onChange={(event) => setPrompt(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === 'Enter' && !event.shiftKey) {
                  event.preventDefault();
                  void send();
                }
              }}
            />
            <UiIconButton label="Send prompt" disabled={!prompt.trim() || busy} type="submit">
              <Send size={15} />
            </UiIconButton>
            <small>Enter to send · Shift+Enter for a new line</small>
          </form>
        </>
      )}
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

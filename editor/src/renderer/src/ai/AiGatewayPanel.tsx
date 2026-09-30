import { Bot, Check, MessageSquarePlus, Send, ShieldCheck, Sparkles, User, X } from 'lucide-react';
import { useEffect, useMemo, useRef, useState } from 'react';
import type { ArcAiGatewayStatus } from '../../../preload/preload';
import { UiButton, UiDrawerPanel, UiIconButton } from '../ui';
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

const replaceConversation = (conversations: readonly AiConversation[], next: AiConversation): AiConversation[] => {
  const existing = conversations.findIndex((conversation) => conversation.id === next.id);
  if (existing === -1) return [next, ...conversations];
  return conversations.map((conversation, index) => (index === existing ? next : conversation));
};

export function AiChatPanel({ provider = unavailableAiModelProvider }: { provider?: AiModelProvider }) {
  const initial = useMemo(() => loadAiConversations(), []);
  const [conversations, setConversations] = useState<AiConversation[]>(() =>
    initial.length > 0 ? initial : [createAiConversation()],
  );
  const [activeConversationId, setActiveConversationId] = useState(() => initial[0]?.id ?? conversations[0]?.id ?? '');
  const [prompt, setPrompt] = useState('');
  const streamGeneration = useRef(0);
  const messageEnd = useRef<HTMLDivElement | null>(null);

  const activeConversation =
    conversations.find((conversation) => conversation.id === activeConversationId) ?? conversations[0];
  const busy = activeConversation?.messages.some((message) => message.state === 'streaming') ?? false;

  useEffect(() => {
    saveAiConversations(conversations);
  }, [conversations]);

  useEffect(() => {
    const scrollIntoView = messageEnd.current?.scrollIntoView;
    if (typeof scrollIntoView === 'function') {
      scrollIntoView.call(messageEnd.current, { block: 'end' });
    }
  }, [activeConversation?.messages]);

  const newChat = () => {
    ++streamGeneration.current;
    const next = createAiConversation();
    setConversations((current) => [next, ...current]);
    setActiveConversationId(next.id);
    setPrompt('');
  };

  const send = async () => {
    const text = prompt.trim();
    if (!text || !activeConversation || busy) return;

    const generation = ++streamGeneration.current;
    const userMessage = createAiMessage('user', text);
    const assistantMessage = createAiMessage('assistant', '', 'streaming');
    const started: AiConversation = {
      ...activeConversation,
      title: activeConversation.messages.length === 0 ? conversationTitleFromPrompt(text) : activeConversation.title,
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
    <UiDrawerPanel className="ai-assistant-panel" aria-label="AI Chat">
      <header className="ai-assistant-header">
        <span>
          <Sparkles size={16} />
        </span>
        <div className="ai-assistant-heading">
          <strong>AI Chat</strong>
          <small>{`${provider.label}${provider.configured ? '' : ' · setup required'}`}</small>
        </div>
        <UiIconButton label="New AI chat" onClick={newChat}>
          <MessageSquarePlus size={15} />
        </UiIconButton>
      </header>

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
    </UiDrawerPanel>
  );
}

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

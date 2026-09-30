import { ArrowLeft, Send } from 'lucide-react';
import { useEffect, useMemo, useState } from 'react';
import { requestSettingsDialogOpen } from '../settings/settingsDialogRoute';
import { UiAgentTextCard, UiButton, UiDrawerPanel, UiIconButton } from '../ui';
import {
  conversationTitleFromPrompt,
  createAiConversation,
  createAiMessage,
  loadAiConversations,
  saveAiConversations,
  unavailableAiModelProvider,
  type AiChatMessage,
  type AiConversation,
  type AiModelProvider,
} from './aiChat';
import './aiGateway.css';

const openAiConnectivitySettings = () => requestSettingsDialogOpen('editorPreferences', 'ai.providers');

const cloneConversation = (conversation: AiConversation): AiConversation => ({
  ...conversation,
  messages: conversation.messages.map((message) => ({ ...message })),
});

type AiChatPanelProps = {
  provider?: AiModelProvider;
  providers?: readonly AiModelProvider[];
  initialConversations?: readonly AiConversation[];
  initialMessages?: readonly AiChatMessage[];
  conversationLabel?: string;
  persistConversations?: boolean;
};

export function AiChatPanel({
  provider,
  providers,
  initialConversations,
  initialMessages,
  conversationLabel = 'New conversation',
  persistConversations,
}: AiChatPanelProps) {
  const configuredProviders = useMemo(
    () => (providers ?? [provider ?? unavailableAiModelProvider]).filter((candidate) => candidate.configured),
    [provider, providers],
  );
  const connected = configuredProviders.length > 0;
  const shouldPersistConversations =
    persistConversations ?? (initialConversations === undefined && initialMessages === undefined);
  const [conversations, setConversations] = useState<AiConversation[]>(() => {
    if (initialConversations) return initialConversations.map(cloneConversation);
    if (initialMessages?.length) {
      const firstTimestamp = initialMessages[0]?.createdAt ?? new Date().toISOString();
      const lastTimestamp = initialMessages[initialMessages.length - 1]?.createdAt ?? firstTimestamp;
      return [
        {
          id: 'seeded-conversation',
          title: conversationLabel,
          createdAt: firstTimestamp,
          updatedAt: lastTimestamp,
          messages: initialMessages.map((message) => ({ ...message })),
          modelId: configuredProviders[0]?.id,
          modelLabel: configuredProviders[0]?.label,
        },
      ];
    }
    return loadAiConversations();
  });
  const [activeConversationId, setActiveConversationId] = useState<string | null>(null);
  const [selectedModelId, setSelectedModelId] = useState(() => configuredProviders[0]?.id ?? '');
  const [prompt, setPrompt] = useState('');
  const [streaming, setStreaming] = useState(false);

  useEffect(() => {
    if (!configuredProviders.length) {
      setSelectedModelId('');
      return;
    }
    if (!configuredProviders.some((candidate) => candidate.id === selectedModelId)) {
      setSelectedModelId(configuredProviders[0].id);
    }
  }, [configuredProviders, selectedModelId]);

  useEffect(() => {
    if (shouldPersistConversations) saveAiConversations(conversations);
  }, [conversations, shouldPersistConversations]);

  const activeConversation = conversations.find((conversation) => conversation.id === activeConversationId) ?? null;
  const activeProvider = activeConversation
    ? configuredProviders.find((candidate) => candidate.id === activeConversation.modelId) ?? null
    : null;
  const recentConversations = useMemo(
    () =>
      conversations
        .filter((conversation) => conversation.messages.length > 0)
        .sort((left, right) => right.updatedAt.localeCompare(left.updatedAt)),
    [conversations],
  );

  const updateAssistantMessage = (
    conversationId: string,
    messageId: string,
    update: (message: AiChatMessage) => AiChatMessage,
  ) => {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id !== conversationId
          ? conversation
          : {
              ...conversation,
              updatedAt: new Date().toISOString(),
              messages: conversation.messages.map((message) => (message.id === messageId ? update(message) : message)),
            },
      ),
    );
  };

  const streamResponse = async (
    conversationId: string,
    model: AiModelProvider,
    requestMessages: AiChatMessage[],
    assistantMessage: AiChatMessage,
  ) => {
    setStreaming(true);
    try {
      let completed = false;
      for await (const event of model.stream({ conversationId, messages: requestMessages })) {
        if (event.type === 'delta') {
          updateAssistantMessage(conversationId, assistantMessage.id, (message) => ({
            ...message,
            content: `${message.content}${event.text}`,
          }));
          continue;
        }
        if (event.type === 'error') {
          updateAssistantMessage(conversationId, assistantMessage.id, (message) => ({
            ...message,
            content: message.content ? `${message.content}\n\n${event.message}` : event.message,
            state: 'error',
          }));
          return;
        }
        completed = true;
        updateAssistantMessage(conversationId, assistantMessage.id, (message) => ({ ...message, state: 'complete' }));
      }

      if (!completed) {
        updateAssistantMessage(conversationId, assistantMessage.id, (message) => ({ ...message, state: 'complete' }));
      }
    } catch (error) {
      updateAssistantMessage(conversationId, assistantMessage.id, (message) => ({
        ...message,
        content: message.content || (error instanceof Error ? error.message : String(error)),
        state: 'error',
      }));
    } finally {
      setStreaming(false);
    }
  };

  const startConversation = async () => {
    const content = prompt.trim();
    const model = configuredProviders.find((candidate) => candidate.id === selectedModelId);
    if (!model || streaming || !content) return;

    const conversation = createAiConversation();
    const userMessage = createAiMessage('user', content);
    const assistantMessage = createAiMessage('assistant', '', 'streaming');
    conversation.title = conversationTitleFromPrompt(content);
    conversation.modelId = model.id;
    conversation.modelLabel = model.label;
    conversation.messages = [userMessage, assistantMessage];

    setPrompt('');
    setConversations((current) => [conversation, ...current]);
    setActiveConversationId(conversation.id);
    await streamResponse(conversation.id, model, [userMessage], assistantMessage);
  };

  const sendPrompt = async () => {
    const content = prompt.trim();
    if (!activeConversation || !activeProvider || streaming || !content) return;

    const userMessage = createAiMessage('user', content);
    const assistantMessage = createAiMessage('assistant', '', 'streaming');
    const requestMessages = [...activeConversation.messages, userMessage];

    setPrompt('');
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === activeConversation.id
          ? {
              ...conversation,
              updatedAt: new Date().toISOString(),
              messages: [...conversation.messages, userMessage, assistantMessage],
            }
          : conversation,
      ),
    );
    await streamResponse(activeConversation.id, activeProvider, requestMessages, assistantMessage);
  };

  const renderMessages = (conversation: AiConversation) => (
    <div className="ai-chat-message-list">
      {conversation.messages.map((message) => {
        if (message.role === 'assistant') {
          return (
            <UiAgentTextCard
              key={message.id}
              state={message.state}
              subtitle={conversation.modelLabel ?? activeProvider?.label}
              text={message.content}
              title="ARC"
            />
          );
        }
        if (message.role === 'user') {
          return (
            <div className="ai-chat-user-message" key={message.id}>
              {message.content}
            </div>
          );
        }
        return (
          <div className="ai-chat-system-message" key={message.id}>
            {message.content}
          </div>
        );
      })}
    </div>
  );

  return (
    <UiDrawerPanel className="ai-chat-panel" aria-label="AI Chat">
      {activeConversation ? (
        <section className="ai-chat-active" aria-label="Active conversation">
          <header className="ai-chat-active-header">
            <UiIconButton label="Back to conversations" onClick={() => setActiveConversationId(null)}>
              <ArrowLeft size={15} />
            </UiIconButton>
            <div>
              <strong>{activeConversation.title}</strong>
              <span>{activeConversation.modelLabel ?? activeProvider?.label ?? 'Model unavailable'}</span>
            </div>
          </header>

          <div className="ai-chat-history" aria-label="Chat history" aria-busy={streaming}>
            {renderMessages(activeConversation)}
          </div>

          <form
            className="ai-chat-composer ai-chat-active-composer"
            onSubmit={(event) => {
              event.preventDefault();
              void sendPrompt();
            }}
          >
            <div className="ai-chat-composer-surface">
              <textarea
                aria-label="Ask ARC"
                disabled={!activeProvider || streaming}
                placeholder={activeProvider ? 'Ask ARC...' : 'The model for this conversation is unavailable'}
                value={prompt}
                rows={3}
                onChange={(event) => setPrompt(event.target.value)}
              />
              <div className="ai-chat-composer-toolbar ai-chat-composer-toolbar-active">
                <span className="ai-chat-locked-model">{activeConversation.modelLabel ?? 'Model unavailable'}</span>
                <UiIconButton
                  label="Send prompt"
                  disabled={!activeProvider || streaming || !prompt.trim()}
                  type="submit"
                >
                  <Send size={15} />
                </UiIconButton>
              </div>
            </div>
          </form>
        </section>
      ) : (
        <section className="ai-chat-home" aria-label="Conversations">
          {connected && recentConversations.length > 0 && (
            <div className="ai-chat-recent" aria-label="Recent conversations">
              <span className="ai-chat-recent-label">Recent conversations</span>
              <div className="ai-chat-recent-list">
                {recentConversations.map((conversation) => (
                  <button
                    aria-label={`Open conversation ${conversation.title}`}
                    disabled={streaming}
                    key={conversation.id}
                    onClick={() => {
                      setPrompt('');
                      setActiveConversationId(conversation.id);
                    }}
                    type="button"
                  >
                    <span>{conversation.title}</span>
                    <small>
                      {conversation.modelLabel ?? 'AI'} · {conversation.messages.length}{' '}
                      {conversation.messages.length === 1 ? 'message' : 'messages'}
                    </small>
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="ai-chat-home-center">
            {!connected ? (
              <div className="ai-chat-connect-empty">
                <strong>Connect your AI service</strong>
                <span>Connect an AI provider in Editor Preferences to start a conversation.</span>
                <UiButton onClick={openAiConnectivitySettings} variant="primary">
                  Open AI settings
                </UiButton>
              </div>
            ) : (
              <form
                className="ai-chat-home-composer"
                aria-label="New conversation"
                onSubmit={(event) => {
                  event.preventDefault();
                  void startConversation();
                }}
              >
                <div className="ai-chat-composer-surface">
                  <textarea
                    aria-label="Start a conversation"
                    disabled={streaming}
                    placeholder="Ask ARC..."
                    value={prompt}
                    rows={3}
                    onChange={(event) => setPrompt(event.target.value)}
                  />
                  <div className="ai-chat-composer-toolbar">
                    <select
                      aria-label="Model"
                      disabled={streaming}
                      onChange={(event) => setSelectedModelId(event.target.value)}
                      value={selectedModelId}
                    >
                      {configuredProviders.map((candidate) => (
                        <option key={candidate.id} value={candidate.id}>
                          {candidate.label}
                        </option>
                      ))}
                    </select>
                    <UiIconButton label="Start conversation" disabled={streaming || !prompt.trim()} type="submit">
                      <Send size={15} />
                    </UiIconButton>
                  </div>
                </div>
              </form>
            )}
          </div>
        </section>
      )}
    </UiDrawerPanel>
  );
}

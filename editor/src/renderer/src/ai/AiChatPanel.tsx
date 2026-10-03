import { Asterisk, ArrowLeft, Bot, Plus, Send, Sparkles, Square } from 'lucide-react';
import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from 'react';
import type { AiConversationContextReference } from '../../../common/aiConversationTypes';
import { requestSettingsDialogOpen } from '../settings/settingsDialogRoute';
import { UiAgentTextCard, UiButton, UiDrawerPanel, UiDropdown, UiIconButton, type UiDropdownOption } from '../ui';
import {
  conversationCaptionFromResponse,
  conversationCaptionInstruction,
  conversationTitleFromPrompt,
  createAiConversation,
  createAiMessage,
  unavailableAiModelProvider,
  type AiChatMessage,
  type AiConversation,
  type AiModelProvider,
} from './aiChat';
import { renderAiChatMessageText } from './AiChatMessageText';
import { AiContextChips, AiContextPicker } from './AiContextPickerView';
import { appendAiContextAttachments } from './aiContextAttachments';
import {
  prepareAiContextBudget,
  type AiContextBudgetDiagnostics,
  type AiProjectContextSource,
} from './aiContextBudget';
import { createAiAssetContextProvider } from './aiContextPicker';
import { loadAiConversationStore, saveAiConversationStore } from './aiConversationStore';
import { createDefaultAiContextProviders, createWindowAiProjectContextService } from './aiProjectContextService';
import './aiGateway.css';
import './aiChatMessageCards.css';

const openAiConnectivitySettings = () => requestSettingsDialogOpen('editorPreferences', 'ai.providers');

const cloneConversation = (conversation: AiConversation): AiConversation => ({
  ...conversation,
  messages: conversation.messages.map((message) => ({ ...message })),
});

const modelIcon = (providerId: string) => {
  if (providerId.startsWith('openai:')) return <Sparkles aria-hidden="true" size={13} />;
  if (providerId.startsWith('anthropic:')) return <Asterisk aria-hidden="true" size={13} />;
  return <Bot aria-hidden="true" size={13} />;
};

const agentToneClass = (providerId: string | undefined) => {
  if (providerId?.startsWith('openai:')) return 'is-openai';
  if (providerId?.startsWith('anthropic:')) return 'is-anthropic';
  if (providerId?.includes('mock')) return 'is-mock';
  return 'is-generic';
};

const formatMessageTime = (createdAt: string) => {
  const timestamp = new Date(createdAt);
  if (Number.isNaN(timestamp.getTime())) return '';
  return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit' }).format(timestamp);
};

type ActiveStream = {
  controller: AbortController;
  conversationId: string;
  messageId: string;
};

type StreamResponseResult = {
  completed: boolean;
  text: string;
};

type AiChatPanelProps = {
  provider?: AiModelProvider;
  providers?: readonly AiModelProvider[];
  projectGuid?: string;
  initialConversations?: readonly AiConversation[];
  initialMessages?: readonly AiChatMessage[];
  conversationLabel?: string;
  persistConversations?: boolean;
  contextSource?: AiProjectContextSource | null;
  onContextBudget?: (diagnostics: AiContextBudgetDiagnostics) => void;
};

export function AiChatPanel({
  provider,
  providers,
  projectGuid,
  initialConversations,
  initialMessages,
  conversationLabel = 'New conversation',
  persistConversations,
  contextSource,
  onContextBudget,
}: AiChatPanelProps) {
  const configuredProviders = useMemo(
    () => (providers ?? [provider ?? unavailableAiModelProvider]).filter((candidate) => candidate.configured),
    [provider, providers],
  );
  const connected = configuredProviders.length > 0;
  const shouldPersistConversations =
    persistConversations ??
    (Boolean(projectGuid) && initialConversations === undefined && initialMessages === undefined);
  const persistedStore = useMemo(
    () => (shouldPersistConversations && projectGuid ? loadAiConversationStore(projectGuid) : null),
    [projectGuid, shouldPersistConversations],
  );
  const ownedContextService = useMemo(() => {
    if (contextSource !== undefined || !projectGuid || typeof window === 'undefined') return null;
    if (!window.arc?.projects?.snapshot || !window.arc?.host?.query) return null;
    return createWindowAiProjectContextService({
      providers: [...createDefaultAiContextProviders(), createAiAssetContextProvider()],
    });
  }, [contextSource, projectGuid]);
  const projectContextSource = contextSource === undefined ? ownedContextService : contextSource;
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
    return persistedStore?.conversations.map(cloneConversation) ?? [];
  });
  const [activeConversationId, setActiveConversationId] = useState<string | null>(
    () => persistedStore?.uiState.activeConversationId ?? null,
  );
  const [selectedModelId, setSelectedModelId] = useState(
    () => persistedStore?.uiState.selectedModelId ?? configuredProviders[0]?.id ?? '',
  );
  const [prompt, setPrompt] = useState('');
  const [streaming, setStreaming] = useState(false);
  const [contextPickerOpen, setContextPickerOpen] = useState(false);
  const [pendingContext, setPendingContext] = useState<AiConversationContextReference[]>([]);
  const activeStreamRef = useRef<ActiveStream | null>(null);
  const captionControllersRef = useRef(new Set<AbortController>());

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
    if (!shouldPersistConversations || !projectGuid) return;
    saveAiConversationStore(projectGuid, conversations, {
      ...(activeConversationId ? { activeConversationId } : {}),
      ...(selectedModelId ? { selectedModelId } : {}),
    });
  }, [activeConversationId, conversations, projectGuid, selectedModelId, shouldPersistConversations]);

  useEffect(() => () => ownedContextService?.dispose(), [ownedContextService]);

  useEffect(
    () => () => {
      activeStreamRef.current?.controller.abort();
      for (const controller of captionControllersRef.current) controller.abort();
      captionControllersRef.current.clear();
    },
    [],
  );

  const activeConversation = conversations.find((conversation) => conversation.id === activeConversationId) ?? null;
  const activeProvider = activeConversation
    ? (configuredProviders.find((candidate) => candidate.id === activeConversation.modelId) ?? null)
    : null;
  const selectedProvider = configuredProviders.find((candidate) => candidate.id === selectedModelId) ?? null;
  const composerProvider = activeConversation ? activeProvider : selectedProvider;
  const modelOptions = useMemo<ReadonlyArray<UiDropdownOption<string>>>(
    () =>
      configuredProviders.map((candidate) => ({
        value: candidate.id,
        label: candidate.label,
        icon: modelIcon(candidate.id),
      })),
    [configuredProviders],
  );
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

  const updateConversationSummary = (conversationId: string, summary: NonNullable<AiConversation['summary']>) => {
    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === conversationId &&
        (conversation.summary?.text !== summary.text ||
          conversation.summary?.throughMessageId !== summary.throughMessageId)
          ? { ...conversation, summary, updatedAt: new Date().toISOString() }
          : conversation,
      ),
    );
  };

  const stopResponse = () => {
    const activeStream = activeStreamRef.current;
    if (!activeStream) return;

    activeStreamRef.current = null;
    activeStream.controller.abort();
    updateAssistantMessage(activeStream.conversationId, activeStream.messageId, (message) => ({
      ...message,
      state: 'complete',
    }));
    setStreaming(false);
  };

  const streamResponse = async (
    conversationId: string,
    model: AiModelProvider,
    requestConversation: AiConversation,
    requestMessages: AiChatMessage[],
    assistantMessage: AiChatMessage,
  ): Promise<StreamResponseResult> => {
    const controller = new AbortController();
    activeStreamRef.current = {
      controller,
      conversationId,
      messageId: assistantMessage.id,
    };
    setStreaming(true);

    let responseText = '';
    let completed = false;
    try {
      const contextPlan = await prepareAiContextBudget({
        conversation: { ...requestConversation, messages: requestMessages },
        messages: requestMessages,
        modelCapabilities: model.capabilities,
        projectGuid,
        projectContextSource,
      });
      if (controller.signal.aborted) return { completed: false, text: responseText };
      if (contextPlan.summary) updateConversationSummary(conversationId, contextPlan.summary);
      onContextBudget?.(contextPlan.diagnostics);

      const currentReferences = new Map(
        (requestMessages[requestMessages.length - 1]?.contextReferences ?? []).map((reference) => [
          reference.id,
          reference,
        ]),
      );
      const rejectedCurrent = contextPlan.diagnostics.rejectedReferences.filter(
        (reference) => reference.origin === 'explicit' && currentReferences.has(reference.id),
      );
      if (rejectedCurrent.length) {
        const labels = rejectedCurrent.map((reference) => currentReferences.get(reference.id)?.label ?? reference.kind);
        throw new Error(
          `Attached context is stale or unavailable: ${labels.join(', ')}. Re-add the context and retry.`,
        );
      }

      const providerMessages = appendAiContextAttachments(contextPlan.messages, requestMessages, model.capabilities);
      for await (const event of model.stream({
        conversationId,
        messages: providerMessages,
        signal: controller.signal,
      })) {
        if (controller.signal.aborted) break;
        if (event.type === 'delta') {
          responseText = `${responseText}${event.text}`;
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
          return { completed: false, text: responseText };
        }
        if (event.type === 'done') {
          completed = true;
          updateAssistantMessage(conversationId, assistantMessage.id, (message) => ({ ...message, state: 'complete' }));
        }
      }

      if (!completed && !controller.signal.aborted) {
        completed = true;
        updateAssistantMessage(conversationId, assistantMessage.id, (message) => ({ ...message, state: 'complete' }));
      }
      return { completed: completed && !controller.signal.aborted, text: responseText };
    } catch (error) {
      if (!controller.signal.aborted) {
        updateAssistantMessage(conversationId, assistantMessage.id, (message) => ({
          ...message,
          content: message.content || (error instanceof Error ? error.message : String(error)),
          state: 'error',
        }));
      }
      return { completed: false, text: responseText };
    } finally {
      if (activeStreamRef.current?.controller === controller) {
        activeStreamRef.current = null;
        setStreaming(false);
      }
    }
  };

  const generateConversationCaption = async (
    conversationId: string,
    model: AiModelProvider,
    openingPrompt: string,
    openingResponse: string,
  ) => {
    const controller = new AbortController();
    captionControllersRef.current.add(controller);
    const fallbackTitle = conversationTitleFromPrompt(openingPrompt);
    const messages = [
      createAiMessage('system', conversationCaptionInstruction),
      createAiMessage('user', `Opening request:\n${openingPrompt}\n\nOpening response:\n${openingResponse}`),
    ];
    let generated = '';

    try {
      for await (const event of model.stream({
        conversationId: `${conversationId}:caption`,
        messages,
        metadata: { purpose: 'conversation-caption' },
        signal: controller.signal,
      })) {
        if (controller.signal.aborted) return;
        if (event.type === 'delta') generated = `${generated}${event.text}`;
        if (event.type === 'error') return;
        if (event.type === 'done') break;
      }
      const caption = conversationCaptionFromResponse(generated);
      if (!caption) return;
      setConversations((current) =>
        current.map((conversation) =>
          conversation.id === conversationId && conversation.title === fallbackTitle
            ? { ...conversation, title: caption }
            : conversation,
        ),
      );
    } catch {
      // The opening prompt remains a useful fallback if background caption generation fails.
    } finally {
      captionControllersRef.current.delete(controller);
    }
  };

  const messageWithPendingContext = (content: string): AiChatMessage => {
    const message = createAiMessage('user', content);
    return pendingContext.length
      ? { ...message, contextReferences: pendingContext.map((reference) => ({ ...reference })) }
      : message;
  };

  const startConversation = async () => {
    const content = prompt.trim();
    const model = configuredProviders.find((candidate) => candidate.id === selectedModelId);
    if (!model || streaming || !content) return;

    const conversation = createAiConversation();
    const userMessage = messageWithPendingContext(content);
    const assistantMessage: AiChatMessage = {
      ...createAiMessage('assistant', '', 'streaming'),
      modelId: model.id,
      modelLabel: model.label,
    };
    conversation.title = conversationTitleFromPrompt(content);
    conversation.modelId = model.id;
    conversation.modelLabel = model.label;
    conversation.messages = [userMessage, assistantMessage];

    setPrompt('');
    setContextPickerOpen(false);
    setConversations((current) => [conversation, ...current]);
    setActiveConversationId(conversation.id);
    const result = await streamResponse(conversation.id, model, conversation, [userMessage], assistantMessage);
    if (result.completed) setPendingContext([]);
    if (result.completed && result.text.trim()) {
      void generateConversationCaption(conversation.id, model, content, result.text);
    }
  };

  const sendPrompt = async () => {
    const content = prompt.trim();
    if (!activeConversation || !activeProvider || streaming || !content) return;

    const userMessage = messageWithPendingContext(content);
    const assistantMessage: AiChatMessage = {
      ...createAiMessage('assistant', '', 'streaming'),
      modelId: activeProvider.id,
      modelLabel: activeProvider.label,
    };
    const requestMessages = [...activeConversation.messages, userMessage];

    setPrompt('');
    setContextPickerOpen(false);
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
    const result = await streamResponse(
      activeConversation.id,
      activeProvider,
      activeConversation,
      requestMessages,
      assistantMessage,
    );
    if (result.completed) setPendingContext([]);
  };

  const selectActiveModel = (modelId: string) => {
    if (!activeConversation || streaming) return;
    const model = configuredProviders.find((candidate) => candidate.id === modelId);
    if (!model) return;

    setConversations((current) =>
      current.map((conversation) =>
        conversation.id === activeConversation.id
          ? {
              ...conversation,
              modelId: model.id,
              modelLabel: model.label,
              updatedAt: new Date().toISOString(),
            }
          : conversation,
      ),
    );
  };

  const handleComposerKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key !== 'Enter' || event.shiftKey || event.nativeEvent.isComposing) return;
    event.preventDefault();
    event.currentTarget.form?.requestSubmit();
  };

  const addPendingContext = (reference: AiConversationContextReference) => {
    setPendingContext((current) =>
      current.some((candidate) => candidate.id === reference.id) ? current : [...current, reference],
    );
  };

  const removePendingContext = (id: string) => {
    setPendingContext((current) => current.filter((reference) => reference.id !== id));
  };

  const clearComposerContext = () => {
    setPendingContext([]);
    setContextPickerOpen(false);
  };

  const renderMessages = (conversation: AiConversation) => (
    <div className="ai-chat-message-list">
      {conversation.messages.map((message) => {
        const timestamp = <time dateTime={message.createdAt}>{formatMessageTime(message.createdAt)}</time>;

        if (message.role === 'assistant') {
          const responseModelId = message.modelId ?? conversation.modelId ?? activeProvider?.id;
          return (
            <UiAgentTextCard
              className={`ai-chat-message-card ai-chat-agent-card ${agentToneClass(responseModelId)}`}
              data-model-id={responseModelId}
              key={message.id}
              renderText={renderAiChatMessageText}
              side="left"
              state={message.state}
              text={message.content}
              timestamp={timestamp}
              tone="agent"
            />
          );
        }
        if (message.role === 'user') {
          return (
            <UiAgentTextCard
              className="ai-chat-message-card ai-chat-user-card"
              key={message.id}
              renderText={renderAiChatMessageText}
              side="right"
              state={message.state}
              text={message.content}
              timestamp={timestamp}
              tone="user"
            />
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

  const renderAddContextButton = () => (
    <UiIconButton
      className="ai-chat-add-button"
      disabled={!projectContextSource || streaming}
      label="Add context"
      type="button"
      variant="ghost"
      onClick={() => setContextPickerOpen((current) => !current)}
    >
      <Plus size={17} />
    </UiIconButton>
  );

  const renderContextPicker = () =>
    contextPickerOpen ? (
      <AiContextPicker
        onAdd={addPendingContext}
        onClose={() => setContextPickerOpen(false)}
        selected={pendingContext}
        source={projectContextSource}
        supportsImages={composerProvider?.capabilities?.inputModalities.includes('image') ?? false}
      />
    ) : null;

  const renderSubmitButton = (label: string, disabled: boolean) => (
    <UiIconButton
      className="ai-chat-submit-button"
      label={streaming ? 'Stop response' : label}
      disabled={streaming ? false : disabled}
      onClick={streaming ? stopResponse : undefined}
      type={streaming ? 'button' : 'submit'}
      variant="primary"
    >
      {streaming ? <Square fill="currentColor" size={10} strokeWidth={0} /> : <Send size={14} />}
    </UiIconButton>
  );

  return (
    <UiDrawerPanel className="ai-chat-panel" aria-label="AI Chat">
      {activeConversation ? (
        <section className="ai-chat-active" aria-label="Active conversation">
          <header className="ai-chat-active-header">
            <UiIconButton
              label="Back to conversations"
              onClick={() => {
                if (streaming) stopResponse();
                clearComposerContext();
                setActiveConversationId(null);
              }}
            >
              <ArrowLeft size={15} />
            </UiIconButton>
            <strong>{activeConversation.title}</strong>
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
              <AiContextChips references={pendingContext} onRemove={removePendingContext} />
              <textarea
                aria-label="Chat prompt"
                disabled={!activeProvider || streaming}
                placeholder={activeProvider ? 'Ask anything...' : 'The model for this conversation is unavailable'}
                value={prompt}
                rows={3}
                onChange={(event) => setPrompt(event.target.value)}
                onKeyDown={handleComposerKeyDown}
              />
              <div className="ai-chat-composer-toolbar ai-chat-composer-toolbar-active">
                {renderAddContextButton()}
                <div className="ai-chat-composer-actions">
                  <UiDropdown
                    ariaLabel="Model"
                    className="ai-chat-model-dropdown"
                    disabled={streaming || !activeProvider}
                    onValueChange={selectActiveModel}
                    options={modelOptions}
                    value={activeProvider?.id ?? configuredProviders[0]?.id ?? ''}
                  />
                  {renderSubmitButton('Send prompt', !activeProvider || !prompt.trim())}
                </div>
              </div>
            </div>
            {renderContextPicker()}
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
                      clearComposerContext();
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
                <span>
                  Connect an AI provider in Editor Preferences
                  <br />
                  to start a conversation.
                </span>
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
                  <AiContextChips references={pendingContext} onRemove={removePendingContext} />
                  <textarea
                    aria-label="Start a conversation"
                    disabled={streaming}
                    placeholder="Ask anything..."
                    value={prompt}
                    rows={3}
                    onChange={(event) => setPrompt(event.target.value)}
                    onKeyDown={handleComposerKeyDown}
                  />
                  <div className="ai-chat-composer-toolbar">
                    {renderAddContextButton()}
                    <div className="ai-chat-composer-actions">
                      <UiDropdown
                        ariaLabel="Model"
                        className="ai-chat-model-dropdown"
                        disabled={streaming}
                        onValueChange={setSelectedModelId}
                        options={modelOptions}
                        value={selectedModelId}
                      />
                      {renderSubmitButton('Start conversation', !prompt.trim())}
                    </div>
                  </div>
                </div>
                {renderContextPicker()}
              </form>
            )}
          </div>
        </section>
      )}
    </UiDrawerPanel>
  );
}

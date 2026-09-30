import { Send } from 'lucide-react';
import { useState } from 'react';
import { UiAgentTextCard, UiButton, UiDrawerPanel, UiIconButton } from '../ui';
import {
  createAiMessage,
  unavailableAiModelProvider,
  type AiChatMessage,
  type AiModelProvider,
} from './aiChat';
import './aiGateway.css';

const openAiConnectivitySettings = () => {
  window.dispatchEvent(
    new KeyboardEvent('keydown', {
      key: ',',
      code: 'Comma',
      ctrlKey: true,
      bubbles: true,
    }),
  );
  window.setTimeout(() => {
    window.dispatchEvent(new CustomEvent('arc-settings-navigate', { detail: { id: 'ai.providers' } }));
  }, 0);
};

type AiChatPanelProps = {
  provider?: AiModelProvider;
  initialMessages?: readonly AiChatMessage[];
  conversationLabel?: string;
};

export function AiChatPanel({
  provider = unavailableAiModelProvider,
  initialMessages = [],
  conversationLabel = 'New conversation',
}: AiChatPanelProps) {
  const [prompt, setPrompt] = useState('');
  const [messages, setMessages] = useState<AiChatMessage[]>(() => [...initialMessages]);
  const [streaming, setStreaming] = useState(false);
  const connected = provider.configured;

  const updateAssistantMessage = (id: string, update: (message: AiChatMessage) => AiChatMessage) => {
    setMessages((current) => current.map((message) => (message.id === id ? update(message) : message)));
  };

  const sendPrompt = async () => {
    const content = prompt.trim();
    if (!connected || streaming || !content) return;

    const userMessage = createAiMessage('user', content);
    const assistantMessage = createAiMessage('assistant', '', 'streaming');
    const requestMessages = [...messages, userMessage];

    setPrompt('');
    setStreaming(true);
    setMessages([...requestMessages, assistantMessage]);

    try {
      let completed = false;
      for await (const event of provider.stream({ conversationId: 'active', messages: requestMessages })) {
        if (event.type === 'delta') {
          updateAssistantMessage(assistantMessage.id, (message) => ({
            ...message,
            content: `${message.content}${event.text}`,
          }));
          continue;
        }
        if (event.type === 'error') {
          completed = true;
          updateAssistantMessage(assistantMessage.id, (message) => ({
            ...message,
            content: message.content ? `${message.content}\n\n${event.message}` : event.message,
            state: 'error',
          }));
          continue;
        }
        completed = true;
        updateAssistantMessage(assistantMessage.id, (message) => ({ ...message, state: 'complete' }));
      }

      if (!completed) {
        updateAssistantMessage(assistantMessage.id, (message) => ({ ...message, state: 'complete' }));
      }
    } catch (error) {
      updateAssistantMessage(assistantMessage.id, (message) => ({
        ...message,
        content: message.content || (error instanceof Error ? error.message : String(error)),
        state: 'error',
      }));
    } finally {
      setStreaming(false);
    }
  };

  return (
    <UiDrawerPanel className="ai-chat-panel" aria-label="AI Chat">
      <section className="ai-chat-conversations" aria-label="Conversations">
        <select aria-label="Conversation" defaultValue="active" disabled={!connected}>
          <option value="active">{conversationLabel}</option>
        </select>
      </section>

      <section className="ai-chat-session" aria-label="Chat" aria-disabled={!connected}>
        <div className="ai-chat-history" aria-label="Chat history" aria-busy={streaming}>
          {!connected ? (
            <div className="ai-chat-connect-empty">
              <strong>Connect your AI service</strong>
              <span>Connect an AI provider in Editor Preferences to start a conversation.</span>
              <UiButton onClick={openAiConnectivitySettings} variant="primary">
                Open AI settings
              </UiButton>
            </div>
          ) : messages.length > 0 ? (
            <div className="ai-chat-message-list">
              {messages.map((message) => {
                if (message.role === 'assistant') {
                  return (
                    <UiAgentTextCard
                      key={message.id}
                      state={message.state}
                      subtitle={provider.label}
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
          ) : null}
        </div>
        <form
          className="ai-chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
            void sendPrompt();
          }}
        >
          <div className="ai-chat-composer-surface">
            <textarea
              aria-label="Ask ARC"
              disabled={!connected}
              placeholder="Ask ARC..."
              value={prompt}
              rows={3}
              onChange={(event) => setPrompt(event.target.value)}
            />
            <div className="ai-chat-composer-toolbar">
              <label className="ai-chat-model-picker">
                <span>Model</span>
                <select aria-label="Model" defaultValue={provider.id} disabled={!connected || streaming}>
                  <option value={provider.id}>{provider.label}</option>
                </select>
              </label>
              <UiIconButton label="Send prompt" disabled={!connected || streaming || !prompt.trim()} type="submit">
                <Send size={15} />
              </UiIconButton>
            </div>
          </div>
        </form>
      </section>
    </UiDrawerPanel>
  );
}

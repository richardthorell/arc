import { MessageSquarePlus, Send } from 'lucide-react';
import { useState } from 'react';
import { UiDrawerPanel, UiIconButton } from '../ui';
import { unavailableAiModelProvider, type AiModelProvider } from './aiChat';
import './aiGateway.css';

export function AiChatPanel({ provider = unavailableAiModelProvider }: { provider?: AiModelProvider }) {
  const [prompt, setPrompt] = useState('');

  return (
    <UiDrawerPanel className="ai-chat-panel" aria-label="AI Chat">
      <section className="ai-chat-conversations" aria-label="Conversations">
        <header className="ai-chat-section-header">
          <strong>Conversations</strong>
          <UiIconButton label="New conversation" onClick={() => undefined}>
            <MessageSquarePlus size={15} />
          </UiIconButton>
        </header>

        <div className="ai-chat-conversation-controls">
          <label>
            <span>Conversation</span>
            <select aria-label="Conversation" defaultValue="new">
              <option value="new">New conversation</option>
            </select>
          </label>
          <label>
            <span>Model</span>
            <select aria-label="Model" defaultValue={provider.id}>
              <option value={provider.id}>{provider.label}</option>
            </select>
          </label>
        </div>
      </section>

      <section className="ai-chat-session" aria-label="Chat">
        <div className="ai-chat-history" aria-label="Chat history" />
        <form
          className="ai-chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
          }}
        >
          <textarea
            aria-label="Ask ARC"
            placeholder="Ask ARC..."
            value={prompt}
            rows={3}
            onChange={(event) => setPrompt(event.target.value)}
          />
          <UiIconButton label="Send prompt" disabled type="submit">
            <Send size={15} />
          </UiIconButton>
        </form>
      </section>
    </UiDrawerPanel>
  );
}

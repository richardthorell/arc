import { Send } from 'lucide-react';
import { useState } from 'react';
import { UiDrawerPanel, UiIconButton } from '../ui';
import { unavailableAiModelProvider, type AiModelProvider } from './aiChat';
import './aiGateway.css';

export function AiChatPanel({ provider = unavailableAiModelProvider }: { provider?: AiModelProvider }) {
  const [prompt, setPrompt] = useState('');

  return (
    <UiDrawerPanel className="ai-chat-panel" aria-label="AI Chat">
      <section className="ai-chat-conversations" aria-label="Conversations">
        <select aria-label="Conversation" defaultValue="new">
          <option value="new">New conversation</option>
        </select>
      </section>

      <section className="ai-chat-session" aria-label="Chat">
        <div className="ai-chat-history" aria-label="Chat history" />
        <form
          className="ai-chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
          }}
        >
          <div className="ai-chat-composer-surface">
            <textarea
              aria-label="Ask ARC"
              placeholder="Ask ARC..."
              value={prompt}
              rows={3}
              onChange={(event) => setPrompt(event.target.value)}
            />
            <div className="ai-chat-composer-toolbar">
              <label className="ai-chat-model-picker">
                <span>Model</span>
                <select aria-label="Model" defaultValue={provider.id}>
                  <option value={provider.id}>{provider.label}</option>
                </select>
              </label>
              <UiIconButton label="Send prompt" disabled type="submit">
                <Send size={15} />
              </UiIconButton>
            </div>
          </div>
        </form>
      </section>
    </UiDrawerPanel>
  );
}

import { Send } from 'lucide-react';
import { useState } from 'react';
import { UiButton, UiDrawerPanel, UiIconButton } from '../ui';
import { unavailableAiModelProvider, type AiModelProvider } from './aiChat';
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

export function AiChatPanel({ provider = unavailableAiModelProvider }: { provider?: AiModelProvider }) {
  const [prompt, setPrompt] = useState('');
  const connected = provider.configured;

  return (
    <UiDrawerPanel className="ai-chat-panel" aria-label="AI Chat">
      <section className="ai-chat-conversations" aria-label="Conversations">
        <select aria-label="Conversation" defaultValue="new" disabled={!connected}>
          <option value="new">New conversation</option>
        </select>
      </section>

      <section className="ai-chat-session" aria-label="Chat" aria-disabled={!connected}>
        <div className="ai-chat-history" aria-label="Chat history">
          {!connected && (
            <div className="ai-chat-connect-empty">
              <strong>Connect your AI service</strong>
              <span>Connect an AI provider in Editor Preferences to start a conversation.</span>
              <UiButton onClick={openAiConnectivitySettings} variant="primary">
                Open AI settings
              </UiButton>
            </div>
          )}
        </div>
        <form
          className="ai-chat-composer"
          onSubmit={(event) => {
            event.preventDefault();
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
                <select aria-label="Model" defaultValue={provider.id} disabled={!connected}>
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

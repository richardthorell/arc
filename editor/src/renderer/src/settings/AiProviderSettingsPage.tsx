import { useEffect, useState } from 'react';

import type { AiProviderAccountsSnapshot, AiProviderId } from '../../../common/aiProviderTypes';
import { UiButton, UiTextInput } from '../ui';

type AiProviderSettingsPageProps = {
  onMessage: (message: string) => void;
};

export function AiProviderSettingsPage({ onMessage }: AiProviderSettingsPageProps) {
  const [snapshot, setSnapshot] = useState<AiProviderAccountsSnapshot | null>(null);
  const [editingProvider, setEditingProvider] = useState<AiProviderId | null>(null);
  const [credential, setCredential] = useState('');
  const [busyProvider, setBusyProvider] = useState<AiProviderId | null>(null);

  useEffect(() => {
    void window.arcAiProviders
      .snapshot()
      .then(setSnapshot)
      .catch((error) => {
        onMessage(error instanceof Error ? error.message : String(error));
      });
  }, [onMessage]);

  const connect = async (providerId: AiProviderId) => {
    setBusyProvider(providerId);
    try {
      const next = await window.arcAiProviders.connect(providerId, credential);
      setSnapshot(next);
      setCredential('');
      setEditingProvider(null);
      const provider = next.providers.find((candidate) => candidate.id === providerId);
      onMessage(`${provider?.label ?? providerId} connected`);
    } catch (error) {
      onMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyProvider(null);
    }
  };

  const disconnect = async (providerId: AiProviderId) => {
    setBusyProvider(providerId);
    try {
      const next = await window.arcAiProviders.disconnect(providerId);
      setSnapshot(next);
      const provider = next.providers.find((candidate) => candidate.id === providerId);
      onMessage(`${provider?.label ?? providerId} disconnected`);
    } catch (error) {
      onMessage(error instanceof Error ? error.message : String(error));
    } finally {
      setBusyProvider(null);
    }
  };

  if (!snapshot) return <div className="tool-empty">Loading AI provider accounts...</div>;

  return (
    <div className="recovery-browser">
      <p>
        Connect provider API keys for ARC AI features. Keys are validated with the provider and encrypted using your
        operating system credential protection; they are never written to project or editor settings.
      </p>

      {!snapshot.secureStorageAvailable && (
        <div className="tool-error">
          Secure credential storage is unavailable. {snapshot.secureStorageDetail ?? 'ARC cannot save API keys.'}
        </div>
      )}

      {snapshot.providers.map((provider) => (
        <article key={provider.id}>
          <span>
            <strong>{provider.label}</strong>
            <small>{provider.connected ? 'Connected' : 'Not connected'}</small>
          </span>

          {provider.connected ? (
            <UiButton disabled={busyProvider !== null} onClick={() => void disconnect(provider.id)} variant="toolbar">
              {busyProvider === provider.id ? 'Disconnecting...' : 'Disconnect'}
            </UiButton>
          ) : editingProvider === provider.id ? (
            <>
              <UiTextInput
                aria-label={`${provider.label} API key`}
                autoFocus
                disabled={busyProvider !== null}
                onChange={(event) => setCredential(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === 'Enter' && credential.trim()) void connect(provider.id);
                  if (event.key === 'Escape') {
                    setCredential('');
                    setEditingProvider(null);
                  }
                }}
                placeholder={`${provider.label} API key`}
                type="password"
                value={credential}
              />
              <UiButton
                disabled={busyProvider !== null || !credential.trim()}
                onClick={() => void connect(provider.id)}
                variant="toolbar"
              >
                {busyProvider === provider.id ? 'Connecting...' : 'Connect'}
              </UiButton>
              <UiButton
                disabled={busyProvider !== null}
                onClick={() => {
                  setCredential('');
                  setEditingProvider(null);
                }}
                variant="toolbar"
              >
                Cancel
              </UiButton>
            </>
          ) : (
            <UiButton
              disabled={!snapshot.secureStorageAvailable || busyProvider !== null}
              onClick={() => {
                setCredential('');
                setEditingProvider(provider.id);
              }}
              variant="toolbar"
            >
              Connect
            </UiButton>
          )}
        </article>
      ))}
    </div>
  );
}

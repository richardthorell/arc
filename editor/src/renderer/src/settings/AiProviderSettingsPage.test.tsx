// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { AiProviderSettingsPage } from './AiProviderSettingsPage';

const disconnectedSnapshot = {
  secureStorageAvailable: true,
  providers: [
    { id: 'openai' as const, label: 'OpenAI', connected: false },
    { id: 'anthropic' as const, label: 'Anthropic', connected: false },
  ],
};

beforeEach(() => {
  vi.stubGlobal('arcAiProviders', {
    snapshot: vi.fn().mockResolvedValue(disconnectedSnapshot),
    connect: vi.fn().mockImplementation(async (providerId: 'openai' | 'anthropic') => ({
      ...disconnectedSnapshot,
      providers: disconnectedSnapshot.providers.map((provider) => ({
        ...provider,
        connected: provider.id === providerId,
      })),
    })),
    disconnect: vi.fn().mockResolvedValue(disconnectedSnapshot),
  });
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

describe('AiProviderSettingsPage', () => {
  it('connects a provider without exposing the saved credential', async () => {
    const onMessage = vi.fn();
    render(<AiProviderSettingsPage onMessage={onMessage} />);

    await waitFor(() => expect(window.arcAiProviders.snapshot).toHaveBeenCalledTimes(1));
    const connectButtons = screen.getAllByRole('button', { name: 'Connect' });
    fireEvent.click(connectButtons[0]);

    const key = screen.getByRole('textbox', { name: 'OpenAI API key' });
    fireEvent.change(key, { target: { value: 'sk-test-secret' } });
    fireEvent.click(screen.getByRole('button', { name: 'Connect' }));

    await waitFor(() =>
      expect(window.arcAiProviders.connect).toHaveBeenCalledWith('openai', 'sk-test-secret'),
    );
    expect(await screen.findByText('Connected')).toBeInTheDocument();
    expect(screen.queryByDisplayValue('sk-test-secret')).not.toBeInTheDocument();
  });

  it('disables connections when secure storage is unavailable', async () => {
    window.arcAiProviders.snapshot = vi.fn().mockResolvedValue({
      ...disconnectedSnapshot,
      secureStorageAvailable: false,
      secureStorageDetail: 'A Linux secret store is required',
    });

    render(<AiProviderSettingsPage onMessage={vi.fn()} />);

    expect(await screen.findByText(/A Linux secret store is required/)).toBeInTheDocument();
    for (const button of screen.getAllByRole('button', { name: 'Connect' })) expect(button).toBeDisabled();
  });
});

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import type { EditorSettingsSnapshot } from '../../../common/editorWorkflowTypes';
import { AiGatewayPanel } from './AiGatewayPanel';

const settingsSnapshot = (connected: boolean): EditorSettingsSnapshot => ({
  revision: connected ? 2 : 1,
  values: { 'ai.openai.model': 'gpt-5.6-sol' },
  sources: {},
  restartRequired: [],
  schema: [
    {
      key: 'ai.openai.model',
      section: 'OpenAI',
      label: 'Model',
      description: 'OpenAI model',
      type: 'enum',
      defaultValue: 'gpt-5.6-sol',
      options: ['gpt-5.6-sol', 'gpt-5.6-luna'],
      optionLabels: { 'gpt-5.6-sol': 'GPT-5.6 Sol', 'gpt-5.6-luna': 'GPT-5.6 Luna' },
      scopes: ['user'],
    },
  ],
  aiProviders: {
    secureStorageAvailable: true,
    providers: [
      {
        id: 'openai',
        label: 'OpenAI',
        connected,
        connectionStatus: connected ? 'connected' : 'disconnected',
      },
    ],
  },
});

const originalArc = Object.getOwnPropertyDescriptor(window, 'arc');

afterEach(() => {
  cleanup();
  if (originalArc) Object.defineProperty(window, 'arc', originalArc);
  else Reflect.deleteProperty(window, 'arc');
});

describe('AiGatewayPanel runtime providers', () => {
  it('rechecks provider connectivity when Editor Preferences closes', async () => {
    const snapshot = vi.fn().mockResolvedValueOnce(settingsSnapshot(false)).mockResolvedValue(settingsSnapshot(true));
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: { settings: { snapshot } },
    });

    render(
      <AiGatewayPanel
        onApprove={vi.fn()}
        onCancelEdit={vi.fn()}
        onDeny={vi.fn()}
        onRevoke={vi.fn()}
        onUndoLastEdit={vi.fn()}
        status={null}
      />,
    );

    await waitFor(() => expect(screen.getByText('Connect your AI service')).toBeVisible());

    window.dispatchEvent(new Event('arc-editor-settings-closed'));

    await waitFor(() => expect(screen.getByLabelText('Model')).toHaveTextContent('GPT-5.6 Sol'));
    fireEvent.click(screen.getByLabelText('Model'));
    expect(screen.getByRole('option', { name: 'GPT-5.6 Sol' })).toHaveAttribute('aria-selected', 'true');
    expect(snapshot).toHaveBeenCalledTimes(2);
  });

  it('applies settings-change snapshots without waiting for the dialog to close', async () => {
    const snapshot = vi.fn().mockResolvedValue(settingsSnapshot(false));
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: { settings: { snapshot } },
    });

    render(
      <AiGatewayPanel
        onApprove={vi.fn()}
        onCancelEdit={vi.fn()}
        onDeny={vi.fn()}
        onRevoke={vi.fn()}
        onUndoLastEdit={vi.fn()}
        status={null}
      />,
    );

    await waitFor(() => expect(screen.getByText('Connect your AI service')).toBeVisible());
    window.dispatchEvent(new CustomEvent('arc-editor-settings-changed', { detail: settingsSnapshot(true) }));

    await waitFor(() => expect(screen.getByLabelText('Model')).toHaveTextContent('GPT-5.6 Sol'));
  });
});

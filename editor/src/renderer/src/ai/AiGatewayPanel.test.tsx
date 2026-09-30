// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AiGatewayApprovalPrompt } from './AiGatewayPanel';
import { AiChatPanel } from './AiChatPanel';
import type { AiChatMessage, AiModelProvider } from './aiChat';
import type { ArcAiGatewayStatus } from '../../../preload/preload';
import {
  requestedSettingsDialogKind,
  requestedSettingsDialogPageId,
  resetSettingsDialogRequest,
  subscribeSettingsDialogOpenRequests,
} from '../settings/settingsDialogRoute';

afterEach(() => {
  cleanup();
  resetSettingsDialogRequest();
  vi.useRealTimers();
});

const status: ArcAiGatewayStatus = {
  enabled: true,
  endpoint: 'http://127.0.0.1:43123',
  discoveryFile: 'C:/Users/Test/ARC/ai-gateway/active.json',
  protocolVersion: 1,
  sceneRevision: 7,
  worldEpoch: 2,
  frameRevision: 42,
  eventSequence: 3,
  clients: [
    {
      id: 'codex',
      name: 'Codex',
      connectedAt: '2026-01-01T00:00:00Z',
      lastSeenAt: '2026-01-01T00:00:01Z',
    },
  ],
  pendingEditRequests: [
    {
      id: 'request',
      clientId: 'codex',
      clientName: 'Codex',
      label: 'Fix scene',
      requestedAt: '2026-01-01T00:00:00Z',
      state: 'pending',
    },
  ],
  activeEditSession: null,
  lastCommittedEdit: null,
  viewportLease: { clientId: 'codex', expiresAt: '2026-01-01T00:01:00Z' },
  audit: [],
};

const configuredProvider: AiModelProvider = {
  id: 'test',
  label: 'Test Agent',
  configured: true,
  async *stream() {
    yield { type: 'delta', text: 'Hello ' };
    yield { type: 'delta', text: 'from ARC.' };
    yield { type: 'done' };
  },
};

describe('AiChatPanel', () => {
  it('renders an enabled two-section chat shell for a configured provider', () => {
    render(<AiChatPanel provider={configuredProvider} />);

    expect(screen.getByRole('region', { name: 'AI Chat' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Conversations' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Chat' })).toBeInTheDocument();
    expect(screen.getByLabelText('Conversation')).toHaveValue('active');
    expect(screen.getByLabelText('Conversation')).toBeEnabled();
    expect(screen.getByLabelText('Model')).toHaveValue('test');
    expect(screen.getByLabelText('Model')).toBeEnabled();
    expect(screen.getByText('Test Agent')).toBeInTheDocument();
    expect(screen.getByLabelText('Chat history')).toBeEmptyDOMElement();
    expect(screen.getByLabelText('Ask ARC')).toBeEnabled();
    expect(screen.getByLabelText('Send prompt')).toBeDisabled();
  });

  it('renders agent responses with the text-card specialization and streams mock replies', async () => {
    const initialMessages: readonly AiChatMessage[] = [
      {
        id: 'user',
        role: 'user',
        content: 'What is selected?',
        createdAt: '2026-09-30T17:00:00Z',
        state: 'complete',
      },
      {
        id: 'assistant',
        role: 'assistant',
        content: 'A cabin mesh is selected.',
        createdAt: '2026-09-30T17:00:01Z',
        state: 'complete',
      },
    ];

    render(
      <AiChatPanel conversationLabel="Scene review" initialMessages={initialMessages} provider={configuredProvider} />,
    );

    expect(screen.getByText('Scene review')).toBeInTheDocument();
    expect(screen.getByText('A cabin mesh is selected.')).toBeInTheDocument();
    expect(screen.getByText('ARC')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Ask ARC'), { target: { value: 'Suggest a polish pass' } });
    expect(screen.getByLabelText('Send prompt')).toBeEnabled();
    fireEvent.click(screen.getByLabelText('Send prompt'));

    expect(screen.getByText('Suggest a polish pass')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Hello from ARC.')).toBeInTheDocument());
  });

  it('routes disconnected users directly to AI provider settings', () => {
    const requests: Array<{ kind: string; pageId: string | null }> = [];
    const unsubscribe = subscribeSettingsDialogOpenRequests((request) => requests.push(request));

    render(<AiChatPanel />);

    expect(screen.getByLabelText('Conversation')).toBeDisabled();
    expect(screen.getByLabelText('Model')).toBeDisabled();
    expect(screen.getByLabelText('Ask ARC')).toBeDisabled();
    expect(screen.getByText('Connect your AI service')).toBeVisible();

    fireEvent.click(screen.getByRole('button', { name: 'Open AI settings' }));

    expect(requests).toEqual([{ kind: 'editorPreferences', pageId: 'ai.providers' }]);
    expect(requestedSettingsDialogKind()).toBe('editorPreferences');
    expect(requestedSettingsDialogPageId()).toBe('ai.providers');
    unsubscribe();
  });

  it('keeps edit approval outside the chat panel', () => {
    const approve = vi.fn();
    const deny = vi.fn();
    const open = vi.fn();
    render(<AiGatewayApprovalPrompt status={status} onApprove={approve} onDeny={deny} onOpenGateway={open} />);
    expect(screen.getByRole('alertdialog')).toHaveTextContent('Codex requests editor action access');
    fireEvent.click(screen.getByText('Allow'));
    fireEvent.click(screen.getByText('Deny'));
    fireEvent.click(screen.getByText('Open chat'));
    expect(approve).toHaveBeenCalledWith('request');
    expect(deny).toHaveBeenCalledWith('request');
    expect(open).toHaveBeenCalledOnce();
  });
});

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AiGatewayApprovalPrompt } from './AiGatewayPanel';
import { AiChatPanel } from './AiChatPanel';
import type { AiModelProvider } from './aiChat';
import type { ArcAiGatewayStatus } from '../../../preload/preload';

afterEach(() => {
  cleanup();
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

describe('AiChatPanel', () => {
  it('renders an enabled two-section chat shell for a configured provider', () => {
    const provider: AiModelProvider = {
      id: 'test',
      label: 'Test Agent',
      configured: true,
      async *stream() {
        yield { type: 'done' };
      },
    };

    render(<AiChatPanel provider={provider} />);

    expect(screen.getByRole('region', { name: 'AI Chat' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Conversations' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Chat' })).toBeInTheDocument();
    expect(screen.getByLabelText('Conversation')).toHaveValue('new');
    expect(screen.getByLabelText('Conversation')).toBeEnabled();
    expect(screen.getByLabelText('Model')).toHaveValue('test');
    expect(screen.getByLabelText('Model')).toBeEnabled();
    expect(screen.getByText('Test Agent')).toBeInTheDocument();
    expect(screen.getByLabelText('Chat history')).toBeEmptyDOMElement();
    expect(screen.getByLabelText('Ask ARC')).toBeEnabled();
    expect(screen.getByLabelText('Send prompt')).toBeDisabled();
  });

  it('disables conversations and chat until an AI service is connected', () => {
    vi.useFakeTimers();
    const keydown = vi.fn();
    const navigate = vi.fn();
    window.addEventListener('keydown', keydown);
    window.addEventListener('arc-settings-navigate', navigate);

    render(<AiChatPanel />);

    expect(screen.getByLabelText('Conversation')).toBeDisabled();
    expect(screen.getByLabelText('Model')).toBeDisabled();
    expect(screen.getByLabelText('Ask ARC')).toBeDisabled();
    expect(screen.getByText('Connect your AI service')).toBeVisible();

    fireEvent.click(screen.getByRole('button', { name: 'Open AI settings' }));
    expect(keydown).toHaveBeenCalledOnce();
    expect((keydown.mock.calls[0][0] as KeyboardEvent).ctrlKey).toBe(true);
    expect((keydown.mock.calls[0][0] as KeyboardEvent).key).toBe(',');

    vi.runAllTimers();
    expect(navigate).toHaveBeenCalledOnce();
    expect((navigate.mock.calls[0][0] as CustomEvent).detail).toEqual({ id: 'ai.providers' });

    window.removeEventListener('keydown', keydown);
    window.removeEventListener('arc-settings-navigate', navigate);
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

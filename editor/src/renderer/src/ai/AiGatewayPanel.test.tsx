// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AiChatPanel, AiGatewayApprovalPrompt } from './AiGatewayPanel';
import { aiConversationStorageKey, type AiModelProvider } from './aiChat';
import type { ArcAiGatewayStatus } from '../../../preload/preload';

Object.defineProperty(HTMLElement.prototype, 'scrollIntoView', {
  configurable: true,
  value: vi.fn(),
});

afterEach(() => {
  cleanup();
  localStorage.removeItem(aiConversationStorageKey);
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

const renderPanel = (provider?: AiModelProvider) => render(<AiChatPanel provider={provider} />);

describe('AiChatPanel', () => {
  it('renders as AI Chat and streams a provider response', async () => {
    const provider: AiModelProvider = {
      id: 'test',
      label: 'Test Agent',
      configured: true,
      async *stream(request) {
        expect(request.messages.at(-1)?.content).toBe('How should I light this room?');
        yield { type: 'delta', text: 'Start with ' };
        yield { type: 'delta', text: 'a key light.' };
        yield { type: 'done' };
      },
    };

    renderPanel(provider);
    expect(screen.getByRole('region', { name: 'AI Chat' })).toBeInTheDocument();
    expect(screen.getByText('AI Chat')).toBeInTheDocument();
    expect(screen.getByText('Test Agent')).toBeInTheDocument();
    expect(screen.queryByLabelText('Gateway diagnostics')).not.toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Ask ARC'), {
      target: { value: 'How should I light this room?' },
    });
    fireEvent.click(screen.getByLabelText('Send prompt'));

    const promptMessage = screen
      .getAllByText('How should I light this room?')
      .find((element) => element.tagName === 'P');
    expect(promptMessage).toBeVisible();
    await waitFor(() => expect(screen.getByText('Start with a key light.')).toBeVisible());
    expect((screen.getByLabelText('AI conversation') as HTMLSelectElement).value).not.toBe('');
    expect(localStorage.getItem(aiConversationStorageKey)).toContain('How should I light this room?');
  });

  it('creates a fresh conversation from the header', () => {
    renderPanel();
    const selector = screen.getByLabelText('AI conversation');
    expect(selector.querySelectorAll('option')).toHaveLength(1);
    fireEvent.click(screen.getByLabelText('New AI chat'));
    expect(selector.querySelectorAll('option')).toHaveLength(2);
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

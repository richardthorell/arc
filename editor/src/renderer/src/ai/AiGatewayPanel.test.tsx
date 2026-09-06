// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AiGatewayApprovalPrompt, AiGatewayPanel } from './AiGatewayPanel';
import { aiConversationStorageKey, type AiModelProvider } from './aiChat';
import type { ArcAiGatewayStatus } from '../../../preload/preload';

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
  clients: [{ id: 'codex', name: 'Codex', connectedAt: '2026-01-01T00:00:00Z', lastSeenAt: '2026-01-01T00:00:01Z' }],
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
  audit: [
    {
      sequence: 1,
      timestamp: '2026-01-01T00:00:00Z',
      clientId: 'codex',
      category: 'read',
      operation: 'scene.overview',
      succeeded: true,
      detail: '',
    },
  ],
};

const renderPanel = (provider?: AiModelProvider) =>
  render(
    <AiGatewayPanel
      status={status}
      onApprove={() => undefined}
      onDeny={() => undefined}
      onRevoke={() => undefined}
      onCancelEdit={() => undefined}
      onUndoLastEdit={() => undefined}
      provider={provider}
    />,
  );

describe('AiGatewayPanel', () => {
  it('opens as a real assistant chat and streams a provider response', async () => {
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
    expect(screen.getByRole('region', { name: 'ARC Assistant' })).toBeInTheDocument();
    expect(screen.getByText('Test Agent')).toBeInTheDocument();

    fireEvent.change(screen.getByLabelText('Ask ARC'), { target: { value: 'How should I light this room?' } });
    fireEvent.click(screen.getByLabelText('Send prompt'));

    expect(screen.getByText('How should I light this room?')).toBeVisible();
    await waitFor(() => expect(screen.getByText('Start with a key light.')).toBeVisible());
    expect(screen.getByLabelText('AI conversation')).toHaveValue(expect.any(String));
    expect(localStorage.getItem(aiConversationStorageKey)).toContain('How should I light this room?');
  });

  it('creates a fresh conversation from the header', () => {
    renderPanel();
    const selector = screen.getByLabelText('AI conversation');
    expect(selector.querySelectorAll('option')).toHaveLength(1);
    fireEvent.click(screen.getByLabelText('New AI chat'));
    expect(selector.querySelectorAll('option')).toHaveLength(2);
  });

  it('keeps gateway administration behind diagnostics', () => {
    const approve = vi.fn();
    render(
      <AiGatewayPanel
        status={status}
        onApprove={approve}
        onDeny={() => undefined}
        onRevoke={() => undefined}
        onCancelEdit={() => undefined}
        onUndoLastEdit={() => undefined}
      />,
    );

    expect(screen.queryByText(status.endpoint)).not.toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('Gateway diagnostics'));
    expect(screen.getByLabelText('AI Gateway diagnostics')).toBeInTheDocument();
    expect(screen.getByText(status.endpoint)).toBeInTheDocument();
    expect(screen.getAllByText('Codex').length).toBeGreaterThan(0);
    expect(screen.getByText(/Viewport control/)).toBeInTheDocument();
    expect(screen.getByText('scene.overview')).toBeInTheDocument();
    fireEvent.click(screen.getByText(/Allow 15 min/));
    expect(approve).toHaveBeenCalledWith('request');
  });

  it('offers immediate revoke and transaction cancellation in diagnostics', () => {
    const revoke = vi.fn();
    const cancel = vi.fn();
    const undo = vi.fn();
    render(
      <AiGatewayPanel
        status={{
          ...status,
          pendingEditRequests: [],
          activeEditSession: {
            id: 'edit',
            clientId: 'codex',
            label: 'Adjust light',
            startedAt: '2026-01-01T00:00:00Z',
            lastActivityAt: '2026-01-01T00:00:00Z',
            expectedSceneRevision: 9,
          },
          lastCommittedEdit: {
            clientId: 'codex',
            label: 'Previous light edit',
            sceneRevision: 8,
            committedAt: '2026-01-01T00:00:00Z',
          },
        }}
        onApprove={() => undefined}
        onDeny={() => undefined}
        onRevoke={revoke}
        onCancelEdit={cancel}
        onUndoLastEdit={undo}
      />,
    );
    fireEvent.click(screen.getByLabelText('Gateway diagnostics'));
    fireEvent.click(screen.getByLabelText('Revoke Codex'));
    fireEvent.click(screen.getByText('Cancel'));
    fireEvent.click(screen.getByText('Undo'));
    expect(revoke).toHaveBeenCalledWith('codex');
    expect(cancel).toHaveBeenCalledWith('edit', 'codex');
    expect(undo).toHaveBeenCalledOnce();
  });

  it('surfaces edit approval outside the assistant panel', () => {
    const approve = vi.fn();
    const deny = vi.fn();
    const open = vi.fn();
    render(<AiGatewayApprovalPrompt status={status} onApprove={approve} onDeny={deny} onOpenGateway={open} />);
    expect(screen.getByRole('alertdialog')).toHaveTextContent('Codex requests scene edit access');
    fireEvent.click(screen.getByText('Allow'));
    fireEvent.click(screen.getByText('Deny'));
    fireEvent.click(screen.getByText('Details'));
    expect(approve).toHaveBeenCalledWith('request');
    expect(deny).toHaveBeenCalledWith('request');
    expect(open).toHaveBeenCalledOnce();
  });
});

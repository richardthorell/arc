// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AiGatewayApprovalPrompt, AiGatewayPanel } from './AiGatewayPanel';
import type { ArcAiGatewayStatus } from '../../../preload/preload';

afterEach(cleanup);

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
  pendingEditRequests: [{
    id: 'request', clientId: 'codex', clientName: 'Codex', label: 'Fix scene',
    requestedAt: '2026-01-01T00:00:00Z', state: 'pending',
  }],
  activeEditSession: null,
  lastCommittedEdit: null,
  viewportLease: { clientId: 'codex', expiresAt: '2026-01-01T00:01:00Z' },
  audit: [{
    sequence: 1, timestamp: '2026-01-01T00:00:00Z', clientId: 'codex',
    category: 'read', operation: 'scene.overview', succeeded: true, detail: '',
  }],
};

describe('AiGatewayPanel', () => {
  it('shows endpoint, client, lease, approvals, and audit state', () => {
    const approve = vi.fn();
    render(<AiGatewayPanel status={status} onApprove={approve} onDeny={() => undefined}
      onRevoke={() => undefined} onCancelEdit={() => undefined} onUndoLastEdit={() => undefined} />);
    expect(screen.getByText(status.endpoint)).toBeInTheDocument();
    expect(screen.getAllByText('Codex').length).toBeGreaterThan(0);
    expect(screen.getByText(/Viewport control/)).toBeInTheDocument();
    expect(screen.getByText('scene.overview')).toBeInTheDocument();
    fireEvent.click(screen.getByText(/Allow 15 min/));
    expect(approve).toHaveBeenCalledWith('request');
  });

  it('offers immediate revoke and transaction cancellation', () => {
    const revoke = vi.fn();
    const cancel = vi.fn();
    const undo = vi.fn();
    render(<AiGatewayPanel status={{
      ...status,
      pendingEditRequests: [],
      activeEditSession: {
        id: 'edit', clientId: 'codex', label: 'Adjust light',
        startedAt: '2026-01-01T00:00:00Z', lastActivityAt: '2026-01-01T00:00:00Z',
        expectedSceneRevision: 9,
      },
      lastCommittedEdit: {
        clientId: 'codex', label: 'Previous light edit', sceneRevision: 8,
        committedAt: '2026-01-01T00:00:00Z',
      },
    }} onApprove={() => undefined} onDeny={() => undefined} onRevoke={revoke}
      onCancelEdit={cancel} onUndoLastEdit={undo} />);
    fireEvent.click(screen.getByLabelText('Revoke Codex'));
    fireEvent.click(screen.getByText('Cancel'));
    fireEvent.click(screen.getByText('Undo'));
    expect(revoke).toHaveBeenCalledWith('codex');
    expect(cancel).toHaveBeenCalledWith('edit', 'codex');
    expect(undo).toHaveBeenCalledOnce();
  });

  it('surfaces edit approval outside the gateway panel', () => {
    const approve = vi.fn();
    const deny = vi.fn();
    const open = vi.fn();
    render(<AiGatewayApprovalPrompt status={status} onApprove={approve} onDeny={deny}
      onOpenGateway={open} />);
    expect(screen.getByRole('alertdialog')).toHaveTextContent('Codex requests scene edit access');
    fireEvent.click(screen.getByText('Allow'));
    fireEvent.click(screen.getByText('Deny'));
    fireEvent.click(screen.getByText('Details'));
    expect(approve).toHaveBeenCalledWith('request');
    expect(deny).toHaveBeenCalledWith('request');
    expect(open).toHaveBeenCalledOnce();
  });
});

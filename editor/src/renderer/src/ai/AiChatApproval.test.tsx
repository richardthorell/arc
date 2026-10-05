// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AiChatPanel } from './AiChatPanel';
import type { AiAgentApprovalRequest } from './aiAgentApproval';
import type { AiModelProvider } from './aiChat';

const provider: AiModelProvider = {
  id: 'openai:test',
  providerId: 'openai',
  modelId: 'test',
  label: 'Test Agent',
  configured: true,
  capabilities: { streaming: true, tools: true, inputModalities: ['text'] },
  async *stream() {
    yield { type: 'delta', text: 'Working.' };
    yield { type: 'done', finishReason: 'stop' };
  },
};

const pendingApproval: AiAgentApprovalRequest = {
  id: 'request-1',
  clientId: 'arc.builtin-ai',
  clientName: 'ARC Built-in AI',
  label: 'Create cube above Floor',
  requestedAt: '2026-10-04T05:40:00.000Z',
  state: 'pending',
};

afterEach(() => cleanup());

describe('AiChatPanel approvals', () => {
  it('shows ask-mode approval as compact awaiting progress and approves it', async () => {
    const onApproveRequest = vi.fn(async () => true);
    render(
      <AiChatPanel
        approvalMode="ask"
        onApprovalModeChange={vi.fn()}
        onApproveRequest={onApproveRequest}
        onDenyRequest={vi.fn(async () => true)}
        pendingApproval={pendingApproval}
        provider={provider}
      />,
    );

    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Create a cube' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));

    await waitFor(() => expect(screen.getByRole('alertdialog', { name: 'AI editor action approval' })).toBeVisible());
    expect(screen.getByText('Awaiting approval')).toBeVisible();
    expect(screen.getByText('Create cube above Floor')).toBeVisible();
    expect(screen.getByLabelText('Agent approval mode')).toHaveTextContent('Ask');

    fireEvent.click(screen.getByRole('button', { name: /Allow/u }));
    await waitFor(() => expect(onApproveRequest).toHaveBeenCalledWith('request-1'));
  });

  it('keeps denial wired to the harness coordinator so the agent receives the denied result', async () => {
    const onDenyRequest = vi.fn(async () => true);
    render(
      <AiChatPanel
        approvalMode="ask"
        onApproveRequest={vi.fn(async () => true)}
        onDenyRequest={onDenyRequest}
        pendingApproval={pendingApproval}
        provider={provider}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: /Deny/u }));
    await waitFor(() => expect(onDenyRequest).toHaveBeenCalledWith('request-1'));
  });

  it('does not render pending approval UI in auto mode', () => {
    render(<AiChatPanel approvalMode="auto" pendingApproval={pendingApproval} provider={provider} />);

    expect(screen.queryByRole('alertdialog', { name: 'AI editor action approval' })).not.toBeInTheDocument();
    expect(screen.queryByText('Awaiting approval')).not.toBeInTheDocument();
  });

  it('offers auto approve next to the model control', async () => {
    const onApprovalModeChange = vi.fn();
    render(
      <AiChatPanel
        approvalMode="ask"
        onApprovalModeChange={onApprovalModeChange}
        pendingApproval={null}
        provider={provider}
      />,
    );

    expect(screen.getByLabelText('Agent approval mode')).toBeVisible();
    fireEvent.click(screen.getByLabelText('Agent approval mode'));
    fireEvent.click(await screen.findByRole('option', { name: /Auto approve/u }));
    expect(onApprovalModeChange).toHaveBeenCalledWith('auto');
  });
});

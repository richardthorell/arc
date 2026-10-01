// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { AiGatewayApprovalPrompt } from './AiGatewayPanel';
import { AiChatPanel } from './AiChatPanel';
import { aiConversationStorageKey, type AiChatMessage, type AiConversation, type AiModelProvider } from './aiChat';
import type { ArcAiGatewayStatus } from '../../../preload/preload';
import {
  requestedSettingsDialogKind,
  requestedSettingsDialogPageId,
  resetSettingsDialogRequest,
  subscribeSettingsDialogOpenRequests,
} from '../settings/settingsDialogRoute';

afterEach(() => {
  cleanup();
  localStorage.removeItem(aiConversationStorageKey);
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
  id: 'openai:test',
  label: 'Test Agent',
  configured: true,
  async *stream() {
    yield { type: 'delta', text: 'Hello ' };
    yield { type: 'delta', text: 'from ARC.' };
    yield { type: 'done' };
  },
};

const alternateProvider: AiModelProvider = {
  id: 'anthropic:alternate',
  label: 'Alternate Agent',
  configured: true,
  async *stream() {
    yield { type: 'delta', text: 'Alternate reply.' };
    yield { type: 'done' };
  },
};

const recentConversation: AiConversation = {
  id: 'recent',
  title: 'Cabin polish',
  createdAt: '2026-09-30T17:00:00Z',
  updatedAt: '2026-09-30T17:00:01Z',
  modelId: 'openai:test',
  modelLabel: 'Test Agent',
  messages: [
    {
      id: 'recent-user',
      role: 'user',
      content: 'Polish the cabin.',
      createdAt: '2026-09-30T17:00:00Z',
      state: 'complete',
    },
  ],
};

describe('AiChatPanel', () => {
  it('starts on a conversation home with the polished composer controls and model icons', () => {
    render(<AiChatPanel persistConversations={false} providers={[configuredProvider, alternateProvider]} />);

    expect(screen.getByRole('region', { name: 'AI Chat' })).toBeInTheDocument();
    expect(screen.getByRole('region', { name: 'Conversations' })).toBeInTheDocument();
    expect(screen.queryByRole('region', { name: 'Active conversation' })).not.toBeInTheDocument();
    expect(screen.getByLabelText('Start a conversation')).toBeEnabled();
    expect(screen.getByLabelText('Start a conversation')).toHaveAttribute('placeholder', 'Ask anything...');
    expect(screen.getByRole('button', { name: 'Add context' })).toBeInTheDocument();

    const modelDropdown = screen.getByLabelText('Model');
    expect(modelDropdown).toBeEnabled();
    expect(modelDropdown).toHaveTextContent('Test Agent');
    fireEvent.click(modelDropdown);

    const testOption = screen.getByRole('option', { name: 'Test Agent' });
    const alternateOption = screen.getByRole('option', { name: 'Alternate Agent' });
    expect(testOption.querySelector('.ui-dropdown-icon')).toBeInTheDocument();
    expect(alternateOption.querySelector('.ui-dropdown-icon')).toBeInTheDocument();
    expect(screen.getByLabelText('Start conversation')).toBeDisabled();
    expect(screen.queryByLabelText('Recent conversations')).not.toBeInTheDocument();
  });

  it('does not show empty placeholder conversations in recent history', () => {
    const emptyConversation: AiConversation = {
      id: 'empty',
      title: 'New Chat',
      createdAt: '2026-09-30T17:00:00Z',
      updatedAt: '2026-09-30T17:00:00Z',
      messages: [],
    };

    render(
      <AiChatPanel
        initialConversations={[emptyConversation]}
        persistConversations={false}
        provider={configuredProvider}
      />,
    );

    expect(screen.queryByLabelText('Recent conversations')).not.toBeInTheDocument();
    expect(screen.queryByText('New Chat')).not.toBeInTheDocument();
  });

  it('carries the selected model into the conversation and allows switching models', async () => {
    render(<AiChatPanel persistConversations={false} providers={[configuredProvider, alternateProvider]} />);

    fireEvent.click(screen.getByLabelText('Model'));
    fireEvent.click(screen.getByRole('option', { name: 'Alternate Agent' }));
    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Polish the cabin material' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));

    const activeConversation = screen.getByRole('region', { name: 'Active conversation' });
    expect(activeConversation).toBeInTheDocument();
    expect(activeConversation).toHaveTextContent('Polish the cabin material');
    expect(screen.getByLabelText('Model')).toHaveTextContent('Alternate Agent');
    expect(screen.getByRole('button', { name: 'Add context' })).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Alternate reply.')).toBeInTheDocument());

    fireEvent.click(screen.getByLabelText('Model'));
    fireEvent.click(screen.getByRole('option', { name: 'Test Agent' }));
    expect(screen.getByLabelText('Model')).toHaveTextContent('Test Agent');

    fireEvent.click(screen.getByLabelText('Back to conversations'));

    expect(screen.getByRole('region', { name: 'Conversations' })).toBeInTheDocument();
    expect(screen.getByLabelText('Model')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Open conversation Polish the cabin material' })).toBeInTheDocument();
  });

  it('opens a recent conversation with the model dropdown available while chatting', async () => {
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

    fireEvent.click(screen.getByRole('button', { name: 'Open conversation Scene review' }));

    expect(screen.getByText('A cabin mesh is selected.')).toBeInTheDocument();
    expect(screen.getByLabelText('Model')).toHaveTextContent('Test Agent');

    fireEvent.change(screen.getByLabelText('Chat prompt'), { target: { value: 'Suggest a polish pass' } });
    expect(screen.getByLabelText('Chat prompt')).toHaveAttribute('placeholder', 'Ask anything...');
    expect(screen.getByLabelText('Send prompt')).toBeEnabled();
    fireEvent.click(screen.getByLabelText('Send prompt'));

    expect(screen.getByText('Suggest a polish pass')).toBeInTheDocument();
    await waitFor(() => expect(screen.getByText('Hello from ARC.')).toBeInTheDocument());
  });

  it('turns the round send action into a stop action while a response streams', async () => {
    let releaseStream: (() => void) | undefined;
    let streamSignal: AbortSignal | undefined;
    const slowProvider: AiModelProvider = {
      id: 'slow',
      label: 'Slow Agent',
      configured: true,
      async *stream(request) {
        streamSignal = request.signal;
        yield { type: 'delta', text: 'Working' };
        await new Promise<void>((resolve) => {
          releaseStream = resolve;
        });
        yield { type: 'delta', text: ' should not arrive' };
        yield { type: 'done' };
      },
    };

    render(<AiChatPanel persistConversations={false} provider={slowProvider} />);

    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Do some work' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));

    await waitFor(() => expect(screen.getByText('Working')).toBeInTheDocument());
    await waitFor(() => expect(releaseStream).toBeTypeOf('function'));
    expect(screen.getByLabelText('Stop response')).toBeEnabled();
    expect(screen.getByLabelText('Model')).toBeDisabled();

    fireEvent.click(screen.getByLabelText('Stop response'));

    expect(streamSignal?.aborted).toBe(true);
    expect(screen.queryByLabelText('Stop response')).not.toBeInTheDocument();
    expect(screen.getByLabelText('Chat prompt')).toBeEnabled();
    expect(screen.getByLabelText('Model')).toBeEnabled();

    releaseStream?.();
    await waitFor(() => expect(screen.queryByText(/should not arrive/)).not.toBeInTheDocument());
  });

  it('routes disconnected users directly to AI provider settings without showing conversation history', () => {
    const requests: Array<{ kind: string; pageId: string | null }> = [];
    const unsubscribe = subscribeSettingsDialogOpenRequests((request) => requests.push(request));

    render(<AiChatPanel initialConversations={[recentConversation]} persistConversations={false} />);

    expect(screen.queryByLabelText('Recent conversations')).not.toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Open conversation Cabin polish' })).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Start a conversation')).not.toBeInTheDocument();
    expect(screen.queryByLabelText('Model')).not.toBeInTheDocument();
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

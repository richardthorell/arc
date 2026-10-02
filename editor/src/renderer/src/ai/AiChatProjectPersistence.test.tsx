// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { AiChatPanel } from './AiChatPanel';
import type { AiModelProvider } from './aiChat';

const projectA = '11111111-1111-1111-1111-111111111111';
const projectB = '22222222-2222-2222-2222-222222222222';

const provider: AiModelProvider = {
  id: 'openai:test',
  providerId: 'openai',
  modelId: 'test',
  label: 'Test Agent',
  configured: true,
  capabilities: { streaming: true, tools: false, inputModalities: ['text'] },
  async *stream() {
    yield { type: 'delta', text: 'Saved response.' };
    yield { type: 'done', finishReason: 'stop' };
  },
};

afterEach(() => {
  cleanup();
  localStorage.clear();
});

describe('AiChatPanel project persistence', () => {
  it('restores a project conversation without exposing it to another project', async () => {
    const first = render(<AiChatPanel projectGuid={projectA} provider={provider} />);
    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Project A question' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));
    await waitFor(() => expect(screen.getByText('Saved response.')).toBeInTheDocument());
    fireEvent.click(screen.getByLabelText('Back to conversations'));
    expect(screen.getByRole('button', { name: 'Open conversation Project A question' })).toBeInTheDocument();
    first.unmount();

    const second = render(<AiChatPanel projectGuid={projectB} provider={provider} />);
    expect(screen.queryByRole('button', { name: 'Open conversation Project A question' })).not.toBeInTheDocument();
    second.unmount();

    render(<AiChatPanel projectGuid={projectA} provider={provider} />);
    expect(screen.getByRole('button', { name: 'Open conversation Project A question' })).toBeInTheDocument();
  });

  it('restores persisted active conversation and model UI state', async () => {
    const first = render(<AiChatPanel projectGuid={projectA} provider={provider} />);
    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Resume me' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));
    await waitFor(() => expect(screen.getByText('Saved response.')).toBeInTheDocument());
    first.unmount();

    render(<AiChatPanel projectGuid={projectA} provider={provider} />);
    expect(screen.getByRole('region', { name: 'Active conversation' })).toHaveTextContent('Resume me');
    expect(screen.getByLabelText('Model')).toHaveTextContent('Test Agent');
  });
});

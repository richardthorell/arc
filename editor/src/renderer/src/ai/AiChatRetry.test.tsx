// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import type { AiModelProvider } from './aiChat';
import { AiChatPanel } from './AiChatPanel';

afterEach(cleanup);

describe('AiChatPanel response retry', () => {
  it('retries the failed assistant turn in place and replaces the error with the recovered response', async () => {
    let attempts = 0;
    const provider: AiModelProvider = {
      id: 'openai:test',
      providerId: 'openai',
      modelId: 'test',
      label: 'Test Agent',
      configured: true,
      capabilities: { streaming: true, tools: true, inputModalities: ['text'] },
      async *stream() {
        attempts += 1;
        if (attempts === 1) {
          yield { type: 'error', message: 'Temporary transport failure', code: 'transport', retryable: true };
          return;
        }
        yield { type: 'delta', text: 'Recovered response.' };
        yield { type: 'done', finishReason: 'stop' };
      },
    };

    render(<AiChatPanel provider={provider} />);

    fireEvent.change(screen.getByLabelText('Start a conversation'), { target: { value: 'Create something' } });
    fireEvent.click(screen.getByLabelText('Start conversation'));

    await waitFor(() => expect(screen.getByText('Something went wrong')).toBeVisible());
    expect(screen.getByText("The response couldn't be completed. You can try again.")).toBeVisible();
    expect(screen.getByRole('button', { name: 'Retry' })).toBeVisible();

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));

    await waitFor(() => expect(screen.getByText('Recovered response.')).toBeVisible());
    expect(screen.queryByText('Something went wrong')).not.toBeInTheDocument();
    expect(attempts).toBe(2);
    const userCards = document.querySelectorAll('.ai-chat-user-card');
    expect(userCards).toHaveLength(1);
    expect(userCards[0]).toHaveTextContent('Create something');
  });
});

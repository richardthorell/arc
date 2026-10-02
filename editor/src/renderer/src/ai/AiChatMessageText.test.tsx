// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { renderAiChatMessageText } from './AiChatMessageText';

afterEach(() => {
  vi.restoreAllMocks();
});

describe('renderAiChatMessageText', () => {
  it('renders HTTP links without swallowing trailing sentence punctuation', () => {
    render(<div>{renderAiChatMessageText('Billing: https://platform.openai.com/settings/organization/billing/.')}</div>);

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', 'https://platform.openai.com/settings/organization/billing/');
    expect(link).toHaveAttribute('title', 'Ctrl+click to open link');
    expect(screen.getByText(/\.$/u)).toBeInTheDocument();
  });

  it('opens a link only when Ctrl or Command is held', () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    render(<div>{renderAiChatMessageText('See https://example.com/docs')}</div>);
    const link = screen.getByRole('link');

    fireEvent.click(link);
    expect(open).not.toHaveBeenCalled();

    fireEvent.click(link, { ctrlKey: true });
    expect(open).toHaveBeenCalledWith('https://example.com/docs', '_blank', 'noopener,noreferrer');
  });
});

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { renderAiChatMessageText } from './AiChatMessageText';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('renderAiChatMessageText', () => {
  it('renders common inline Markdown and headings', () => {
    render(
      <div>
        {renderAiChatMessageText('# Result\n\nThis is **bold**, *italic*, ~~old~~, and `inline code`.\nSecond line.')}
      </div>,
    );

    expect(screen.getByRole('heading', { level: 1, name: 'Result' })).toBeInTheDocument();
    expect(screen.getByText('bold')).toHaveProperty('tagName', 'STRONG');
    expect(screen.getByText('italic')).toHaveProperty('tagName', 'EM');
    expect(screen.getByText('old')).toHaveProperty('tagName', 'DEL');
    expect(screen.getByText('inline code')).toHaveProperty('tagName', 'CODE');
    expect(screen.getByText('Second line.')).toBeInTheDocument();
  });

  it('renders lists, blockquotes, fenced code, and tables', () => {
    render(
      <div>
        {renderAiChatMessageText(
          '- one\n- two\n\n> quoted\n\n```ts\nconst value = 42;\n```\n\n| Name | Value |\n| --- | --- |\n| Count | 42 |',
        )}
      </div>,
    );

    const list = screen.getByRole('list');
    expect(within(list).getAllByRole('listitem')).toHaveLength(2);
    expect(screen.getByText('quoted').closest('blockquote')).toBeInTheDocument();

    const blockCode = screen.getByText('const value = 42;');
    expect(blockCode).toHaveAttribute('data-language', 'ts');
    expect(blockCode.closest('pre')).toHaveClass('ai-chat-markdown-code-block');

    const table = screen.getByRole('table');
    expect(within(table).getByRole('columnheader', { name: 'Name' })).toBeInTheDocument();
    expect(within(table).getByRole('cell', { name: '42' })).toBeInTheDocument();
  });

  it('renders HTTP links without swallowing trailing sentence punctuation', () => {
    render(
      <div>{renderAiChatMessageText('Billing: https://platform.openai.com/settings/organization/billing/.')}</div>,
    );

    const link = screen.getByRole('link');
    expect(link).toHaveAttribute('href', 'https://platform.openai.com/settings/organization/billing/');
    expect(link).toHaveAttribute('title', 'Ctrl+click to open link');
    expect(screen.getByText(/\.$/u)).toBeInTheDocument();
  });

  it('renders Markdown links with the existing deliberate external-navigation behavior', () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    render(<div>{renderAiChatMessageText('Read [the docs](https://example.com/docs).')}</div>);
    const link = screen.getByRole('link', { name: 'the docs' });

    fireEvent.click(link);
    expect(open).not.toHaveBeenCalled();

    fireEvent.click(link, { metaKey: true });
    expect(open).toHaveBeenCalledWith('https://example.com/docs', '_blank', 'noopener,noreferrer');
  });

  it('opens a bare link only when Ctrl or Command is held', () => {
    const open = vi.spyOn(window, 'open').mockImplementation(() => null);
    render(<div>{renderAiChatMessageText('See https://example.com/docs')}</div>);
    const link = screen.getByRole('link');

    fireEvent.click(link);
    expect(open).not.toHaveBeenCalled();

    fireEvent.click(link, { ctrlKey: true });
    expect(open).toHaveBeenCalledWith('https://example.com/docs', '_blank', 'noopener,noreferrer');
  });

  it('never executes raw HTML or unsafe Markdown links', () => {
    const { container } = render(
      <div>{renderAiChatMessageText('<script>window.pwned = true</script> [run](javascript:alert(1))')}</div>,
    );

    expect(container.querySelector('script')).not.toBeInTheDocument();
    expect(screen.queryByRole('link')).not.toBeInTheDocument();
    expect(screen.getByText(/<script>window\.pwned = true<\/script>/u)).toBeInTheDocument();
  });

  it('keeps incomplete streaming Markdown readable instead of failing', () => {
    render(<div>{renderAiChatMessageText('Working on **the change\n\n```ts\nconst partial = true;')}</div>);

    expect(screen.getByText('Working on **the change')).toBeInTheDocument();
    expect(screen.getByText('const partial = true;')).toBeInTheDocument();
  });
});

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ConsolePanel } from './ConsolePanel';

const events = [
  { id: 'one', level: 'info' as const, source: 'editor', message: 'First', timestamp: '10:00:00' },
  { id: 'two', level: 'warning' as const, source: 'render', message: 'Second', timestamp: '10:00:01' },
];

afterEach(cleanup);

describe('ConsolePanel', () => {
  it('clears currently displayed events while retaining controls', () => {
    const onClear = vi.fn();
    const { rerender } = render(<ConsolePanel clearedIds={new Set()} events={events} locked
      onClear={onClear} onLockedChange={() => undefined} />);
    fireEvent.click(screen.getByLabelText('Clear console'));
    expect(onClear).toHaveBeenCalledWith(events);

    rerender(<ConsolePanel clearedIds={new Set(events.map((event) => event.id))} events={events} locked
      onClear={onClear} onLockedChange={() => undefined} />);
    expect(screen.getByText(/Console is clear/)).toBeInTheDocument();
    expect(screen.queryByText(/First/)).not.toBeInTheDocument();
  });

  it('follows new messages while locked and manual scrolling disables lock', () => {
    const onLockedChange = vi.fn();
    const { rerender } = render(<ConsolePanel clearedIds={new Set()} events={events.slice(0, 1)} locked
      onClear={() => undefined} onLockedChange={onLockedChange} />);
    const content = screen.getByText(/First/).parentElement!;
    Object.defineProperty(content, 'scrollHeight', { configurable: true, value: 320 });
    rerender(<ConsolePanel clearedIds={new Set()} events={events} locked
      onClear={() => undefined} onLockedChange={onLockedChange} />);
    expect(content.scrollTop).toBe(320);

    fireEvent.wheel(content);
    expect(onLockedChange).toHaveBeenCalledWith(false);
  });

  it('enabling lock is explicit and visually active', () => {
    const onLockedChange = vi.fn();
    render(<ConsolePanel clearedIds={new Set()} events={events} locked={false}
      onClear={() => undefined} onLockedChange={onLockedChange} />);
    fireEvent.click(screen.getByLabelText('Enable console scroll lock'));
    expect(onLockedChange).toHaveBeenCalledWith(true);
  });
});

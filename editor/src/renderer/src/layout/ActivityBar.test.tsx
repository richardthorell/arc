// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { ActivityBar } from './ActivityBar';

afterEach(cleanup);

describe('ActivityBar', () => {
  it('shows global utilities without hierarchy', () => {
    render(<ActivityBar activeActivity="scene" onSelectActivity={vi.fn()} onSettings={vi.fn()} />);

    expect(screen.queryByRole('button', { name: 'Hierarchy' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'AI Gateway' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Version Control' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Settings' })).toBeInTheDocument();
  });

  it('opens a utility and collapses it when clicked again', () => {
    const onSelectActivity = vi.fn();
    const onExpandedChange = vi.fn();
    const { rerender } = render(
      <ActivityBar
        activeActivity="scene"
        expanded={false}
        onExpandedChange={onExpandedChange}
        onSelectActivity={onSelectActivity}
        onSettings={vi.fn()}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(onSelectActivity).toHaveBeenCalledWith('search');
    expect(onExpandedChange).toHaveBeenCalledWith(true);

    rerender(
      <ActivityBar
        activeActivity="search"
        expanded
        onExpandedChange={onExpandedChange}
        onSelectActivity={onSelectActivity}
        onSettings={vi.fn()}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Search' }));
    expect(onExpandedChange).toHaveBeenLastCalledWith(false);
  });

  it('opens settings without treating it as an expandable activity panel', () => {
    const onExpandedChange = vi.fn();
    const onSelectActivity = vi.fn();
    const onSettings = vi.fn();
    render(
      <ActivityBar
        activeActivity="scene"
        expanded={false}
        onExpandedChange={onExpandedChange}
        onSelectActivity={onSelectActivity}
        onSettings={onSettings}
      />,
    );

    const settings = screen.getByRole('button', { name: 'Settings' });
    expect(settings).toHaveAttribute('aria-haspopup', 'dialog');
    fireEvent.click(settings);

    expect(onSettings).toHaveBeenCalledTimes(1);
    expect(onSelectActivity).not.toHaveBeenCalled();
    expect(onExpandedChange).not.toHaveBeenCalled();
  });
});

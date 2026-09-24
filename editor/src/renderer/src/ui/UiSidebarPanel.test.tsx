// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { requestedSettingsDialogKind, resetSettingsDialogRequest } from '../settings/settingsDialogRoute';
import { UiSidebarPanel, UiSidebarPanelButton } from './UiSidebarPanel';

afterEach(() => {
  cleanup();
  resetSettingsDialogRequest();
});

describe('UiSidebarPanelButton', () => {
  it('does not show a counter when the value is zero', () => {
    render(
      <UiSidebarPanelButton aria-label="Repository" counter={0} variant="ghost">
        Repository
      </UiSidebarPanelButton>,
    );

    expect(screen.queryByLabelText('0 unread')).not.toBeInTheDocument();
  });

  it('shows a positive counter with its accessible label', () => {
    render(
      <UiSidebarPanelButton aria-label="Repository" counter={7} counterLabel="7 changed files" variant="ghost">
        Repository
      </UiSidebarPanelButton>,
    );

    expect(screen.getByLabelText('7 changed files')).toHaveTextContent('7');
  });
});

describe('UiSidebarPanel', () => {
  it('shows global utilities without hierarchy', () => {
    render(<UiSidebarPanel activeActivity="scene" onSelectActivity={vi.fn()} onSettings={vi.fn()} />);

    expect(screen.getByRole('complementary', { name: 'Global utilities' })).toHaveClass('ui-sidebar-panel');
    expect(screen.queryByRole('button', { name: 'Hierarchy' })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Search' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'AI Gateway' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Version Control' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Editor Preferences' })).toBeInTheDocument();
  });

  it('opens a utility drawer and collapses it when clicked again', () => {
    const onSelectActivity = vi.fn();
    const onExpandedChange = vi.fn();
    const { rerender } = render(
      <UiSidebarPanel
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
      <UiSidebarPanel
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

  it('opens editor preferences without treating it as a drawer activity', () => {
    const onExpandedChange = vi.fn();
    const onSelectActivity = vi.fn();
    const onSettings = vi.fn();
    render(
      <UiSidebarPanel
        activeActivity="scene"
        expanded={false}
        onExpandedChange={onExpandedChange}
        onSelectActivity={onSelectActivity}
        onSettings={onSettings}
      />,
    );

    const settings = screen.getByRole('button', { name: 'Editor Preferences' });
    expect(settings).toHaveAttribute('aria-haspopup', 'dialog');
    fireEvent.click(settings);

    expect(requestedSettingsDialogKind()).toBe('editorPreferences');
    expect(onSettings).toHaveBeenCalledTimes(1);
    expect(onSelectActivity).not.toHaveBeenCalled();
    expect(onExpandedChange).not.toHaveBeenCalled();
  });
});

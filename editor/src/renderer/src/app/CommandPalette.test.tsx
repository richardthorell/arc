// @vitest-environment jsdom

import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { CommandPalette } from './CommandPalette';
import type { CommandContext } from './workbenchTypes';

const context: CommandContext = {
  editorFocused: true,
  viewportFocused: true,
  textInputFocused: false,
  modalOpen: true,
  playing: false,
  hasSelection: true,
  canUndo: true,
  canRedo: false,
  projectOpen: true,
};

afterEach(cleanup);

describe('CommandPalette', () => {
  it('filters commands and executes the selected result', () => {
    const onCommand = vi.fn();
    const onClose = vi.fn();
    render(
      <CommandPalette
        context={context}
        onClose={onClose}
        onCommand={onCommand}
        shortcut={(command) => (command === 'viewport.frameSelected' ? 'F' : '')}
      />,
    );

    const input = screen.getByLabelText('Search commands');
    fireEvent.change(input, { target: { value: 'frame selected' } });
    expect(screen.getByText('Frame Selected')).toBeInTheDocument();
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(onClose).toHaveBeenCalledOnce();
    expect(onCommand).toHaveBeenCalledWith('viewport.frameSelected');
  });

  it('shows disabled contextual commands without executing them', () => {
    const onCommand = vi.fn();
    render(
      <CommandPalette
        context={{ ...context, canRedo: false }}
        onClose={() => undefined}
        onCommand={onCommand}
        shortcut={() => ''}
      />,
    );
    fireEvent.change(screen.getByLabelText('Search commands'), { target: { value: 'redo' } });
    expect(screen.getByRole('option', { name: /Redo/ })).toBeDisabled();
  });
});

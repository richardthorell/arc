// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { MenuBar } from './MenuBar';

describe('MenuBar', () => {
  it('exposes the editor menu hierarchy', () => {
    render(<MenuBar projectTitle="Scene" onCommand={vi.fn()} />);

    for (const menu of ['File', 'Edit', 'Entity', 'View', 'Window', 'Tools', 'Help']) {
      expect(screen.getByRole('button', { name: menu })).toBeInTheDocument();
    }
  });

  it('keeps existing menu items wired to their commands', () => {
    const onCommand = vi.fn();
    render(<MenuBar projectTitle="Scene" onCommand={onCommand} canUndo canRedo />);

    fireEvent.click(screen.getByRole('button', { name: 'File' }));
    fireEvent.click(screen.getByRole('menuitem', { name: /Open Scene/ }));
    expect(onCommand).toHaveBeenLastCalledWith('file.open');

    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    fireEvent.click(screen.getByRole('menuitem', { name: /Duplicate/ }));
    expect(onCommand).toHaveBeenLastCalledWith('entity.duplicate');

    fireEvent.click(screen.getByRole('button', { name: 'Tools' }));
    fireEvent.click(screen.getByRole('menuitem', { name: /Command Palette/ }));
    expect(onCommand).toHaveBeenLastCalledWith('view.commandPalette');
  });

  it('shows unimplemented menu items disabled', () => {
    render(<MenuBar projectTitle="Scene" onCommand={vi.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Edit' }));
    expect(screen.getByRole('menuitem', { name: /Cut/ })).toBeDisabled();
    expect(screen.getByRole('menuitem', { name: /Rename/ })).toBeDisabled();
  });

  it('exposes the grid as a checked View menu command', () => {
    const toggle = vi.fn();
    render(<MenuBar projectTitle="Scene" onCommand={vi.fn()} gridVisible onToggleGrid={toggle} />);

    fireEvent.click(screen.getByRole('button', { name: 'View' }));
    const grid = screen.getByRole('menuitemcheckbox', { name: /Grid/ });
    expect(grid).toHaveAttribute('aria-checked', 'true');
    fireEvent.click(grid);
    expect(toggle).toHaveBeenCalledOnce();
  });
});

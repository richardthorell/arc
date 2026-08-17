// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { MenuBar } from './MenuBar';

afterEach(cleanup);

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
    expect(grid.querySelector('.menu-entry-check')).toHaveClass('is-checked');
    expect(grid.querySelector('.menu-leading svg')).toBeInTheDocument();
    fireEvent.click(grid);
    expect(toggle).toHaveBeenCalledOnce();
  });

  it('switches top-level menus on hover while a menu is already open', () => {
    render(<MenuBar projectTitle="Scene" onCommand={vi.fn()} canUndo canRedo />);

    const file = screen.getByRole('button', { name: 'File' });
    const edit = screen.getByRole('button', { name: 'Edit' });
    fireEvent.click(file);
    expect(file).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('menuitem', { name: /New Scene/ })).toBeInTheDocument();

    fireEvent.pointerEnter(edit.parentElement as HTMLElement);
    expect(file).toHaveAttribute('aria-expanded', 'false');
    expect(edit).toHaveAttribute('aria-expanded', 'true');
    expect(screen.getByRole('menuitem', { name: /Undo/ })).toBeInTheDocument();
  });

  it('uses the shared dropdown surface and exclusive menu entry slots', () => {
    render(<MenuBar projectTitle="Scene" onCommand={vi.fn()} />);

    fireEvent.click(screen.getByRole('button', { name: 'File' }));
    const newScene = screen.getByRole('menuitem', { name: /New Scene/ });
    const openRecent = screen.getByRole('menuitem', { name: 'Open Recent' });

    expect(newScene.closest('.menu-dropdown')).toHaveClass('menu-bar-dropdown');
    expect(newScene).toHaveClass('menu-entry');
    expect(newScene.querySelector('.menu-leading svg')).toBeInTheDocument();
    expect(newScene.querySelector('.menu-entry-check')).not.toBeInTheDocument();
    expect(newScene.querySelector('.menu-shortcut')).toHaveTextContent('Ctrl+N');
    expect(newScene.querySelector('.menu-submenu-chevron')).not.toBeInTheDocument();

    expect(openRecent.querySelector('.menu-shortcut')).not.toBeInTheDocument();
    expect(openRecent.querySelector('.menu-submenu-chevron')).toBeInTheDocument();
  });
});

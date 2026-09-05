// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiTreeView } from './UiTreeView';

const nodes = [
  {
    id: 'editing',
    label: 'Editing',
    children: [
      { id: 'viewport', label: 'Viewport', keywords: ['renderer', 'camera'] },
      { id: 'navigation', label: 'Navigation' },
    ],
  },
  { id: 'system', label: 'System' },
] as const;

afterEach(cleanup);

describe('UiTreeView', () => {
  it('supports hierarchy, expansion and selection', () => {
    const onSelect = vi.fn();
    render(<UiTreeView ariaLabel="Editor settings" nodes={nodes} onSelect={onSelect} />);

    expect(screen.queryByRole('treeitem', { name: /Viewport/ })).not.toBeInTheDocument();
    fireEvent.doubleClick(screen.getByRole('treeitem', { name: /Editing/ }));
    expect(screen.getByRole('treeitem', { name: /Viewport/ })).toBeInTheDocument();

    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));
    expect(onSelect).toHaveBeenLastCalledWith(expect.objectContaining({ id: 'viewport' }));
  });

  it('filters ancestors and matching descendants', () => {
    render(<UiTreeView ariaLabel="Editor settings" nodes={nodes} query="camera" />);

    expect(screen.getByRole('treeitem', { name: /Editing/ })).toBeInTheDocument();
    expect(screen.getByRole('treeitem', { name: /Viewport/ })).toBeInTheDocument();
    expect(screen.queryByRole('treeitem', { name: /Navigation/ })).not.toBeInTheDocument();
    expect(screen.queryByRole('treeitem', { name: /System/ })).not.toBeInTheDocument();
  });

  it('navigates visible nodes with the keyboard', () => {
    render(<UiTreeView ariaLabel="Editor settings" defaultExpandedIds={['editing']} nodes={nodes} selectedId="editing" />);

    const editing = screen.getByRole('treeitem', { name: /Editing/ });
    editing.focus();
    fireEvent.keyDown(editing, { key: 'ArrowDown' });
    expect(screen.getByRole('treeitem', { name: /Viewport/ })).toHaveFocus();
  });
});

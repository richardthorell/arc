// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
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

  it('navigates visible nodes with the keyboard', async () => {
    render(
      <UiTreeView ariaLabel="Editor settings" defaultExpandedIds={['editing']} nodes={nodes} selectedId="editing" />,
    );

    const editing = screen.getByRole('treeitem', { name: /Editing/ });
    editing.focus();
    fireEvent.keyDown(editing, { key: 'ArrowDown' });
    await waitFor(() => expect(screen.getByRole('treeitem', { name: /Viewport/ })).toHaveFocus());
  });

  it('supports additive multi-selection', () => {
    const onSelectionChange = vi.fn();
    const selectedIds = new Set(['viewport']);
    render(
      <UiTreeView
        ariaLabel="Scene hierarchy"
        defaultExpandedIds={['editing']}
        nodes={nodes}
        onSelectionChange={onSelectionChange}
        selectedIds={selectedIds}
      />,
    );

    fireEvent.click(screen.getByRole('treeitem', { name: /Navigation/ }), { ctrlKey: true });
    const [next] = onSelectionChange.mock.calls.at(-1) ?? [];
    expect([...next]).toEqual(['viewport', 'navigation']);
    expect(screen.getByRole('tree')).toHaveAttribute('aria-multiselectable', 'true');
  });

  it('supports contiguous range selection across visible hierarchy rows', () => {
    const onSelectionChange = vi.fn();
    render(
      <UiTreeView
        ariaLabel="Scene hierarchy"
        defaultExpandedIds={['editing']}
        nodes={nodes}
        onSelectionChange={onSelectionChange}
        selectedIds={new Set()}
      />,
    );

    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));
    fireEvent.click(screen.getByRole('treeitem', { name: /System/ }), { shiftKey: true });
    const [next] = onSelectionChange.mock.calls.at(-1) ?? [];
    expect([...next]).toEqual(['viewport', 'navigation', 'system']);
  });

  it('reparents the full selection when a selected row is dragged', () => {
    const onReparent = vi.fn();
    render(
      <UiTreeView
        ariaLabel="Scene hierarchy"
        defaultExpandedIds={['editing']}
        nodes={nodes}
        onReparent={onReparent}
        selectedIds={new Set(['viewport', 'navigation'])}
      />,
    );

    const dataTransfer = {
      dropEffect: 'none',
      effectAllowed: 'none',
      setData: vi.fn(),
    };
    const viewport = screen.getByRole('treeitem', { name: /Viewport/ });
    const system = screen.getByRole('treeitem', { name: /System/ });
    fireEvent.dragStart(viewport, { dataTransfer });
    fireEvent.dragOver(system, { dataTransfer });
    expect(system).toHaveClass('is-drop-target');
    fireEvent.drop(system, { dataTransfer });

    expect(onReparent).toHaveBeenCalledWith(['viewport', 'navigation'], expect.objectContaining({ id: 'system' }));
    expect(dataTransfer.setData).toHaveBeenCalledWith(
      'application/x-arc-tree-nodes',
      JSON.stringify(['viewport', 'navigation']),
    );
  });

  it('rejects reparenting a node beneath its own descendant', () => {
    const onReparent = vi.fn();
    render(
      <UiTreeView
        ariaLabel="Scene hierarchy"
        defaultExpandedIds={['editing']}
        nodes={nodes}
        onReparent={onReparent}
      />,
    );

    const dataTransfer = {
      dropEffect: 'none',
      effectAllowed: 'none',
      setData: vi.fn(),
    };
    const editing = screen.getByRole('treeitem', { name: /Editing/ });
    const viewport = screen.getByRole('treeitem', { name: /Viewport/ });
    fireEvent.dragStart(editing, { dataTransfer });
    fireEvent.dragOver(viewport, { dataTransfer });
    fireEvent.drop(viewport, { dataTransfer });

    expect(viewport).not.toHaveClass('is-drop-target');
    expect(onReparent).not.toHaveBeenCalled();
  });
});

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { FlowGraphEditor } from './FlowGraphEditor';
import { createDefaultFlowGraph, createFlowNode } from './flowGraphTypes';

const flowState = vi.hoisted(() => ({
  redoFlowGraph: vi.fn(),
  replaceFlowGraph: vi.fn(),
  undoFlowGraph: vi.fn(),
}));

vi.mock('./flowDocumentState', () => flowState);

const document = {
  id: 'flow:test',
  kind: 'flow',
  title: 'Test.arcflow',
  path: 'Content/Test.arcflow',
  assetScope: 'project',
  dirty: false,
  readOnly: false,
} satisfies EditorDocument;

afterEach(cleanup);
beforeEach(() => {
  flowState.redoFlowGraph.mockClear();
  flowState.replaceFlowGraph.mockClear();
  flowState.undoFlowGraph.mockClear();
});

describe('FlowGraphEditor', () => {
  it('renders through the shared graph viewport, wire, and pin primitives', () => {
    const view = render(<FlowGraphEditor document={document} graph={createDefaultFlowGraph()} />);

    expect(screen.getByRole('application', { name: 'Flow graph' })).toBeInTheDocument();
    expect(view.container.querySelector('[data-graph-viewport]')).toBeInTheDocument();
    expect(view.container.querySelector('[data-graph-wires]')).toBeInTheDocument();
    expect(view.container.querySelector('[data-graph-pin-key]')).toBeInTheDocument();
    expect(screen.getByText('Begin Play')).toBeInTheDocument();
  });

  it('adds a Branch node from the Flow palette', () => {
    render(<FlowGraphEditor document={document} graph={createDefaultFlowGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add Flow node' });
    fireEvent.change(within(menu).getByRole('textbox', { name: 'Search Flow nodes' }), {
      target: { value: 'branch' },
    });
    fireEvent.click(within(menu).getByRole('menuitem', { name: /Branch/ }));

    expect(flowState.replaceFlowGraph).toHaveBeenCalledTimes(1);
    const nextGraph = flowState.replaceFlowGraph.mock.calls[0][1];
    expect(nextGraph.nodes.some((node: { type: string }) => node.type === 'branch')).toBe(true);
  });

  it('adds gameplay nodes from the shared Flow palette', () => {
    render(<FlowGraphEditor document={document} graph={createDefaultFlowGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add Flow node' });
    fireEvent.change(within(menu).getByRole('textbox', { name: 'Search Flow nodes' }), {
      target: { value: 'set transform' },
    });
    fireEvent.click(within(menu).getByRole('menuitem', { name: /Set Transform/ }));

    const nextGraph = flowState.replaceFlowGraph.mock.calls[0][1];
    expect(nextGraph.nodes.some((node: { type: string }) => node.type === 'setTransform')).toBe(true);
  });

  it('edits authored literal node values', () => {
    const graph = createDefaultFlowGraph();
    graph.nodes.push(createFlowNode('stringLiteral', [320, 140], { value: 'Player' }));
    render(<FlowGraphEditor document={document} graph={graph} />);

    fireEvent.change(screen.getByRole('textbox', { name: 'String value' }), { target: { value: 'Hero' } });

    expect(flowState.replaceFlowGraph).toHaveBeenCalledTimes(1);
    const nextGraph = flowState.replaceFlowGraph.mock.calls[0][1];
    expect(nextGraph.nodes.find((node: { type: string }) => node.type === 'stringLiteral').values.value).toBe('Hero');
  });

  it('does not mutate a read-only Flow graph', () => {
    render(<FlowGraphEditor document={{ ...document, readOnly: true }} graph={createDefaultFlowGraph()} />);

    expect(screen.getByRole('button', { name: 'Add Node' })).toBeDisabled();
    expect(screen.getByRole('button', { name: 'Delete' })).toBeDisabled();
    fireEvent.wheel(screen.getByRole('application', { name: 'Flow graph' }), {
      clientX: 200,
      clientY: 120,
      deltaY: -1,
    });
    expect(flowState.replaceFlowGraph).not.toHaveBeenCalled();
  });
});

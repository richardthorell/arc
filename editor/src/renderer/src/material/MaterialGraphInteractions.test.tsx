// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { materialConnectionFlowIds, MaterialGraphWithInteractions } from './MaterialGraphInteractions';
import { createDefaultMaterialGraph, createMaterialNode, type MaterialGraph } from './materialGraphTypes';

const materialState = vi.hoisted(() => ({
  redoMaterialGraph: vi.fn(),
  replaceMaterialGraph: vi.fn(),
  saveMaterialDocument: vi.fn(async () => true),
  undoMaterialGraph: vi.fn(),
}));

vi.mock('./materialDocumentState', () => materialState);

const document = { readOnly: false } as EditorDocument;

const createBranchedFlowGraph = (): MaterialGraph => {
  const left = { ...createMaterialNode('constant', [40, 80]), id: 'left' };
  const right = { ...createMaterialNode('constant', [40, 220]), id: 'right' };
  const multiply = { ...createMaterialNode('multiply', [260, 140]), id: 'multiply' };
  const output = { ...createMaterialNode('output', [520, 140]), id: 'material-output' };
  return {
    version: 1,
    nodes: [left, right, multiply, output],
    connections: [
      {
        id: 'left-to-multiply',
        from: { nodeId: left.id, pin: 'value' },
        to: { nodeId: multiply.id, pin: 'a' },
      },
      {
        id: 'right-to-multiply',
        from: { nodeId: right.id, pin: 'value' },
        to: { nodeId: multiply.id, pin: 'b' },
      },
      {
        id: 'multiply-to-output',
        from: { nodeId: multiply.id, pin: 'result' },
        to: { nodeId: output.id, pin: 'roughness' },
      },
    ],
    viewport: { x: 0, y: 0, zoom: 1 },
  };
};

afterEach(cleanup);
beforeEach(() => {
  materialState.redoMaterialGraph.mockClear();
  materialState.replaceMaterialGraph.mockClear();
  materialState.saveMaterialDocument.mockClear();
  materialState.undoMaterialGraph.mockClear();
});

describe('materialConnectionFlowIds', () => {
  it('includes every upstream connection that contributes to the hovered connection', () => {
    expect(materialConnectionFlowIds(createBranchedFlowGraph(), 'multiply-to-output')).toEqual(
      new Set(['multiply-to-output', 'left-to-multiply', 'right-to-multiply']),
    );
  });
});

describe('MaterialGraphWithInteractions', () => {
  it('shows animated flow direction and highlights both endpoint sockets on hover', () => {
    const { container } = render(
      <MaterialGraphWithInteractions document={document} graph={createDefaultMaterialGraph()} />,
    );
    const wire = container.querySelector<SVGPathElement>('.material-wire-hit');
    expect(wire).not.toBeNull();

    fireEvent.pointerEnter(wire!, { clientX: 160, clientY: 120 });

    expect(screen.getByRole('tooltip')).toHaveTextContent('Vector3 • Color.rgb → Base Color');
    expect(container.querySelectorAll('.material-wire-chevron')).toHaveLength(0);
    expect(container.querySelectorAll('.material-wire-flow-texture')).toHaveLength(3);
    expect(wire!.closest('.material-wire-interaction')).toHaveClass('is-flow', 'is-primary');

    const color = screen.getByText('Color', { selector: '.ui-node-card-title' }).closest('article');
    const output = screen.getByText('Material Output').closest('article');
    expect(color).not.toBeNull();
    expect(output).not.toBeNull();
    expect(within(color!).getByRole('button', { name: 'RGB' })).toHaveClass('is-wire-endpoint');
    expect(within(output!).getByRole('button', { name: 'Base Color' })).toHaveClass('is-wire-endpoint');
  });

  it('highlights the full upstream flow when hovering a downstream connection', () => {
    const { container } = render(
      <MaterialGraphWithInteractions document={document} graph={createBranchedFlowGraph()} />,
    );
    const hovered = container.querySelector<SVGPathElement>(
      '[data-material-wire-id="multiply-to-output"] .material-wire-hit',
    );
    expect(hovered).not.toBeNull();

    fireEvent.pointerEnter(hovered!, { clientX: 420, clientY: 160 });

    expect(container.querySelector('[data-material-wire-id="multiply-to-output"]')).toHaveClass(
      'is-flow',
      'is-primary',
    );
    expect(container.querySelector('[data-material-wire-id="left-to-multiply"]')).toHaveClass('is-flow');
    expect(container.querySelector('[data-material-wire-id="right-to-multiply"]')).toHaveClass('is-flow');
  });

  it('shows direction, type and semantic meaning when hovering a socket', () => {
    render(<MaterialGraphWithInteractions document={document} graph={createDefaultMaterialGraph()} />);
    const output = screen.getByText('Material Output').closest('article');
    expect(output).not.toBeNull();
    const baseColor = within(output!).getByRole('button', { name: 'Base Color' });

    fireEvent.pointerEnter(baseColor, { clientX: 480, clientY: 250 });

    expect(screen.getByRole('tooltip')).toHaveTextContent('Input · Vector3 • Base Color — surface albedo');
  });

  it('softly marks compatible sockets and tints incompatible sockets while connecting', () => {
    render(<MaterialGraphWithInteractions document={document} graph={createDefaultMaterialGraph()} />);
    const color = screen.getByText('Color', { selector: '.ui-node-card-title' }).closest('article');
    const output = screen.getByText('Material Output').closest('article');
    expect(color).not.toBeNull();
    expect(output).not.toBeNull();

    const rgb = within(color!).getByRole('button', { name: 'RGB' });
    const baseColor = within(output!).getByRole('button', { name: 'Base Color' });
    const metallic = within(output!).getByRole('button', { name: 'Metallic' });

    fireEvent.pointerDown(rgb, { button: 0 });

    expect(rgb).toHaveClass('is-connection-source');
    expect(baseColor).toHaveClass('is-compatible-target');
    expect(metallic).toHaveClass('is-incompatible-target');
  });
});

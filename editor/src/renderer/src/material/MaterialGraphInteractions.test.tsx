// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { MaterialGraphWithInteractions } from './MaterialGraphInteractions';
import { createDefaultMaterialGraph } from './materialGraphTypes';

const materialState = vi.hoisted(() => ({
  redoMaterialGraph: vi.fn(),
  replaceMaterialGraph: vi.fn(),
  saveMaterialDocument: vi.fn(async () => true),
  undoMaterialGraph: vi.fn(),
}));

vi.mock('./materialDocumentState', () => materialState);

const document = { readOnly: false } as EditorDocument;

afterEach(cleanup);
beforeEach(() => {
  materialState.redoMaterialGraph.mockClear();
  materialState.replaceMaterialGraph.mockClear();
  materialState.saveMaterialDocument.mockClear();
  materialState.undoMaterialGraph.mockClear();
});

describe('MaterialGraphWithInteractions', () => {
  it('shows wire direction and highlights both endpoint sockets on hover', () => {
    const { container } = render(<MaterialGraphWithInteractions document={document} graph={createDefaultMaterialGraph()} />);
    const wire = container.querySelector<SVGPathElement>('.material-wire-hit');
    expect(wire).not.toBeNull();

    fireEvent.pointerEnter(wire!, { clientX: 160, clientY: 120 });

    expect(screen.getByRole('tooltip')).toHaveTextContent('Vector3 • Color.rgb → Base Color');
    expect(container.querySelectorAll('.material-wire-chevron')).toHaveLength(9);

    const color = screen.getByText('Color', { selector: '.ui-node-card-title' }).closest('article');
    const output = screen.getByText('Material Output').closest('article');
    expect(color).not.toBeNull();
    expect(output).not.toBeNull();
    expect(within(color!).getByRole('button', { name: 'RGB' })).toHaveClass('is-wire-endpoint');
    expect(within(output!).getByRole('button', { name: 'Base Color' })).toHaveClass('is-wire-endpoint');
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

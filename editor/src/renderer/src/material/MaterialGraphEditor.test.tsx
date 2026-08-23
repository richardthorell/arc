// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { MaterialGraphEditor } from './MaterialGraphEditor';
import { createDefaultMaterialGraph } from './materialGraphTypes';

const materialState = vi.hoisted(() => ({
  redoMaterialGraph: vi.fn(),
  replaceMaterialGraph: vi.fn(),
  undoMaterialGraph: vi.fn(),
}));

vi.mock('./materialDocumentState', () => materialState);

const document = { readOnly: false } as EditorDocument;

afterEach(cleanup);
beforeEach(() => {
  materialState.redoMaterialGraph.mockClear();
  materialState.replaceMaterialGraph.mockClear();
  materialState.undoMaterialGraph.mockClear();
});

describe('MaterialGraphEditor menus', () => {
  it('keeps Add Node menu scrolling from zooming the graph', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    const item = within(menu).getByRole('menuitem', { name: /Constant/ });

    materialState.replaceMaterialGraph.mockClear();
    fireEvent.wheel(item, { clientX: 80, clientY: 100, deltaY: 120 });
    expect(materialState.replaceMaterialGraph).not.toHaveBeenCalled();

    fireEvent.wheel(screen.getByRole('application', { name: 'Material graph' }), {
      clientX: 300,
      clientY: 220,
      deltaY: 120,
    });
    expect(materialState.replaceMaterialGraph).toHaveBeenCalledTimes(1);
  });

  it('uses the same isolated shared menu for right-click node creation', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);
    const canvas = screen.getByRole('application', { name: 'Material graph' });

    fireEvent.contextMenu(canvas, { clientX: 240, clientY: 180 });
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    expect(menu).toHaveClass('menu-dropdown', 'ui-context-menu');

    materialState.replaceMaterialGraph.mockClear();
    fireEvent.wheel(menu, { clientX: 240, clientY: 200, deltaY: -120 });
    expect(materialState.replaceMaterialGraph).not.toHaveBeenCalled();
  });

  it('renders material graph nodes through the shared node-card base', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    expect(screen.getByText('Material Output').closest('article')).toHaveClass('ui-node-card', 'ui-node-card-accent');
    expect(screen.getByText('Vector 3 / Color').closest('article')).toHaveClass('ui-node-card');
  });
});

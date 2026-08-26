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
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Values/ }));
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Constants/ }));
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

  it('opens material categories and subcategories on hover', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    const math = within(menu).getByRole('menuitem', { name: /Math/ });
    expect(within(menu).queryByRole('menuitem', { name: /Arithmetic/ })).not.toBeInTheDocument();

    fireEvent.mouseEnter(math);
    expect(within(menu).getByRole('menuitem', { name: /Arithmetic/ })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /Trigonometry/ })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /Measurement/ })).toBeInTheDocument();

    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Arithmetic/ }));
    expect(within(menu).getByRole('menuitem', { name: 'Add' })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /Fmod/ })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: /One Minus/ })).toBeInTheDocument();
  });

  it('offers separate RGB and RGBA color nodes under Values', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Values/ }));
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Colors/ }));

    expect(within(menu).getByRole('menuitem', { name: 'Color (RGB)' })).toBeInTheDocument();
    expect(within(menu).getByRole('menuitem', { name: 'Color (RGBA)' })).toBeInTheDocument();
  });

  it('searches across material node subcategories', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    fireEvent.change(within(menu).getByRole('textbox', { name: 'Search material nodes' }), {
      target: { value: 'arctangent2' },
    });

    expect(within(menu).getByRole('menuitem', { name: /Arctangent2/ })).toBeInTheDocument();
    expect(within(menu).getByText('Math / Trigonometry')).toBeInTheDocument();
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

  it('renders the default base color as a dedicated color node with a picker', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    expect(screen.getByText('Material Output').closest('article')).toHaveClass('ui-node-card', 'ui-node-card-accent');
    expect(screen.getByText('Color (RGB)').closest('article')).toHaveClass('ui-node-card');
    expect(screen.getByLabelText('colorRgb color picker')).toHaveAttribute('type', 'color');
  });
});

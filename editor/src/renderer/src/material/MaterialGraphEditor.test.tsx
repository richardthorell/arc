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

describe('MaterialGraphEditor menus', () => {
  it('keeps Add Node menu scrolling from zooming the graph', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Values/ }));
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Constants/ }));
    const constantsMenu = screen.getByRole('menu', { name: 'Constants material nodes' });
    const item = within(constantsMenu).getByRole('menuitem', { name: 'Constant' });

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

  it('opens material categories and subcategories as cascading side menus', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    const math = within(menu).getByRole('menuitem', { name: /Math/ });
    expect(screen.queryByRole('menu', { name: 'Math material node categories' })).not.toBeInTheDocument();

    fireEvent.mouseEnter(math.closest('.material-node-menu-cascade-entry')!);
    const categoryMenu = screen.getByRole('menu', { name: 'Math material node categories' });
    expect(categoryMenu).toHaveClass('material-node-menu-submenu');
    expect(within(menu).getByRole('menuitem', { name: /Values/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Arithmetic/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Trigonometry/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Measurement/ })).toBeInTheDocument();

    const arithmetic = within(categoryMenu).getByRole('menuitem', { name: /Arithmetic/ });
    fireEvent.mouseEnter(arithmetic.closest('.material-node-menu-cascade-entry')!);
    const commandMenu = screen.getByRole('menu', { name: 'Arithmetic material nodes' });
    expect(commandMenu).toHaveClass('material-node-menu-submenu');
    expect(within(commandMenu).getByRole('menuitem', { name: 'Add' })).toBeInTheDocument();
    expect(within(commandMenu).getByRole('menuitem', { name: /Fmod/ })).toBeInTheDocument();
    expect(within(commandMenu).getByRole('menuitem', { name: /One Minus/ })).toBeInTheDocument();
  });

  it('offers the unified Color node under Values', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Values/ }));
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Colors/ }));

    expect(within(menu).getByRole('menuitem', { name: 'Color' })).toBeInTheDocument();
    expect(within(menu).queryByRole('menuitem', { name: 'Color (RGB)' })).not.toBeInTheDocument();
    expect(within(menu).queryByRole('menuitem', { name: 'Color (RGBA)' })).not.toBeInTheDocument();
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
    expect(screen.getByText('Color', { selector: '.ui-node-card-title' }).closest('article')).toHaveClass(
      'ui-node-card',
      'material-graph-node-colorRgba',
    );
    expect(screen.getByRole('button', { name: 'Open Color color picker' })).toBeEnabled();
  });
});

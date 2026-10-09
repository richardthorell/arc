// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import type { EditorDocument } from '../editors/editorTypes';
import { MaterialGraphEditor } from './MaterialGraphEditor';
import { createDefaultMaterialGraph, createMaterialNode } from './materialGraphTypes';

const materialState = vi.hoisted(() => ({
  redoMaterialGraph: vi.fn(),
  replaceMaterialGraph: vi.fn(),
  replaceMaterialGraphViewport: vi.fn(),
  saveMaterialDocument: vi.fn(async () => true),
  undoMaterialGraph: vi.fn(),
}));

vi.mock('./materialDocumentState', () => materialState);

const document = { readOnly: false } as EditorDocument;

afterEach(cleanup);
beforeEach(() => {
  materialState.redoMaterialGraph.mockClear();
  materialState.replaceMaterialGraph.mockClear();
  materialState.replaceMaterialGraphViewport.mockClear();
  materialState.saveMaterialDocument.mockClear();
  materialState.undoMaterialGraph.mockClear();
});

describe('MaterialGraphEditor', () => {
  it('renders canvas mechanics through the shared graph primitives', () => {
    const { container } = render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    expect(container.querySelector('[data-graph-viewport]')).toHaveClass('material-graph-transform');
    expect(container.querySelector('[data-graph-wires]')).toHaveClass('material-graph-wires');
    expect(container.querySelectorAll('[data-graph-pin-key]').length).toBeGreaterThan(0);
  });

  it('renders authored material graph groups behind their nodes', () => {
    const graph = createDefaultMaterialGraph();
    const roughness = graph.nodes.find((node) => node.parameter?.name === 'Roughness')!;
    graph.groups = [{ id: 'surface', name: 'Surface', nodeIds: [roughness.id], order: 10 }];

    const { container } = render(<MaterialGraphEditor document={document} graph={graph} />);

    const group = container.querySelector('.material-graph-group');
    expect(group).not.toBeNull();
    expect(group).toHaveTextContent('Surface');
    expect(group).toHaveStyle({ position: 'absolute' });
  });

  it('moves member nodes with a dragged material group', async () => {
    const graph = createDefaultMaterialGraph();
    const roughness = graph.nodes.find((node) => node.parameter?.name === 'Roughness')!;
    graph.groups = [{ id: 'surface', name: 'Surface', nodeIds: [roughness.id], position: [40, 40], size: [500, 320] }];
    const onGraphChange = vi.fn();

    render(<MaterialGraphEditor document={document} graph={graph} onGraphChange={onGraphChange} />);

    const origin = [...roughness.position] as [number, number];
    fireEvent.pointerDown(screen.getByRole('button', { name: 'Surface' }), {
      button: 0,
      clientX: 100,
      clientY: 100,
    });
    fireEvent.pointerMove(window, { clientX: 140, clientY: 160 });

    await waitFor(() => expect(onGraphChange).toHaveBeenCalled());
    const nextGraph = onGraphChange.mock.calls.at(-1)![0];
    const nextRoughness = nextGraph.nodes.find((node: { id: string }) => node.id === roughness.id);
    const nextGroup = nextGraph.groups.find((group: { id: string }) => group.id === 'surface');

    expect(nextRoughness.position[0] - origin[0]).toBe(nextGroup.position[0] - 40);
    expect(nextRoughness.position[1] - origin[1]).toBe(nextGroup.position[1] - 40);
  });

  it('renders shared diagnostic details and focuses the affected node', () => {
    const graph = createDefaultMaterialGraph();
    const roughness = graph.nodes.find((node) => node.parameter?.name === 'Roughness');
    expect(roughness).toBeDefined();

    const { container } = render(
      <MaterialGraphEditor
        diagnostics={[
          {
            id: 'range-warning',
            severity: 'warning',
            message: 'Roughness can produce 0..1.4, outside the expected 0..1 range',
            target: { kind: 'node', nodeId: roughness!.id },
          },
        ]}
        document={document}
        graph={graph}
      />,
    );

    const node = container.querySelector<HTMLElement>(`[data-node-id="${roughness!.id}"]`);
    expect(node).not.toBeNull();
    const badge = within(node!).getByRole('button', { name: 'Warning: 1 graph diagnostic' });
    expect(within(node!).getByRole('tooltip')).toHaveTextContent('Roughness can produce 0..1.4');

    fireEvent.click(badge);
    expect(node).toHaveClass('is-selected');
  });

  it('applies toolbar-controlled graph view options', () => {
    const graph = createDefaultMaterialGraph();
    const isolated = createMaterialNode('normalMap', [980, 720]);
    graph.nodes.push(isolated);

    const { container } = render(
      <MaterialGraphEditor document={document} graph={graph} showGrid={false} dimUnrelated />,
    );

    const canvas = screen.getByRole('application', { name: 'Material graph' });
    expect(canvas).toHaveClass('hide-grid', 'dim-unrelated');

    const output = screen.getByText('Material Output').closest('article');
    expect(output).not.toBeNull();
    fireEvent.pointerDown(output!, { button: 0 });

    const isolatedNode = container.querySelector<HTMLElement>(`[data-node-id="${isolated.id}"]`);
    expect(isolatedNode).toHaveClass('is-unrelated');
    expect(container.querySelectorAll('.material-graph-node.is-unrelated').length).toBeGreaterThan(0);
  });

  it('shows graph navigation controls with snap enabled by default', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    expect(screen.getByRole('button', { name: /Frame All/ })).toBeEnabled();
    expect(screen.getByRole('button', { name: /Arrange/ })).toBeEnabled();
    expect(screen.getByRole('slider', { name: 'Material graph zoom' })).toHaveValue('85');

    const snap = screen.getByRole('button', { name: /Snap/ });
    expect(snap).toHaveAttribute('aria-pressed', 'true');
    fireEvent.click(snap);
    expect(snap).toHaveAttribute('aria-pressed', 'false');
  });

  it('keeps Add Node menu scrolling from zooming the graph', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Values/ }));
    const valuesMenu = screen.getByRole('menu', { name: 'Values material node categories' });
    fireEvent.mouseEnter(within(valuesMenu).getByRole('menuitem', { name: /Constants/ }));
    const constantsMenu = screen.getByRole('menu', { name: 'Constants material nodes' });
    const item = within(constantsMenu).getByRole('menuitem', { name: 'Scalar' });

    materialState.replaceMaterialGraphViewport.mockClear();
    fireEvent.wheel(item, { clientX: 80, clientY: 100, deltaY: 120 });
    expect(materialState.replaceMaterialGraphViewport).not.toHaveBeenCalled();

    fireEvent.wheel(screen.getByRole('application', { name: 'Material graph' }), {
      clientX: 300,
      clientY: 220,
      deltaY: 120,
    });
    expect(materialState.replaceMaterialGraphViewport).toHaveBeenCalledTimes(1);
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
    expect(categoryMenu).toHaveStyle({ position: 'fixed' });
    expect(menu).not.toContainElement(categoryMenu);
    expect(within(menu).getByRole('menuitem', { name: /Values/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Arithmetic/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Trigonometry/ })).toBeInTheDocument();
    expect(within(categoryMenu).getByRole('menuitem', { name: /Measurement/ })).toBeInTheDocument();

    const arithmetic = within(categoryMenu).getByRole('menuitem', { name: /Arithmetic/ });
    fireEvent.mouseEnter(arithmetic.closest('.material-node-menu-cascade-entry')!);
    const commandMenu = screen.getByRole('menu', { name: 'Arithmetic material nodes' });
    expect(commandMenu).toHaveClass('material-node-menu-submenu');
    expect(commandMenu).toHaveStyle({ position: 'fixed' });
    expect(categoryMenu).not.toContainElement(commandMenu);
    expect(within(commandMenu).getByRole('menuitem', { name: 'Add' })).toBeInTheDocument();
    expect(within(commandMenu).getByRole('menuitem', { name: /Fmod/ })).toBeInTheDocument();
    expect(within(commandMenu).getByRole('menuitem', { name: /One Minus/ })).toBeInTheDocument();
  });

  it('offers the unified Color node under Values', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    fireEvent.click(screen.getByRole('button', { name: 'Add Node' }));
    const menu = screen.getByRole('menu', { name: 'Add material node' });
    fireEvent.mouseEnter(within(menu).getByRole('menuitem', { name: /Values/ }));
    const valuesMenu = screen.getByRole('menu', { name: 'Values material node categories' });
    fireEvent.mouseEnter(within(valuesMenu).getByRole('menuitem', { name: /Colors/ }));
    const colorsMenu = screen.getByRole('menu', { name: 'Colors material nodes' });

    expect(within(colorsMenu).getByRole('menuitem', { name: 'Color' })).toBeInTheDocument();
    expect(within(colorsMenu).queryByRole('menuitem', { name: 'Color (RGB)' })).not.toBeInTheDocument();
    expect(within(colorsMenu).queryByRole('menuitem', { name: 'Color (RGBA)' })).not.toBeInTheDocument();
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

  it('shows a slider for ranged Scalars and keeps unrestricted Scalars numeric-only', () => {
    const graph = createDefaultMaterialGraph();
    const ranged = createMaterialNode('constant', [900, 500], { value: 0.4, min: 0, max: 1 });
    const unrestricted = createMaterialNode('constant', [900, 650], { value: 2 });
    graph.nodes.push(ranged, unrestricted);

    const { container } = render(<MaterialGraphEditor document={document} graph={graph} />);

    const rangedNode = container.querySelector<HTMLElement>(`[data-node-id="${ranged.id}"]`);
    expect(rangedNode).not.toBeNull();
    expect(within(rangedNode!).getByRole('slider', { name: 'Scalar range value' })).toHaveValue('0.4');
    expect(within(rangedNode!).getByRole('spinbutton', { name: 'Scalar minimum' })).toHaveValue(0);
    expect(within(rangedNode!).getByRole('spinbutton', { name: 'Scalar maximum' })).toHaveValue(1);

    const unrestrictedNode = container.querySelector<HTMLElement>(`[data-node-id="${unrestricted.id}"]`);
    expect(unrestrictedNode).not.toBeNull();
    expect(within(unrestrictedNode!).queryByRole('slider', { name: 'Scalar range value' })).not.toBeInTheDocument();
  });

  it('enabling a Scalar range creates hard 0..1 bounds and clamps the authored value', () => {
    const graph = createDefaultMaterialGraph();
    const scalar = createMaterialNode('constant', [900, 500], { value: 2 });
    graph.nodes.push(scalar);

    const { container } = render(<MaterialGraphEditor document={document} graph={graph} />);
    const scalarNode = container.querySelector<HTMLElement>(`[data-node-id="${scalar.id}"]`);
    expect(scalarNode).not.toBeNull();

    fireEvent.click(within(scalarNode!).getByRole('switch', { name: /Range/ }));
    expect(materialState.replaceMaterialGraph).toHaveBeenCalled();
    const nextGraph = materialState.replaceMaterialGraph.mock.calls.at(-1)![1];
    const nextScalar = nextGraph.nodes.find((node: { id: string }) => node.id === scalar.id);
    expect(nextScalar.values).toMatchObject({ value: 1, min: 0, max: 1 });
  });

  it('uses the shared parameter control for editable value nodes', () => {
    const graph = createDefaultMaterialGraph();
    const scalar = createMaterialNode('constant', [900, 500], { value: 0.5 });
    graph.nodes.push(scalar);

    const { container } = render(<MaterialGraphEditor document={document} graph={graph} />);
    const scalarNode = container.querySelector<HTMLElement>(`[data-node-id="${scalar.id}"]`);
    expect(scalarNode).not.toBeNull();

    const parameter = within(scalarNode!).getByRole('switch', { name: 'Parameter' });
    expect(parameter).toHaveClass('ui-toggle-button');
    expect(within(scalarNode!).queryByRole('checkbox')).not.toBeInTheDocument();

    fireEvent.click(parameter);
    expect(materialState.replaceMaterialGraph).toHaveBeenCalled();
  });

  it('renders the default base color as a dedicated color node with a picker', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    expect(screen.getByText('Material Output').closest('article')).toHaveClass('ui-node-card', 'ui-node-card-accent');
    expect(screen.getAllByText('Color', { selector: '.ui-node-card-title' })[0]!.closest('article')).toHaveClass(
      'ui-node-card',
      'material-graph-node-colorRgba',
    );
    const baseColor = screen.getAllByText('Color', { selector: '.ui-node-card-title' })[0]!.closest('article');
    expect(baseColor).not.toBeNull();
    expect(within(baseColor!).getByRole('button', { name: 'Open Color color picker' })).toBeEnabled();
  });

  it('adds compatible Material Function references through the Function Call list picker', async () => {
    const defaultPath = 'Content/functions/default.arcmatfn';
    const checkerPath = 'Content/functions/checker.arcmatfn';
    const functionDocument = (name: string) =>
      JSON.stringify({
        kind: 'materialFunction',
        version: 1,
        name,
        inputs: [],
        outputs: [{ id: 'color', name: 'Color', type: 'vec3' }],
        graph: { version: 1, nodes: [], connections: [] },
      });
    Object.defineProperty(window, 'arc', {
      configurable: true,
      value: {
        host: {
          query: vi.fn(async () => ({
            succeeded: true,
            payload: {
              assets: [
                {
                  guid: 'default-guid',
                  path: defaultPath,
                  sourcePath: defaultPath,
                  scope: 'project',
                  readOnly: false,
                  kind: 'materialFunction',
                  state: 'ready',
                },
                {
                  guid: 'checker-guid',
                  path: checkerPath,
                  sourcePath: checkerPath,
                  scope: 'project',
                  readOnly: false,
                  kind: 'materialFunction',
                  state: 'ready',
                },
              ],
            },
          })),
        },
        projects: {
          readText: vi.fn(async (path: string) => ({
            text: path === checkerPath ? functionDocument('Checker') : functionDocument('Default Base Color'),
          })),
        },
      },
    });

    const graph = createDefaultMaterialGraph();
    const call = createMaterialNode('functionCall', [720, 160], {
      name: 'Base Color Source',
      path: defaultPath,
      functions: [{ path: defaultPath }],
      inputPins: [],
      outputPins: [{ id: 'color', name: 'Color', type: 'vec3' }],
    });
    graph.nodes.push(call);

    const { container } = render(<MaterialGraphEditor document={document} graph={graph} />);
    const callNode = container.querySelector<HTMLElement>(`[data-node-id="${call.id}"]`);
    expect(callNode).not.toBeNull();

    const add = await within(callNode!).findByRole('button', { name: 'Add Material Function' });
    fireEvent.click(add);
    fireEvent.click(await screen.findByRole('button', { name: 'Select Checker' }));

    await waitFor(() => expect(materialState.replaceMaterialGraph).toHaveBeenCalled());
    const nextGraph = materialState.replaceMaterialGraph.mock.calls.at(-1)![1];
    const nextCall = nextGraph.nodes.find((node: { id: string }) => node.id === call.id);
    expect(nextCall.values.functions).toEqual([{ path: defaultPath }, { path: checkerPath }]);
    expect(nextCall.values.path).toBe(defaultPath);
  });

  it('uses the material graph domain to protect the output node from deletion', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);
    const output = screen.getByText('Material Output').closest('article');
    expect(output).not.toBeNull();

    fireEvent.pointerDown(output!, { button: 0 });
    fireEvent.click(screen.getByRole('button', { name: 'Delete' }));

    expect(materialState.replaceMaterialGraph).toHaveBeenCalledTimes(1);
    const nextGraph = materialState.replaceMaterialGraph.mock.calls[0][1];
    expect(nextGraph.nodes.some((node: { type: string }) => node.type === 'output')).toBe(true);
  });

  it('rejects and flashes a self connection', () => {
    const graph = createDefaultMaterialGraph();
    const selfConnected = createMaterialNode('normalMap', [320, 520]);
    graph.nodes.push(selfConnected);
    const { container } = render(<MaterialGraphEditor document={document} graph={graph} />);

    const normalMap = container.querySelector<HTMLElement>(`[data-node-id="${selfConnected.id}"]`);
    expect(normalMap).not.toBeNull();
    fireEvent.pointerDown(within(normalMap!).getByRole('button', { name: 'Normal' }), { button: 0 });
    materialState.replaceMaterialGraph.mockClear();
    fireEvent.pointerDown(within(normalMap!).getByRole('button', { name: 'Texture RGB' }), { button: 0 });

    expect(materialState.replaceMaterialGraph).not.toHaveBeenCalled();
    expect(normalMap).toHaveClass('is-connection-invalid');
  });

  it('rejects and flashes an incompatible pin type', () => {
    render(<MaterialGraphEditor document={document} graph={createDefaultMaterialGraph()} />);

    const color = screen.getAllByText('Color', { selector: '.ui-node-card-title' })[0]!.closest('article');
    const output = screen.getByText('Material Output').closest('article');
    expect(color).not.toBeNull();
    expect(output).not.toBeNull();

    fireEvent.pointerDown(within(color!).getByRole('button', { name: 'RGBA' }), { button: 0 });
    materialState.replaceMaterialGraph.mockClear();
    fireEvent.pointerDown(within(output!).getByRole('button', { name: 'Base Color' }), { button: 0 });

    expect(materialState.replaceMaterialGraph).not.toHaveBeenCalled();
    expect(output).toHaveClass('is-connection-invalid');
  });
});

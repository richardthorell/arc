// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { GraphPin, GraphSelectionBox, GraphViewportLayer, GraphWireLayer } from './GraphPrimitives';

afterEach(cleanup);

describe('shared graph primitives', () => {
  it('renders graph-domain-neutral pins and wires', () => {
    const { container } = render(
      <GraphViewportLayer viewport={{ x: 12, y: 24, zoom: 1.5 }}>
        <GraphWireLayer pendingPath="M 0 0 L 20 20" wires={[{ id: 'flow', path: 'M 1 2 L 3 4' }]} />
        <GraphPin connected direction="output" label="Then" pinKey="begin:output:then" />
      </GraphViewportLayer>,
    );

    expect(container.querySelector('[data-graph-viewport]')).toHaveStyle('transform: translate(12px, 24px) scale(1.5)');
    expect(container.querySelector('[data-graph-viewport]')).toHaveAttribute('data-graph-zoom', '1.5');
    expect(container.querySelector('[data-graph-wires] path:not(.pending)')).toHaveAttribute('d', 'M 1 2 L 3 4');
    expect(screen.getByRole('button', { name: 'Then' })).toHaveAttribute('data-graph-pin-key', 'begin:output:then');
    expect(screen.getByRole('button', { name: 'Then' })).toHaveClass('output', 'connected');
  });

  it('exposes shared wire hover metadata and endpoint identity', () => {
    const onHoveredWireChange = vi.fn();
    const { container } = render(
      <GraphWireLayer
        onHoveredWireChange={onHoveredWireChange}
        wires={[
          {
            destinationPinKey: 'surface:input:baseColor',
            id: 'color-wire',
            path: 'M 1 2 L 3 4',
            sourcePinKey: 'color:output:rgb',
            tooltip: 'Vector3 • Color.rgb → Base Color',
          },
        ]}
      />,
    );

    const wire = container.querySelector<SVGPathElement>('[data-graph-wire-id="color-wire"]');
    expect(wire).toHaveAttribute('data-source-pin-key', 'color:output:rgb');
    expect(wire).toHaveAttribute('data-destination-pin-key', 'surface:input:baseColor');
    expect(wire?.querySelector('title')).toHaveTextContent('Vector3 • Color.rgb → Base Color');

    fireEvent.pointerEnter(wire!);
    expect(wire).toHaveClass('is-hovered');
    expect(wire).toHaveAttribute('data-hovered', 'true');
    expect(onHoveredWireChange).toHaveBeenLastCalledWith(expect.objectContaining({ id: 'color-wire' }));

    fireEvent.pointerLeave(wire!);
    expect(onHoveredWireChange).toHaveBeenLastCalledWith(null);
  });

  it('standardizes pin compatibility, highlighting, and semantic tooltips', () => {
    render(
      <GraphPin
        compatibility="compatible"
        direction="input"
        highlighted
        label="Base Color"
        pinKey="surface:input:baseColor"
        semanticDescription="surface albedo"
        typeLabel="Vector3"
      />,
    );

    const pin = screen.getByRole('button', { name: 'Base Color' });
    expect(pin).toHaveClass('input', 'is-highlighted', 'is-compatible-target');
    expect(pin).toHaveAttribute('data-graph-pin-compatibility', 'compatible');
    expect(pin).toHaveAttribute('data-graph-pin-direction', 'input');
    expect(pin).toHaveAttribute('title', 'Input · Vector3 • Base Color — surface albedo');
  });

  it('renders a reusable selection rectangle', () => {
    const { container } = render(<GraphSelectionBox rect={{ left: 10, top: 20, width: 30, height: 40 }} />);
    expect(container.querySelector('[data-graph-selection-box]')).toHaveStyle({
      left: '10px',
      top: '20px',
      width: '30px',
      height: '40px',
    });
  });
});

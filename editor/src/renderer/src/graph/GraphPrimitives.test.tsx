// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

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

    expect(container.querySelector('[data-graph-viewport]')).toHaveStyle(
      'transform: translate(12px, 24px) scale(1.5)',
    );
    expect(container.querySelector('[data-graph-wires] path:not(.pending)')).toHaveAttribute('d', 'M 1 2 L 3 4');
    expect(screen.getByRole('button', { name: 'Then' })).toHaveAttribute('data-graph-pin-key', 'begin:output:then');
    expect(screen.getByRole('button', { name: 'Then' })).toHaveClass('output', 'connected');
  });

  it('renders a reusable selection rectangle', () => {
    const { container } = render(
      <GraphSelectionBox rect={{ left: 10, top: 20, width: 30, height: 40 }} />,
    );
    expect(container.querySelector('[data-graph-selection-box]')).toHaveStyle({
      left: '10px',
      top: '20px',
      width: '30px',
      height: '40px',
    });
  });
});

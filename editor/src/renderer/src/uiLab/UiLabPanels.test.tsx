// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { panelRegistry } from '../app/panelRegistry';
import { UiLabPanels } from './UiLabPanels';
import { UiLabWindow } from './UiLabWindow';

describe('UiLabPanels', () => {
  it('renders every registered editor panel', () => {
    const { container } = render(<UiLabPanels />);
    const previews = Array.from(container.querySelectorAll<HTMLElement>('[data-panel-id]'));
    const ids = previews.map((preview) => preview.dataset.panelId);

    expect(previews).toHaveLength(Object.keys(panelRegistry).length);
    expect(ids).toEqual(expect.arrayContaining(Object.keys(panelRegistry)));
    expect(screen.getByText('16 registered panels')).toBeInTheDocument();
  });

  it('includes the primary panel iteration targets', () => {
    const { container } = render(<UiLabPanels />);

    expect(container.querySelector('[data-panel-id="viewport"]')).toBeInTheDocument();
    expect(container.querySelector('[data-panel-id="hierarchy"]')).toBeInTheDocument();
    expect(container.querySelector('[data-panel-id="inspector"]')).toBeInTheDocument();
    expect(screen.getByLabelText('Search hierarchy')).toBeInTheDocument();
    expect(screen.getByLabelText('Viewport projection')).toBeInTheDocument();
  });
});

describe('UiLabWindow pages', () => {
  it('switches between control and panel pages from the top-level tabs', () => {
    render(<UiLabWindow />);

    expect(screen.getByText('ARC UI Lab')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Panels' }));
    expect(screen.getByText('Panel Lab')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Controls' }));
    expect(screen.getByText('ARC UI Lab')).toBeInTheDocument();
  });
});

// @vitest-environment jsdom
import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { UiPanelCardRow } from './UiPanelCardRow';

describe('UiPanelCardRow', () => {
  it('renders a reusable label/control row without injecting separator elements', () => {
    const { container } = render(
      <div>
        <UiPanelCardRow label="Domain">
          <button type="button">Surface</button>
        </UiPanelCardRow>
        <UiPanelCardRow
          controlClassName="is-end"
          description="Render both sides of the surface."
          label="Two Sided"
        >
          <button type="button">Off</button>
        </UiPanelCardRow>
      </div>,
    );

    expect(screen.getByText('Domain')).toBeTruthy();
    expect(screen.getByText('Two Sided')).toBeTruthy();
    expect(screen.getByText('Render both sides of the surface.')).toBeTruthy();
    expect(container.querySelector('.ui-panel-card-row-control.is-end')).toBeTruthy();
    expect(container.querySelectorAll('.ui-panel-card-row')).toHaveLength(2);
    expect(container.querySelector('hr')).toBeNull();
  });
});

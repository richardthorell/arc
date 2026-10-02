import { render, screen } from '@testing-library/react';
import { describe, expect, it } from 'vitest';

import { UiViewportNavigation } from './UiViewportNavigation';

describe('UiViewportNavigation', () => {
  it('provides a named toolbar landmark on the shared floating surface', () => {
    render(
      <UiViewportNavigation ariaLabel="Viewport navigation">
        <button type="button">Frame selection</button>
      </UiViewportNavigation>,
    );

    const navigation = screen.getByRole('toolbar', { name: 'Viewport navigation' });
    expect(navigation).toHaveClass('ui-floating-surface', 'ui-viewport-navigation');
    expect(screen.getByRole('button', { name: 'Frame selection' })).toBeInTheDocument();
  });

  it('preserves consumer classes', () => {
    render(
      <UiViewportNavigation ariaLabel="Preview navigation" className="preview-navigation">
        <span>controls</span>
      </UiViewportNavigation>,
    );

    expect(screen.getByRole('toolbar', { name: 'Preview navigation' })).toHaveClass('preview-navigation');
  });
});

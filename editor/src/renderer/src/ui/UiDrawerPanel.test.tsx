// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it } from 'vitest';

import { UiDrawerPanel } from './UiDrawerPanel';

afterEach(cleanup);

describe('UiDrawerPanel', () => {
  it('renders the shared drawer surface with caller classes and attributes', () => {
    render(
      <UiDrawerPanel aria-label="Search drawer" className="search-drawer">
        Search content
      </UiDrawerPanel>,
    );

    const drawer = screen.getByLabelText('Search drawer');
    expect(drawer).toHaveClass('ui-panel', 'ui-drawer-panel', 'search-drawer');
    expect(drawer).toHaveTextContent('Search content');
  });
});

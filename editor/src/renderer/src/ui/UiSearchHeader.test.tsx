// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiSearchHeader } from './UiSearchHeader';

afterEach(cleanup);

const modes = [
  { id: 'assets', label: 'Assets', count: 12 },
  { id: 'commands', label: 'Commands', count: 4 },
];

describe('UiSearchHeader', () => {
  it('renders the active scope, counts, and query field', () => {
    render(
      <UiSearchHeader
        mode="assets"
        modes={modes}
        query="wood"
        resultCount={3}
        onModeChange={() => undefined}
        onQueryChange={() => undefined}
      />,
    );

    expect(screen.getByText('Search')).toBeInTheDocument();
    expect(screen.getByText('3 results')).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /Assets/ })).toHaveAttribute('aria-selected', 'true');
    expect(screen.getByRole('searchbox', { name: 'Search' })).toHaveValue('wood');
  });

  it('forwards scope and query changes', () => {
    const onModeChange = vi.fn();
    const onQueryChange = vi.fn();
    render(
      <UiSearchHeader
        mode="assets"
        modes={modes}
        query=""
        onModeChange={onModeChange}
        onQueryChange={onQueryChange}
      />,
    );

    fireEvent.click(screen.getByRole('tab', { name: /Commands/ }));
    fireEvent.change(screen.getByRole('searchbox', { name: 'Search' }), { target: { value: 'save' } });

    expect(onModeChange).toHaveBeenCalledWith('commands');
    expect(onQueryChange).toHaveBeenCalledWith('save');
  });
});

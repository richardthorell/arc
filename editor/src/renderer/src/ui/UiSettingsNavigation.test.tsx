// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiSettingsNavigation } from './UiSettingsNavigation';

const nodes = [
  { id: 'general', label: 'General' },
  { id: 'editing', label: 'Editing', children: [{ id: 'editing.viewport', label: 'Viewport' }] },
] as const;

afterEach(cleanup);

describe('UiSettingsNavigation', () => {
  it('composes settings search and tree selection', () => {
    const onQueryChange = vi.fn();
    const onSelect = vi.fn();

    render(
      <UiSettingsNavigation
        defaultExpandedIds={['editing']}
        nodes={nodes}
        onQueryChange={onQueryChange}
        onSelect={onSelect}
        query=""
        selectedId="general"
      />,
    );

    fireEvent.change(screen.getByRole('searchbox', { name: 'Search settings' }), { target: { value: 'view' } });
    expect(onQueryChange).toHaveBeenCalledWith('view');

    fireEvent.click(screen.getByRole('treeitem', { name: /Viewport/ }));
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ id: 'editing.viewport' }));
  });
});

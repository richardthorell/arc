// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiSettingsNavigation } from './UiSettingsNavigation';

const nodes = [
  { id: 'general', label: 'General' },
  { id: 'editing', label: 'Editing', children: [{ id: 'editing.viewport', label: 'Viewport' }] },
  { id: 'ai', label: 'AI', children: [{ id: 'ai.providers', label: 'Providers' }] },
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

  it('accepts a settings navigation request from elsewhere in the editor', () => {
    const onQueryChange = vi.fn();
    const onSelect = vi.fn();

    render(
      <UiSettingsNavigation
        defaultExpandedIds={['ai']}
        nodes={nodes}
        onQueryChange={onQueryChange}
        onSelect={onSelect}
        query="provider"
        selectedId="general"
      />,
    );

    window.dispatchEvent(new CustomEvent('arc-settings-navigate', { detail: { id: 'ai.providers' } }));

    expect(onQueryChange).toHaveBeenCalledWith('');
    expect(onSelect).toHaveBeenCalledWith(expect.objectContaining({ id: 'ai.providers' }));
  });
});

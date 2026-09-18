// @vitest-environment jsdom
import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { UiPanelCard } from './UiPanelCard';

describe('UiPanelCard', () => {
  it('shares the component-card structure and supports controlled collapsing', () => {
    const onToggle = vi.fn();
    const { rerender } = render(
      <UiPanelCard title="Material" onToggle={onToggle}>
        <span>Settings</span>
      </UiPanelCard>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Collapse Material' }));
    expect(onToggle).toHaveBeenCalledTimes(1);
    expect(screen.getByText('Settings')).toBeTruthy();

    rerender(
      <UiPanelCard collapsed title="Material" onToggle={onToggle}>
        <span>Settings</span>
      </UiPanelCard>,
    );

    expect(screen.queryByText('Settings')).toBeNull();
    expect(screen.getByRole('button', { name: 'Expand Material' })).toBeTruthy();
  });
});

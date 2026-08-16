// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { fireEvent, render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';

import { UiTab, UiTabs } from './UiTabs';

describe('UiTabs', () => {
  it('renders a stronger active state with an optional leading icon', () => {
    render(
      <UiTabs>
        <UiTab active icon={<span data-testid="tab-icon">G</span>}>
          General
        </UiTab>
        <UiTab>Rendering</UiTab>
      </UiTabs>,
    );

    expect(screen.getByRole('button', { name: 'General' })).toHaveClass('is-active');
    expect(screen.getByTestId('tab-icon')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Rendering' })).not.toHaveClass('is-active');
  });

  it('renders an optional close action without activating the tab', () => {
    const onSelect = vi.fn();
    const onClose = vi.fn();
    render(
      <UiTabs>
        <UiTab active onClick={onSelect} onClose={onClose}>
          General
        </UiTab>
      </UiTabs>,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Close General' }));

    expect(onClose).toHaveBeenCalledTimes(1);
    expect(onSelect).not.toHaveBeenCalled();
  });
});

// @vitest-environment jsdom
import '@testing-library/jest-dom/vitest';

import { cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { Hammer, RefreshCw } from 'lucide-react';
import { afterEach, describe, expect, it, vi } from 'vitest';

import { UiSplitButton } from './UiSplitButton';

afterEach(() => {
  cleanup();
  vi.restoreAllMocks();
});

describe('UiSplitButton', () => {
  it('keeps the primary action separate from the dropdown actions', () => {
    const onClick = vi.fn();
    const onOptionSelect = vi.fn();
    render(
      <UiSplitButton
        icon={<Hammer />}
        label="Build"
        menuAriaLabel="Build actions"
        onClick={onClick}
        onOptionSelect={onOptionSelect}
        options={[
          { value: 'build', label: 'Build', icon: <Hammer /> },
          { value: 'rebuild', label: 'Rebuild', icon: <RefreshCw /> },
        ]}
      />,
    );

    fireEvent.click(screen.getByRole('button', { name: 'Build' }));
    expect(onClick).toHaveBeenCalledOnce();
    expect(onOptionSelect).not.toHaveBeenCalled();

    fireEvent.click(screen.getByRole('button', { name: 'Build actions' }));
    expect(screen.getByRole('menu')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('menuitem', { name: /Rebuild/ }));
    expect(onOptionSelect).toHaveBeenCalledWith('rebuild');
  });

  it('shifts a wide menu back inside the viewport', async () => {
    vi.spyOn(HTMLElement.prototype, 'getBoundingClientRect').mockImplementation(function (this: HTMLElement) {
      if (this.classList.contains('ui-split-button-menu'))
        return { left: -90, right: 270, top: 40, bottom: 260, width: 360, height: 220, x: -90, y: 40, toJSON() {} };
      return { left: 100, right: 220, top: 10, bottom: 42, width: 120, height: 32, x: 100, y: 10, toJSON() {} };
    });
    render(
      <UiSplitButton
        label="Build"
        menuAriaLabel="Build actions"
        onClick={vi.fn()}
        onOptionSelect={vi.fn()}
        options={[{ value: 'build', label: 'A very long build operation that must stay visible' }]}
      />,
    );
    fireEvent.click(screen.getByRole('button', { name: 'Build actions' }));
    await waitFor(() => expect(screen.getByRole('menu')).toHaveStyle({ transform: 'translate(98px, 0px)' }));
  });
});
